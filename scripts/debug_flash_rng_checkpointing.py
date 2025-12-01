import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

"""
Debug script to probe stochastic ProbLoRA-style projections with FlashAttention/SDPA
and gradient checkpointing RNG preservation.

Experiments:
  1) Verify stochasticity with gradient checkpointing enabled (outputs vary across seeds)
  2) Check whether RNG state is preserved during checkpoint recomputation (eps_forward == eps_recompute)
  3) Compare behavior with checkpointing disabled and explicit RNG preservation (validation reference)

Notes:
- We simulate ProbLoRA projections (A produces mu/logvar -> reparameterized z; then z @ B.T)
  without importing the full model. This isolates RNG and attention effects.
- We use torch.scaled_dot_product_attention and request the Flash kernel if available; otherwise
  we fall back to mem_efficient/math kernels. This still tests the same RNG behavior.
"""

# --------------------------- Config ---------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32

BATCH, SEQ, DIM = 2, 16, 64
HEADS = 4
HEAD_DIM = DIM // HEADS
RANK = 8
SCALING = 0.25
LOGVAR_CLAMP_MIN = -5.0
LOGVAR_CLAMP_MAX = 5.0

# For reproducibility across runs
BASE_SEED = 1234

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------- SDPA backend select -------------------
class sdp_select:
    """Context manager to prefer FlashAttention2 if available; fallback otherwise."""
    def __enter__(self):
        self.orig = torch.backends.cuda.sdp_kernel if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            try:
                # Prefer flash if present, else mem_efficient, else math
                torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available() and self.orig is not None:
            # Restore default preference
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)


def current_sdp_backend_name():
    if DEVICE != 'cuda':
        return 'cpu-math'
    # Heuristic: PyTorch chooses kernel at runtime; we can at least report capability
    flags = torch.backends.cuda.sdp_kernel
    parts = []
    if getattr(flags, 'is_flash_enabled', lambda: False)():
        parts.append('flash')
    if getattr(flags, 'is_mem_efficient_enabled', lambda: False)():
        parts.append('mem_efficient')
    if getattr(flags, 'is_math_enabled', lambda: False)():
        parts.append('math')
    return '+'.join(parts) or 'unknown'

# ------------------ ProbLoRA-style projection -----------------
class ProbLoRAProj(nn.Module):
    """Minimal ProbLoRA-like projection: (W + ΔW)x where ΔW = B @ A(ε) and
    A produces (mu, logvar) that are linearly dependent on x.

    We capture the epsilon used on each forward to test RNG preservation.
    """
    def __init__(self, in_features, out_features, rank=RANK, scaling=SCALING):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.base.weight)

        # A parameters packed: [2*rank, in_features] => split into mu_A, logvar_A
        self.A = nn.Parameter(torch.empty(2*rank, in_features))
        nn.init.xavier_uniform_(self.A)

        # B: [out_features, rank]
        self.B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.xavier_uniform_(self.B)

        self.rank = rank
        self.scaling = scaling

        # For debug capture
        self.last_eps = None
        self.last_mu = None
        self.last_logvar = None

    def forward(self, x):
        # base projection
        base_out = self.base(x)

        # compute mu/logvar from packed A
        A = self.A.to(dtype=torch.float32)
        x32 = x.to(dtype=torch.float32)
        mu_A, logvar_A = torch.split(A, self.rank, dim=0)

        # [B,S,rank]
        BS = x32.shape[0] * x32.shape[1]
        x_flat = x32.reshape(BS, x32.shape[-1])
        mu = x_flat @ mu_A.T
        logvar = x_flat @ logvar_A.T
        logvar = logvar.clamp(LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        if self.training and torch.is_grad_enabled():
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
            self.last_eps = eps.detach()
        else:
            z = mu
            self.last_eps = None

        out32 = z @ self.B.to(dtype=torch.float32).T
        out = out32.reshape(x32.size(0), x32.size(1), -1).to(x.dtype)

        # Save for inspection
        self.last_mu = mu.detach()
        self.last_logvar = logvar.detach()

        return base_out + self.scaling * out

# -------------------------- Tiny MHA --------------------------
class TinyAttention(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.q = ProbLoRAProj(dim, dim)
        self.k = ProbLoRAProj(dim, dim)
        self.v = ProbLoRAProj(dim, dim)
        self.o = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.o.weight)

    def _shape(self, x):
        B, S, D = x.shape
        x = x.view(B, S, self.heads, self.head_dim).transpose(1, 2)  # [B,H,S,Hd]
        return x

    def forward(self, x, attn_mask=None):
        q = self._shape(self.q(x))
        k = self._shape(self.k(x))
        v = self._shape(self.v(x))
        with sdp_select():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.o(out)

# ----------------------- Experiment logic ---------------------

def run_with_checkpoint(model, x, use_reentrant=False, preserve_rng_state=True):
    """Run forward under checkpoint; capture eps from forward-build and from recompute.

    We record eps before calling backward (from graph-building forward), and again
    after backward (from recompute forward). This avoids relying on Python-side
    list mutation during autograd recompute, which may not always be reliable.
    """
    model.train()

    def block(inp):
        y = model(inp)
        return y

    x = x.detach().requires_grad_(True)
    y = checkpoint(block, x, use_reentrant=use_reentrant, preserve_rng_state=preserve_rng_state)

    # Capture eps from the graph-building forward pass
    fwd_eps = (
        model.q.last_eps.clone() if model.q.last_eps is not None else None,
        model.k.last_eps.clone() if model.k.last_eps is not None else None,
        model.v.last_eps.clone() if model.v.last_eps is not None else None,
    )

    loss = y.sum()
    loss.backward()

    # After backward, checkpoint has recomputed the forward; capture recompute eps
    rec_eps = (
        model.q.last_eps.clone() if model.q.last_eps is not None else None,
        model.k.last_eps.clone() if model.k.last_eps is not None else None,
        model.v.last_eps.clone() if model.v.last_eps is not None else None,
    )

    return fwd_eps, rec_eps, y.detach()


def run_without_checkpoint(model, x):
    model.train()

    def block(inp):
        y = model(inp)
        return y

    x = x.detach().requires_grad_(True)
    y = block(x)

    eps_tuple = (
        model.q.last_eps.clone() if model.q.last_eps is not None else None,
        model.k.last_eps.clone() if model.k.last_eps is not None else None,
        model.v.last_eps.clone() if model.v.last_eps is not None else None,
    )

    loss = y.sum()
    loss.backward()
    return eps_tuple, y.detach()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return float('nan')
    a = a.flatten().float()
    b = b.flatten().float()
    return F.cosine_similarity(a, b, dim=0).item()


def main():
    print(f"Device: {DEVICE}, DType: {DTYPE}, SDPA preference: {current_sdp_backend_name()}")

    # Create model and data
    set_seed(BASE_SEED)
    model = TinyAttention().to(DEVICE, dtype=DTYPE)
    x = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=DTYPE)

    # 1) Stochasticity with checkpointing enabled
    print("\n[1] Stochasticity with checkpointing enabled (use_reentrant=False, preserve_rng_state=True)")
    set_seed(BASE_SEED)
    _, _, y1 = run_with_checkpoint(model, x, use_reentrant=False, preserve_rng_state=True)
    set_seed(BASE_SEED + 1)
    _, _, y2 = run_with_checkpoint(model, x, use_reentrant=False, preserve_rng_state=True)

    diff_y = (y1 - y2).abs().mean().item()
    print(f"Mean |y(seed={BASE_SEED}) - y(seed={BASE_SEED+1})|: {diff_y:.6f}")

    # 2) RNG preservation under checkpointing (forward vs recompute)
    print("\n[2] RNG preservation under checkpointing (forward vs recompute)")
    set_seed(BASE_SEED)
    fwd_eps, rec_eps, _ = run_with_checkpoint(model, x, use_reentrant=False, preserve_rng_state=True)
    sims = [cosine_sim(fwd_eps[i], rec_eps[i]) for i in range(3)]
    print(f"Cosine sim eps (q,k,v) forward vs recompute: {sims}")

    # 3) No checkpointing + explicit RNG preservation
    print("\n[3] No checkpointing + explicit RNG preservation (validation)")
    # Save RNG state BEFORE first run
    set_seed(BASE_SEED)
    cpu_state_before = torch.get_rng_state()
    cuda_state_before = torch.cuda.get_rng_state() if DEVICE == 'cuda' else None

    eps_nc_1, y_nc1 = run_without_checkpoint(model, x)

    # Restore the exact pre-run RNG state and run again
    torch.set_rng_state(cpu_state_before)
    if DEVICE == 'cuda' and cuda_state_before is not None:
        torch.cuda.set_rng_state(cuda_state_before)

    eps_nc_2, y_nc2 = run_without_checkpoint(model, x)

    sims_nc = [cosine_sim(eps_nc_1[i], eps_nc_2[i]) for i in range(3)]
    diff_nc = (y_nc1 - y_nc2).abs().mean().item()

    print(f"Cosine sim eps (q,k,v) no-ckpt forward1 vs forward2 with restored RNG: {sims_nc}")
    print(f"Mean |y_no_ckpt(run1) - y_no_ckpt(run2)|: {diff_nc:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
