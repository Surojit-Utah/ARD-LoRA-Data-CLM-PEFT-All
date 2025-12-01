import sys
import torch
from pathlib import Path
import torch
from torch import nn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.model_llama import ProbLoRALayer

# Use your ProbLoRALayer and _fp32_ctx definitions here
def make_test_layer(d_in=4096, d_out=4096, rank=32, deterministic=True):
    base = nn.Linear(d_in, d_out, bias=False)
    layer = ProbLoRALayer(
        base_proj=base,
        rank=rank,
        num_tokens=1000,
        ard_prior_samples=10,
        scaling=1.0,
        logvar_clamp_min=-10.0,
        logvar_clamp_max=10.0,
        beta_logvar_clamp_min=-10.0,
        beta_logvar_clamp_max=10.0,
        sample_clamp_min=-5.0,
        sample_clamp_max=5.0,
        deterministic=deterministic,
        enable_clamps=False,
        lora_alpha=None,
    )
    return layer

layer = make_test_layer(d_in=16, d_out=8, rank=4, deterministic=True)
layer.train()

x = torch.randn(2, 3, 16)             # [B, S, d_in]
out = layer(x)                         # [B, S, d_out]
target = torch.zeros_like(out)
loss = ((out - target)**2).mean()      # simple MSE
loss.backward()

print("mu_A grad:", None if not hasattr(layer, "mu_A") or layer.mu_A.grad is None
      else layer.mu_A.grad.norm().item())
print("B grad:", None if layer.B.grad is None else layer.B.grad.norm().item())
