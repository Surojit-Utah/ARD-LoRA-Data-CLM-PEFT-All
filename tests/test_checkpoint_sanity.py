import warnings
import torch
from torch import nn
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset

from model.model_llama import inject_problora_llama


class DummySeqDataset(Dataset):
    def __init__(self, num_samples=8, seq_len=6, vocab_size=64):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inp = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {
            "input_ids": inp,
            "labels": inp.clone()
        }


class DummySelfAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)


class DummyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = DummySelfAttn(dim)


class InnerModel(nn.Module):
    def __init__(self, vocab_size=64, dim=32, n_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([DummyLayer(dim) for _ in range(n_layers)])


class LMWrapper(nn.Module):
    def __init__(self, vocab_size=64, dim=32, n_layers=2):
        super().__init__()
        self.model = InnerModel(vocab_size=vocab_size, dim=dim, n_layers=n_layers)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # input_ids: [S] or [B, S]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, S = input_ids.shape
        x = self.model.embed_tokens(input_ids)  # [B, S, dim]

        # Simple transformer-like pass: apply each layer's o_proj via checkpoint
        for lyr in self.model.layers:
            # checkpoint expects callables and Tensors as inputs
            x = torch.utils.checkpoint.checkpoint(lyr.self_attn.o_proj, x)

        logits = self.lm_head(x)  # [B, S, V]

        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            # causal shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return SimpleNamespace(loss=loss, logits=logits)

        return SimpleNamespace(loss=None, logits=logits)


def register_checkpoint_hooks(model):
    # replicate the small hooks we added to trainer
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            def _embed_forward_hook(module, inp, out):
                try:
                    if isinstance(out, torch.Tensor):
                        out.requires_grad_(True)
                except Exception:
                    pass
            model.model.embed_tokens.register_forward_hook(_embed_forward_hook)

        if hasattr(model.model, 'layers'):
            def _make_pre_hook():
                def _pre_hook(module, inputs):
                    try:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            if not inputs[0].requires_grad:
                                inputs[0].requires_grad_(True)
                    except Exception:
                        pass
                return _pre_hook

            for lyr in model.model.layers:
                try:
                    lyr.register_forward_pre_hook(_make_pre_hook())
                except Exception:
                    continue
    except Exception:
        pass


def test_checkpoint_sanity():
    device = torch.device('cpu')

    # Build tiny model and inject ProbLoRA (wraps linear layers)
    model = LMWrapper(vocab_size=64, dim=32, n_layers=2)
    model = inject_problora_llama(model, rank=4, scaling=1.0, num_tokens=16, ard_prior_samples=4)
    model.to(device)

    # Register hooks (same logic as trainer)
    register_checkpoint_hooks(model)

    # Data loader
    ds = DummySeqDataset(num_samples=8, seq_len=6, vocab_size=64)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)

    model.train()

    batch = next(iter(train_loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', category=UserWarning)

        out = model(**batch)
        loss = out.loss if isinstance(out, SimpleNamespace) else out['loss']
        loss.backward()

        checkpoint_warnings = [str(x.message) for x in w if 'None of the inputs have requires_grad=True' in str(x.message)]

    # Assert no checkpoint warning
    assert len(checkpoint_warnings) == 0, f"Found checkpoint warnings: {checkpoint_warnings}"

    # Check some gradients exist (LoRA params should have grads)
    grad_count = sum(1 for n, p in model.named_parameters() if p.requires_grad and p.grad is not None)
    assert grad_count > 0, "No parameters had gradients after backward"


if __name__ == '__main__':
    test_checkpoint_sanity()
    print('test passed')
