"""
Preflight sanity check for bf16 + gradient checkpointing.

Usage (recommended):
- Import and call from your training script right AFTER you construct `training_args`, `model`,
  and `train_dataloader`, but BEFORE you start training:

    from scripts.preflight_bf16_checkpoint import preflight_check
    preflight_check(training_args, model, train_dataloader)

This module intentionally does not try to construct a model/training_args for you. It validates
that the runtime has been configured correctly and runs a tiny forward/backward to ensure
checkpointing + bf16 + LoRA/ProbLoRA gradient flow are sane.

If you prefer a one-off script to run in a console, import this module and call the function with
objects from your training environment.
"""

from typing import Any
import warnings
import torch


def preflight_check(training_args: Any, model: torch.nn.Module, train_dataloader) -> None:
    """Run preflight checks to validate bf16 + gradient checkpointing setup.

    Args:
        training_args: object with attributes bf16, fp16, gradient_checkpointing
        model: the PEFT/ProbLoRA-wrapped model
        train_dataloader: a DataLoader yielding batches suitable for model(**batch)

    Raises:
        AssertionError on any failed check.
    """

    # 0) Make sure these flags are actually set
    assert getattr(training_args, 'bf16', False) is True and getattr(training_args, 'fp16', False) is False, \
        "Use bf16=True, fp16=False"
    assert getattr(training_args, 'gradient_checkpointing', False) is True, "Enable gradient_checkpointing=True"
    assert getattr(getattr(model, 'config', None), 'use_cache', False) is False, "Set model.config.use_cache=False"

    # 1) Enable checkpointing & ensure inputs to first block require grad
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            # Some model wrappers may not implement this; ignore if not available
            pass

    # Ensure use_cache disabled
    try:
        model.config.use_cache = False
    except Exception:
        pass

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            # ignore if this fails
            pass
    else:
        # Fallback hook (older transformers / custom models)
        def _embed_out_requires_grad(module, inp, out):
            try:
                if isinstance(out, torch.Tensor) and not out.requires_grad:
                    out.requires_grad_(True)
            except Exception:
                pass

        if hasattr(model, "get_input_embeddings"):
            try:
                model.get_input_embeddings().register_forward_hook(_embed_out_requires_grad)
            except Exception:
                pass

    # 2) (Optional) QLoRA compute dtype should match bf16
    qc = getattr(model, "quantization_config", None)
    if qc is not None and hasattr(qc, "bnb_4bit_compute_dtype"):
        cur = getattr(qc, "bnb_4bit_compute_dtype")
        # Accept both dtype object and string-like
        cur_s = str(cur)
        assert cur_s in {"torch.bfloat16", "bfloat16"}, \
            f"bnb_4bit_compute_dtype is {cur}; set to torch.bfloat16"

    # 3) ProbLoRA/LoRA masters should be FP32 (if you follow the FP32-master pattern)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.float32, f"{n} should be FP32 master param (found {p.dtype})"

    # 4) Tiny forward/backward under warnings capture (uses one batch from your train dataloader)
    device = next(p.device for p in model.parameters())
    batch = next(iter(train_dataloader))
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    model.train()
    # Probe: ensure the first checkpointed module receives an input tensor with requires_grad=True
    flag = {"ok": False}
    def _probe(_module, inputs):
        try:
            if inputs and isinstance(inputs[0], torch.Tensor) and inputs[0].requires_grad:
                flag["ok"] = True
        except Exception:
            pass
    h = None
    try:
        # Find the module that is actually checkpointed. Common pattern: layer.self_attn.o_proj
        first_layer = None
        try:
            first_layer = getattr(getattr(model, "model", model), "layers")[0]
        except Exception:
            first_layer = None

        target_mod = None
        if first_layer is not None:
            # Prefer the common o_proj target on self_attn
            try:
                sa = getattr(first_layer, 'self_attn', None)
                if sa is not None and hasattr(sa, 'o_proj'):
                    target_mod = sa.o_proj
            except Exception:
                target_mod = None

            # Fallback: find the first child module that is likely checkpointed
            if target_mod is None:
                try:
                    for name, m in first_layer.named_modules():
                        if m is not first_layer and isinstance(m, torch.nn.Module):
                            target_mod = m
                            break
                except Exception:
                    target_mod = None

        if target_mod is not None:
            try:
                h = target_mod.register_forward_pre_hook(_probe)
                # Run a single forward to exercise the probe
                _ = model(**batch)
            except Exception:
                pass
    finally:
        try:
            if h is not None:
                h.remove()
        except Exception:
            pass

    assert flag["ok"], "Checkpointed block did not receive a tensor with requires_grad=True"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        out = model(**batch)
        # Support both dict-like outputs and objects with .loss
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        loss.backward()

        # If running on CUDA with bf16 enabled, logits should be bfloat16
        # Only assert dtype when model is on CUDA (CPU demos often run in float32)
        if getattr(training_args, "bf16", False) and device.type == 'cuda':
            assert out.logits.dtype == torch.bfloat16, f"Expected bf16 logits on CUDA, got {out.logits.dtype}"

        # No checkpoint warning: "None of the inputs have requires_grad=True"
        msgs = [str(x.message) for x in w if "requires_grad=True" in str(x.message)]
        assert not msgs, f"Checkpoint input-grad warning detected: {msgs}"

    # 5) Ensure we actually got gradients on trainable params
    has_grads = any(p.requires_grad and p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients found on trainable parameters"

    # Check that at least one trainable grad is non-zero (not all-zero)
    nonzero = any(
        p.requires_grad and p.grad is not None and p.grad.detach().abs().sum() > 0
        for p in model.parameters()
    )
    assert nonzero, "All trainable grads are zero — check loss path/masking"

    # Check gradients are finite (no NaNs or Infs)
    finite = all(
        (p.grad is None) or torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    assert finite, "Found non-finite gradients"

    # 6) Clean up grads before real training
    model.zero_grad(set_to_none=True)
    print("✅ Preflight passed: bf16+checkpointing configured; no warnings; grads flowing.")


if __name__ == '__main__':
    print("This module provides `preflight_check(training_args, model, train_dataloader)`.\n"
          "Run it from your training script where those objects exist.")

    # Demo runner: convenience code to run a tiny preflight locally. This mirrors the
    # invocation used in our tests and is safe for quick sanity checks.
    try:
        import sys
        sys.path.append('c:/Users/suroj/Documents/Research/GitHubProjects/ARD-LoRA-Data-CLM')
        from types import SimpleNamespace

        # Import the tiny test model/dataset used for unit tests
        import tests.test_checkpoint_sanity as tc

        args = SimpleNamespace(bf16=True, fp16=False, gradient_checkpointing=True)
        model = tc.LMWrapper(vocab_size=64, dim=32, n_layers=2)
        model = tc.inject_problora_llama(model, rank=4, scaling=1.0, num_tokens=16, ard_prior_samples=4)
        # Ensure compatibility with embedding hook fallback
        try:
            model.get_input_embeddings = lambda: model.model.embed_tokens
        except Exception:
            pass

        train_loader = __import__('torch').utils.data.DataLoader(
            tc.DummySeqDataset(num_samples=8, seq_len=6, vocab_size=64), batch_size=2
        )

        preflight_check(args, model, train_loader)
        print('PREFLIGHT_OK')
    except Exception as _e:
        import traceback
        print('Preflight demo failed:')
        traceback.print_exc()
        raise
