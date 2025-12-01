import os
import sys
import torch
from torch import nn

# Ensure repo root is on sys.path so 'model' package can be imported when running
# the test directly with the venv python from any working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_llama import inject_problora_llama, collect_problora_layers, ProbLoRALayer


class DummyAttention:
    def __init__(self, in_features, out_features):
        # create linear modules similar to HF attention projections
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.o_proj = nn.Linear(in_features, out_features)


class DummyLayer:
    def __init__(self, in_features, out_features):
        self.self_attn = DummyAttention(in_features, out_features)


class DummyModel:
    def __init__(self, n_layers, in_features, out_features):
        class Container:
            pass
        self.model = Container()
        # simple list of layers
        self.model.layers = [DummyLayer(in_features, out_features) for _ in range(n_layers)]


def test_inject_problora_and_grads():
    # Build a tiny dummy model
    in_features = 16
    out_features = 12
    n_layers = 4
    model = DummyModel(n_layers, in_features, out_features)

    # Inject ProbLoRA (small rank to keep things light)
    inject_problora_llama(model, rank=4, scaling=1.0, num_tokens=32, ard_prior_samples=10)

    # Collect injected layers
    layers = collect_problora_layers(model)
    assert len(layers) > 0, "No ProbLoRA layers collected after injection"

    # Prepare a small batch tensor matching ProbLoRA expected input shape: [B, S, in_features]
    B, S = 2, 3
    x = torch.randn(B, S, in_features, requires_grad=False)

    # Take the first injected layer and run forward/backward
    name, layer = layers[0]
    assert isinstance(layer, ProbLoRALayer)

    # forward
    out = layer(x)
    assert out.shape == (B, S, layer.out_features)

    # create a simple scalar loss and backprop
    loss = out.float().sum()
    loss.backward()

    # A and B should have gradients, base_proj weights should not (frozen)
    assert layer.A.grad is not None, "A.grad is None - ProbLoRA A didn't receive gradients"
    assert layer.B.grad is not None, "B.grad is None - ProbLoRA B didn't receive gradients"

    base_w = getattr(layer.base_proj, 'weight', None)
    if base_w is not None:
        # base_proj should be frozen; its grad should be None
        assert base_w.grad is None, "base_proj.weight.grad is not None (base should be frozen)"

    # basic sanity: kl_divergence_latent runs without error
    kld = layer.kl_divergence_latent(x)
    assert torch.isfinite(kld), "KL divergence returned non-finite value"


if __name__ == '__main__':
    test_inject_problora_and_grads()
    print('test passed')
