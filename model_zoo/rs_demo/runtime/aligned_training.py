from __future__ import annotations

from typing import Iterable


def build_dense_stack(torch, input_dim: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


def run_dense_backward(loss, pooled, dense_module, torch, device):
    dense_params = [param for param in dense_module.parameters() if param.requires_grad]
    grads = torch.autograd.grad(
        loss,
        dense_params + [pooled],
        retain_graph=True,
    )
    for param, grad in zip(dense_params, grads[:-1]):
        param.grad = None if grad is None else grad.detach()
    pooled_grad = grads[-1]
    if pooled_grad is None:
        raise RuntimeError("missing pooled gradient after dense backward")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return pooled_grad.detach()
