#!/usr/bin/env python3
"""Gradient-hook tier-1 torch-parity golden (§V16).

Exports the fixture for autograd.Tape.RegisterGradHook — the per-tensor
gradient hook (torch's tensor.register_hook counterpart) — on a small
composite f64 graph with a fan-out point:

    P = A · B          A [4,5] leaf, B [5,3] leaf, P [4,3]
    y = tanh(P) + P    (P consumed twice — fan-out)
    loss = sum(y)

with two hooks:
    P.register_hook(lambda g: g * 0.5)            # mid-tensor: scale the grad
    A.register_hook(lambda g: g.clamp(-0.1, 0.1)) # leaf: per-tensor clipping

Torch fires the P hook once with the SUMMED cotangent from both consumers,
before the matmul backward consumes it (so both A.grad and B.grad see the
scaling), and the A hook with A's final accumulated grad. The Go side
(autograd/hooks_test.go, TestGradHookTorchParity) rebuilds the same graph on
a tape, registers the same hooks, and must match both gradients to ≤ 1e-12
absolute (pure f64 both sides; the tiny matmul is the only reduction).

Writes:
  autograd/testdata/gradhook_torch.npz   entries a, b, ga, gb (all f8)

Run:
  .venv/bin/python scripts/golden_gradhook.py
"""

import os

import numpy as np
import torch

SEED = 7

DEST = os.path.join(
    os.path.dirname(__file__), "..", "autograd", "testdata", "gradhook_torch.npz"
)


def main():
    torch.manual_seed(SEED)
    a = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

    p = a @ b
    p.register_hook(lambda g: g * 0.5)
    a.register_hook(lambda g: g.clamp(-0.1, 0.1))
    y = torch.tanh(p) + p
    y.sum().backward()

    np.savez(
        DEST,
        a=a.detach().numpy(),
        b=b.detach().numpy(),
        ga=a.grad.numpy(),
        gb=b.grad.numpy(),
    )
    print(f"wrote {os.path.normpath(DEST)}")
    print("a.grad[0]:", a.grad[0].numpy())
    print("b.grad[0]:", b.grad[0].numpy())


if __name__ == "__main__":
    main()
