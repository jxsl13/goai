#!/usr/bin/env python3
"""BackwardGrad tier-1 torch-parity golden (§V16).

Exports the fixture for autograd.Tape.BackwardGrad — the explicit-cotangent
VJP primitive (torch's y.backward(gradient=c), JAX's vjp cotangent) — on a
small composite f64 graph:

    y = tanh(A · B)[1:3, :]        A [4,5], B [5,3], y [2,3]

with a random cotangent c [2,3]. PyTorch computes A.grad and B.grad via
y.backward(gradient=c); the Go side (autograd/backwardgrad_test.go,
TestBackwardGradTorchParity) rebuilds the same graph on a tape, calls
BackwardGrad(y, c), and must match both gradients to ≤ 1e-12 absolute
(pure f64 on both sides — same order-free elementwise math, matmul is the
only reduction and it is tiny).

Writes:
  autograd/testdata/backwardgrad_torch.npz   entries a, b, c, ga, gb (all f8)

Run:
  .venv/bin/python scripts/golden_backwardgrad.py
"""

import os

import numpy as np
import torch

SEED = 3

DEST = os.path.join(
    os.path.dirname(__file__), "..", "autograd", "testdata", "backwardgrad_torch.npz"
)


def main():
    torch.manual_seed(SEED)
    a = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    y = torch.tanh(a @ b)[1:3, :]
    c = torch.randn(y.shape, dtype=torch.float64)
    y.backward(gradient=c)

    np.savez(
        DEST,
        a=a.detach().numpy(),
        b=b.detach().numpy(),
        c=c.numpy(),
        ga=a.grad.numpy(),
        gb=b.grad.numpy(),
    )
    print(f"wrote {os.path.normpath(DEST)}")
    print("a.grad[0]:", a.grad[0].numpy())


if __name__ == "__main__":
    main()
