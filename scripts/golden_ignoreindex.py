#!/usr/bin/env python3
"""CrossEntropy ignore_index + reduction tier-1 torch-parity golden (§V16).

Exports the fixture for nn.CrossEntropy's WithIgnoreIndex / WithReduction options
against torch.nn.functional.cross_entropy, which is definitional here (torch's
`ignore_index=-100` is THE convention every instruction-tuning and masked-LM
pipeline masks prompt/pad positions with).

The fixture sweeps the full cross-product

    reduction      ∈ {mean, sum, none}
    ignore_index   ∈ {off, -100}
    label_smoothing ∈ {0.0, 0.1}

= 12 cases on one shared f64 logits[6,5] / targets[6] pair, where the targets
hold -100 in two rows. The label_smoothing × ignore_index × mean intersection is
the point of the sweep: torch smooths only the NON-ignored rows and divides the
mean by the NON-ignored count, and hand-rolled implementations drift exactly there.

Two extra cases pin the degenerate all-ignored batch (every target -100), whose
behaviour was confirmed empirically rather than assumed: mean → NaN (0/0),
sum → 0, none → all zeros, and zero gradients in all three.

Everything is f64 on both sides, so the Go side matches to ≤ 1e-12 absolute.

Writes:
  nn/testdata/ignoreindex_torch.npz
    z, t                    shared inputs (t as f64 class indices, §B12)
    zall, tall              the all-ignored batch
    loss_<case>             forward loss (scalar, or [batch] for reduction=none)
    grad_<case>             logits.grad [batch,classes]
  where <case> is "<reduction>_<ii|noii>_<ls0|ls01>", plus "all_<reduction>".

Run:
  .venv/bin/python scripts/golden_ignoreindex.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

SEED = 11
IGNORE = -100
BATCH, CLASSES = 6, 5

DEST = os.path.join(
    os.path.dirname(__file__), "..", "nn", "testdata", "ignoreindex_torch.npz"
)


def run(z_np, t_np, reduction, ignore_index, label_smoothing):
    """One forward+backward at f64; returns (loss, grad) as numpy arrays."""
    z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(t_np, dtype=torch.long)
    kwargs = {"reduction": reduction, "label_smoothing": label_smoothing}
    if ignore_index is not None:
        kwargs["ignore_index"] = ignore_index
    loss = F.cross_entropy(z, t, **kwargs)
    # reduction="none" yields a [batch] vector; .sum() seeds every row's cotangent
    # with 1.0, which is exactly the cotangent the Go tape seeds for a vector loss.
    loss.sum().backward()
    return loss.detach().numpy().astype(np.float64), z.grad.numpy()


def main():
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((BATCH, CLASSES))
    t = np.array([1, IGNORE, 3, IGNORE, 0, 2], dtype=np.int64)

    zall = rng.standard_normal((3, CLASSES))
    tall = np.full(3, IGNORE, dtype=np.int64)

    out = {
        "z": z,
        "t": t.astype(np.float64),
        "zall": zall,
        "tall": tall.astype(np.float64),
    }

    for reduction in ("mean", "sum", "none"):
        for ii_tag, ii in (("noii", None), ("ii", IGNORE)):
            for ls_tag, ls in (("ls0", 0.0), ("ls01", 0.1)):
                if ii is None:
                    # Without ignore_index the -100 targets are illegal, so the
                    # "off" arm runs on the hand-filtered (surviving) rows — which
                    # doubles as the reference the masked arm must reproduce.
                    keep = t != IGNORE
                    loss, grad_keep = run(z[keep], t[keep], reduction, None, ls)
                    grad = np.zeros_like(z)
                    grad[keep] = grad_keep
                else:
                    loss, grad = run(z, t, reduction, ii, ls)
                tag = f"{reduction}_{ii_tag}_{ls_tag}"
                out[f"loss_{tag}"] = loss
                out[f"grad_{tag}"] = grad

    for reduction in ("mean", "sum", "none"):
        loss, grad = run(zall, tall, reduction, IGNORE, 0.0)
        out[f"loss_all_{reduction}"] = loss
        out[f"grad_all_{reduction}"] = grad

    os.makedirs(os.path.dirname(os.path.normpath(DEST)), exist_ok=True)
    np.savez(DEST, **out)
    print(f"wrote {os.path.normpath(DEST)} ({len(out)} entries)")


if __name__ == "__main__":
    main()
