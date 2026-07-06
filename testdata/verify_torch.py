#!/usr/bin/env python3
"""Verify optimizer goldens against real torch (resolves §B20).

Runs torch.optim.SGD / Adam in float64 over the exact golden gradient sequence
and compares each step to nn/testdata/optim.json. Exit 1 on mismatch.
"""
import json
import os
import sys

import torch


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "..", "nn", "testdata", "optim.json")) as f:
        g = json.load(f)["optim"]

    p0 = torch.tensor(g["p0"], dtype=torch.float64)
    grads = [torch.tensor(gr, dtype=torch.float64) for gr in g["grads"]]
    failures = 0

    def run(opt_name, make_opt, steps_key):
        nonlocal failures
        p = p0.clone().requires_grad_(True)
        opt = make_opt([p])
        for step, gr in enumerate(grads):
            opt.zero_grad()
            p.grad = gr.clone()
            opt.step()
            want = torch.tensor(g[steps_key]["steps"][step], dtype=torch.float64)
            err = (p.detach() - want).abs().max().item()
            if err > 1e-12:
                print(f"MISMATCH {opt_name} step {step}: max abs err {err:.3e}")
                failures += 1
            else:
                print(f"ok {opt_name} step {step}: max abs err {err:.3e}")

    run("sgd", lambda ps: torch.optim.SGD(ps, lr=g["sgd"]["lr"]), "sgd")
    run("sgd_momentum",
        lambda ps: torch.optim.SGD(ps, lr=g["sgd_momentum"]["lr"],
                                   momentum=g["sgd_momentum"]["momentum"]),
        "sgd_momentum")
    run("adam",
        lambda ps: torch.optim.Adam(ps, lr=g["adam"]["lr"],
                                    betas=(g["adam"]["beta1"], g["adam"]["beta2"]),
                                    eps=g["adam"]["eps"]),
        "adam")
    run("adamw",
        lambda ps: torch.optim.AdamW(ps, lr=g["adamw"]["lr"],
                                     betas=(g["adamw"]["beta1"], g["adamw"]["beta2"]),
                                     eps=g["adamw"]["eps"], weight_decay=g["adamw"]["wd"]),
        "adamw")

    if failures:
        print(f"{failures} mismatches — goldens do NOT match real torch")
        return 1
    print(f"all goldens match real torch {torch.__version__} at 1e-12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
