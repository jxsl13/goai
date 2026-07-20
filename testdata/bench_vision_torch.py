#!/usr/bin/env python3
"""Cross-language performance reference: time a ViT and a CNN image classifier --
forward-only, and a full TRAINING STEP (forward + cross-entropy + backward, NO
optimizer update) -- in PyTorch at the IDENTICAL geometry as
internal/benchcompare/vision_train_test.go's BenchmarkViT*/BenchmarkCNN*, so
GoAI's cpu-simd / Metal / Vulkan vision path can be compared head-to-head with
torch-cpu and torch-mps on the same box (BENCHMARKS.md, T884). Companion to
bench_gpt_train_torch.py (T883).

Geometry (pinned to match the Go benchmark exactly):
  shared:  batch=8, channels=3, size=32, classes=10, f32
  ViT:     patch=4 -> 64 patch tokens + 1 class token, dim=128, depth=4,
           heads=4, MLP 4x (512). Pre-LN encoder: token = linear patch embed,
           prepend a learnable class token, add learned 1D position embeddings;
           per block  x += MHA(LN1(x)); x += MLP(LN2(x))  with a bias-free,
           non-causal MHA and an exact-erf-GELU MLP; final LayerNorm; the class
           token runs through the classification head. ~807,306 params.
  CNN:     channels=[8,16], kernel=3, pad=1. Per stage Conv2d -> ReLU ->
           MaxPool2d(2); then global-average-pool -> Linear head. ~1,562 params.

The two torch nn.Modules FAITHFULLY MIRROR GoAI (vision/vit.go, vision/vision.go):
  - LayerNorm eps=1e-5, learnable gamma/beta (torch default).
  - GELU is EXACT (erf), matching GoAI's OpGELU = 0.5x(1+erf(x/sqrt2)); NOT the
    tanh approximation.
  - MHA is BIAS-FREE and NON-CAUSAL: nn.MultiheadAttention(dim,heads,bias=False)
    gives in_proj_weight [3D,D] (= Wq|Wk|Wv) + out_proj.weight [D,D] (= Wo) and
    NO biases, matching GoAI's four [D,D] projection weights.
  - Conv2d and every Linear carry a bias (GoAI's nn.Conv2D / nn.Linear do).
  - GoAI stores each Linear weight [in,out] and computes x*W (torch stores
    [out,in] and computes x*W^T): same math, same param SHAPE/COUNT, transposed
    layout -- a documented divergence that does not affect FLOPs or the anchor.
  - Patch embedding: GoAI patchifies then applies Linear([C*p*p]->D); here Unfold
    + Linear is the literal mirror (Conv2d(C,D,p,stride=p) is the equivalent
    fusion). The per-patch element ORDER differs but the param shape [48->128]
    and the FLOPs are identical (weights are random and never compared).

Fairness notes: batch=8 exactly as the Go side; the img/s unit is
images-classified-per-second = batch / step_time, identical on both sides. torch
batches the ViT attention ([B,N+1,D] in one pass) whereas GoAI's ViT.Forward
loops over the batch internally -- both classify the same 8 images per step, so
exposing that batching gap is intended, not a handicap. MPS is synchronized
before the timer stops (dispatch is async); the CPU thread count torch chose is
printed so the core budget is on the record. Forward-only is timed under
torch.no_grad() (mirrors GoAI's non-recording forward context); the train step
builds the graph and runs backward but does NOT step an optimizer, exactly like
the Go benchmark. Warm-up excluded; we report the median and the min.

The printed per-model param counts are for the main agent to cross-check against
GoAI's ViT.Params()/CNN.Params() (807306 / 1562).

Usage:
    .venv/bin/python testdata/bench_vision_torch.py
    .venv/bin/python testdata/bench_vision_torch.py --json
"""
import json
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH, CHANS, SIZE, CLASSES = 8, 3, 32, 10
PATCH, DIM, DEPTH, HEADS = 4, 128, 4, 4


class ViTBlock(nn.Module):
    """One pre-LN encoder block: x += MHA(LN1(x)); x += MLP(LN2(x))."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-5)
        # bias=False drops BOTH in_proj_bias and out_proj.bias, leaving
        # in_proj_weight [3*DIM,DIM] and out_proj.weight [DIM,DIM] -> 4*DIM*DIM
        # params, matching GoAI's Wq/Wk/Wv/Wo (each [DIM,DIM], no bias).
        self.attn = nn.MultiheadAttention(DIM, HEADS, bias=False, batch_first=True)
        self.norm2 = nn.LayerNorm(DIM, eps=1e-5)
        # exact GELU (approximate='none' is the default) matches GoAI's OpGELU.
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)  # non-causal (no mask)
        x = x + a
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Faithful mirror of vision.NewViT(3, 32, 10, patch=4, dim=128, depth=4, heads=4)."""

    def __init__(self):
        super().__init__()
        n = (SIZE // PATCH) ** 2               # 64 patch tokens
        patch_dim = CHANS * PATCH * PATCH      # 48
        self.unfold = nn.Unfold(kernel_size=PATCH, stride=PATCH)
        self.embed = nn.Linear(patch_dim, DIM)  # [48 -> 128] + bias
        self.cls = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, DIM))
        self.blocks = nn.ModuleList(ViTBlock() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(DIM, eps=1e-5)
        self.head = nn.Linear(DIM, CLASSES)

    def forward(self, x):
        b = x.size(0)
        patches = self.unfold(x).transpose(1, 2)   # [B, N, C*p*p]
        h = self.embed(patches)                    # [B, N, DIM]
        cls = self.cls.expand(b, -1, -1)           # [B, 1, DIM]
        h = torch.cat([cls, h], dim=1) + self.pos  # [B, N+1, DIM]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(h[:, 0])                  # class token -> [B, CLASSES]


class CNN(nn.Module):
    """Faithful mirror of vision.NewCNN(3, 10, channels=[8,16], kernel=3)."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(CHANS, 8, 3, padding=1)  # + bias (GoAI's Conv2D has bias)
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(16, CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv1(x)))  # 16 -> 8
        x = x.mean(dim=(2, 3))                 # global average pool -> [B, 16]
        return self.head(x)


def nparams(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def timed(step, device: str, reps: int) -> tuple[float, float]:
    """Median and min wall time of `step`, warm-up excluded, MPS synced."""
    for _ in range(3):  # warm-up (excluded)
        step()
    if device == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(reps):
        s = time.perf_counter()
        step()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - s)
    return statistics.median(times), min(times)


def bench(device: str, reps: int = 12) -> dict:
    torch.manual_seed(0)
    x = torch.rand(BATCH, CHANS, SIZE, SIZE, device=device)
    tgt = torch.arange(BATCH, device=device) % CLASSES

    vit = ViT().to(device).train()
    cnn = CNN().to(device).train()

    def vit_forward():
        with torch.no_grad():
            vit(x)

    def cnn_forward():
        with torch.no_grad():
            cnn(x)

    def vit_train():
        vit.zero_grad(set_to_none=True)
        F.cross_entropy(vit(x), tgt).backward()

    def cnn_train():
        cnn.zero_grad(set_to_none=True)
        F.cross_entropy(cnn(x), tgt).backward()

    work = [
        ("vit_forward", vit_forward),
        ("vit_train", vit_train),
        ("cnn_forward", cnn_forward),
        ("cnn_train", cnn_train),
    ]
    res = {}
    for name, step in work:
        med, best = timed(step, device, reps)
        res[name] = {"median_ms": med * 1e3, "min_ms": best * 1e3,
                     "img_s_median": BATCH / med, "img_s_best": BATCH / best}
    return res


def main() -> None:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    params = {"vit": nparams(ViT()), "cnn": nparams(CNN())}
    out = {dev: bench(dev) for dev in devices}

    meta = {
        "torch": torch.__version__, "python": sys.version.split()[0],
        "cpu_threads": torch.get_num_threads(),
        "geometry": (f"batch={BATCH} chans={CHANS} size={SIZE} classes={CLASSES} f32 | "
                     f"ViT(patch={PATCH} dim={DIM} depth={DEPTH} heads={HEADS} mlp={4 * DIM}) | "
                     f"CNN(channels=[8,16] kernel=3)"),
        "params": params,
        "timed": ("forward (no_grad) and train step = forward + cross_entropy + "
                  "backward (no optimizer step)"),
    }
    if "--json" in sys.argv:
        print(json.dumps({"meta": meta, "results": out}, indent=2))
        return
    print(f"torch {meta['torch']}, python {meta['python']}, cpu_threads={meta['cpu_threads']}")
    print(f"geometry: {meta['geometry']}")
    print(f"params: ViT={params['vit']}  CNN={params['cnn']}  "
          f"(cross-check vs GoAI ViT.Params()/CNN.Params(): 807306 / 1562)")
    print(f"timed: {meta['timed']}")
    for dev in devices:
        print(f"-- torch-{dev} --")
        for name, r in out[dev].items():
            print(f"  {name:12}: median {r['median_ms']:8.3f} ms  =>  "
                  f"{r['img_s_median']:8.1f} img/s   (best {r['img_s_best']:.1f})")


if __name__ == "__main__":
    main()
