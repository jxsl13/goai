#!/usr/bin/env python3
"""CPU norm/softmax companion: time torch (oneDNN) and numpy on the EXACT same
shapes as GoAI's backend/cpu f32 forward benchmarks, so the pure-Go AVX2 kernels
can be compared apples-to-apples against the industry incumbents on THIS machine.

Motivation: the in-repo CPU-vs-accelerator harness (internal/benchcompare/
compare_test.go) is `//go:build darwin && cgo && vulkan` — macOS-only. Per the
Nx-XPLAT rule a macOS-only-benchmarked gap must be verified on the current system,
so this Linux-runnable companion pins the CPU incumbent numbers here.

Fairness contract:
  * identical shapes  -- softmax {32x2048, 1x32000, 4x32000} matches
    backend/cpu.BenchmarkSoftmaxF32_{32x2048,1x32000,4x32000}_cpu; layernorm /
    rmsnorm 512x1024 matches BenchmarkLayerNormFwdF32_512x1024 /
    BenchmarkRMSNormFwdF32_512x1024.
  * same op semantics -- softmax over the last axis; layernorm over the last axis
    with affine (gamma, beta); rmsnorm = x * rsqrt(mean(x^2)+eps) * gamma. eps 1e-5.
  * same threading    -- torch.set_num_threads matches GOMAXPROCS on the Go side
    (both saturate the box); f32 throughout.
  * NOTE torch's caching allocator reuses the output buffer across iterations while
    GoAI's tensor.NewOn allocates + zeros a fresh output every call — part of the
    remaining gap is allocator, not kernel. Read the ratios with that in mind.

The Go side (run with the AVX2 kernels engaged — the gate is goexperiment.simd, NOT
-tags simd):
  GOEXPERIMENT=simd go test ./backend/cpu/ -run '^$' -benchmem \
    -bench 'BenchmarkSoftmaxF32_(32x2048|1x32000|4x32000)_cpu|Bench(Layer|RMS)NormFwdF32_512x1024'

Run:   .venv/bin/python internal/benchcompare/norm_compare.py
Needs: torch, numpy (the goai-vllm venv already has both).

Reference (RTX3060 box, AVX2+FMA no AVX512, torch 2.11 / numpy 2.3; GoAI is
GOEXPERIMENT=simd min-of-6, post the norm/softmax AVX2 work). torch's per-call times
are NOISY run-to-run (±~2x on the small shapes — thread pool warmup / scheduling), so
these are representative ranges, not fixed points; numpy and GoAI are stable. The
DIRECTION is robust: torch leads on every shape, GoAI is closest on softmax 1x32000
(~2x) and beats numpy on the wide softmaxes.
  softmax 32x2048 : torch 10-20   numpy ~120  GoAI 57.5
  softmax 1x32000 : torch ~24     numpy ~46   GoAI 47.4   (GoAI ~= numpy, ~2x torch)
  softmax 4x32000 : torch 30-60   numpy ~200  GoAI 120    (GoAI 1.7x faster than numpy)
  layernorm 512x1024 : torch 30-43            GoAI 220    (~5-7x torch)
  rmsnorm  512x1024  : (torch has no fused rmsnorm; the manual expr is ~1000us) GoAI 182
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(16)
torch.manual_seed(0)


def bench(fn, iters=200, warm=20):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1e6  # us/op


def show(name, us):
    print(f"{name:26s} {us:9.1f} us/op")


def main():
    print(f"# torch {torch.__version__}, numpy {np.__version__}, "
          f"threads {torch.get_num_threads()}")
    for shape in [(32, 2048), (1, 32000), (4, 32000)]:
        x = torch.randn(*shape)
        show(f"softmax torch {shape}", bench(lambda: torch.softmax(x, dim=-1)))
        xn = x.numpy()

        def npsm():
            m = xn.max(axis=-1, keepdims=True)
            e = np.exp(xn - m)
            return e / e.sum(axis=-1, keepdims=True)

        show(f"softmax numpy {shape}", bench(npsm))
    for shape in [(512, 1024)]:
        x = torch.randn(*shape)
        g = torch.randn(shape[-1])
        b = torch.randn(shape[-1])
        show(f"layernorm torch {shape}", bench(lambda: F.layer_norm(x, (shape[-1],), g, b, 1e-5)))

        def rms():
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g

        show(f"rmsnorm torch {shape}", bench(rms))


if __name__ == "__main__":
    main()
