# CUDA backend — first validation on real NVIDIA hardware (§T42)

> **In plain terms:** GoAI has always had a CUDA/cuBLAS GPU backend
> (`backend/cuda`), but it was written on an arm64 Mac with no NVIDIA GPU, so
> it had **never actually been built or run** — only reasoned about. This page
> records the first time it compiled and passed its tests on a real GPU, how to
> reproduce that, and the measured cuBLAS GEMM throughput.

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB, Ampere, compute capability **8.6**) |
| Driver | 610.43.02 (ships `libcuda.so`) |
| Toolkit | CUDA **12.9** — user-space **pip wheels**, no system install |
| Go / cc | 1.26.5 `linux/amd64` / gcc 16.1.1 |

The machine is an immutable Fedora derivative (bazzite), so the CUDA toolkit is
installed **from pip wheels into a venv** rather than via the system package
manager. The backend is pure cgo + cuBLAS (no `.cu` kernels), so it needs
headers + libraries only — not `nvcc`.

## Build & run (reproducible)

```sh
python3 -m venv .venv-cuda
.venv-cuda/bin/pip install nvidia-cuda-nvcc-cu12 nvidia-cublas-cu12 \
                           nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
source scripts/cuda-pip-env.sh          # sets CGO_CFLAGS/LDFLAGS + rpath, symlinks
go test -tags cuda ./backend/cuda/
```

`scripts/cuda-pip-env.sh` points cgo at the wheel `include`/`lib` dirs, makes
the unversioned `libcublas.so`/`libcudart.so` symlinks cgo's `-l` flags expect
(wheels ship `libcublas.so.12`), and sets an rpath so the test binary loads
them. The `nvidia-cuda-nvcc` wheel supplies `crt/host_config.h` (pulled in by
`cuda_runtime.h`); `cublas` pulls in `nvrtc`.

## Results — all green on the RTX 3060

```
TestCUDACrossReference   PASS   cuBLAS Sgemm == Pure-Go ref across shapes
TestCUDARectangular      PASS   non-square M,K,N
TestCUDAFallback         PASS   non-matmul ops fall back to Pure-Go (§I4)
TestCUDAGPUTraining      PASS   GPU training converged: gpu loss 0.0371 == cpu 0.0371
TestCUDAServesMatmul     PASS
```

Training through the GPU works: `autograd.NewTapeOn(cudaBackend)` dispatches the
forward/backward matmuls to cuBLAS and the model converges to the same loss as
the CPU backend.

## cuBLAS GEMM throughput (§V22, corrected §C3 gate)

f32 square GEMM, `-tags cuda`, this host:

| size | cuda (cuBLAS) | cpu (Pure-Go) | speedup |
|------|---------------|---------------|---------|
| 512³  | 376 GFLOP/s | 42.1 GFLOP/s | **8.9×** |
| 1024³ | 808 GFLOP/s | 42.2 GFLOP/s | **19×** |

> **Bug fixed here:** `BenchmarkMatMulF32_1024_cpu` called `backend.CUDA`
> (copy-paste), so the 1024³ "cpu" row was silently cuBLAS-vs-cuBLAS — the
> §C3 gate's whole point is cuda-vs-cpu. Only running the suite on real
> hardware surfaced it (`benchMatMulOn(b, backend.CPU, 1024)` now).

## Optimizing the memory layer (bottom-up)

808 GFLOP/s at 1024³ is far below the RTX 3060's ~12.7 TFLOP/s f32 peak: the
backend is **alloc + transfer-bound**, not compute-bound. Every `matmulF32`
call spent time on three `cudaMalloc` + three `cudaFree` and ~12 MB of
round-trip PCIe traffic to hide ~2 GFLOP of compute.

### Step 1 — device-buffer pool (landed)

`cudaMalloc`/`cudaFree` cost ~10–100 µs each, so three of each per matmul
dominated small/medium GEMMs. `cuda_bridge.c` now **pools** the device buffers:
`gA/gB/gC` persist across calls and grow to the largest `M*K`/`K*N`/`M*N` seen —
a steady training loop with fixed shapes allocates once. A single mutex
serializes the matmul, so the shared cuBLAS handle and buffers are safe under
concurrent callers (the previous global handle was unguarded).

A/B (§V22, same host, file-toggle baseline, count=2 medians):

| size | malloc-per-call | pooled | speedup |
|------|-----------------|--------|---------|
| 512³  | 374 GFLOP/s | 464 GFLOP/s  | **1.24×** |
| 1024³ | 813 GFLOP/s | 1045 GFLOP/s | **1.29×** |

Bit-exactness is unchanged (cuBLAS result identical; the pool only changes where
the buffers come from) — the cross-reference suite stays green.

### Step 2 — device-resident weights (landed, §V14 Phase-1)

The remaining overhead is the **H2D/D2H copies themselves**. The first piece of
device residency (mirroring the metal §T156 resident-weight seed) is a
**resident weight**: `cuda.NewResidentB(w)` uploads a matrix to the GPU once and
`(*ResidentB).MatMul(a)` reuses it across many activations, skipping the weight's
per-call H2D. It is the inference lever — the weight is fixed, only the
activation varies. Result is *identical* to the per-call cuBLAS matmul (same
Sgemm); `cuda_resident_test.go` checks that plus ref tolerance, reuse, and
use-after-free.

A/B (§V22, RTX 3060, resident vs per-call re-upload of the same weight):

| shape | per-call | resident | speedup |
|-------|----------|----------|---------|
| 1024³ square         | 2.07 ms | 1.65 ms | 1.26× |
| 2048³ square         | 10.0 ms | 7.95 ms | 1.26× |
| **decode** M=8, K=N=4096 | **7.81 ms** | **0.30 ms** | **26×** |

Square GEMM sees a modest ≈1.26× (the weight upload is a fraction of the
compute), but the real inference shape — a small activation against a large
reused weight — is transfer-*dominated*, and skipping the 64 MB weight re-upload
per call is **26×**. This is the concrete payoff of keeping weights on-device.

### Step 3 — full device residency (next)

`ResidentB` keeps *weights* resident; the next step keeps *activations* resident
across ops (§V14 / ADR-0019), so a whole decode step or training graph records
without host round-trips between ops — the larger architectural change the metal
recorder made (§T366–T412), replicated for CUDA.
