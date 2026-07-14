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

## The most-impact next optimization (bottom-up)

808 GFLOP/s at 1024³ is far below the RTX 3060's ~12.7 TFLOP/s f32 peak. The
backend is **transfer-bound**, not compute-bound: every `matmulF32` call
`cudaMalloc`s three device buffers, copies A/B H2D, runs `cublasSgemm`, copies C
D2H, and frees — synchronously (honest per the code's own note). At 1024³ that
is ~12 MB of round-trip PCIe traffic and three allocations to hide ~2 GFLOP of
compute.

The foundational lever (the "bottom layer") is therefore **device-resident
tensors** (§V14): keep tensors on the GPU across ops so a training step's
matmuls don't bounce through host memory each call, and pool device allocations.
That is the next CUDA task — cuBLAS itself is already optimal for the GEMM
kernel, so the win is in the memory/transfer layer around it.
