## §H — hardware & toolchain

H1: CPU = AMD Ryzen 7 5700G (Zen 3, 8c/16t). f32-relevant ISA: AVX2 + FMA. ⊥ AVX-512.
H2: GPU = NVIDIA GeForce RTX 3060 (12 GB, Ampere, compute cap 8.6). driver 610.43.02 (ships `libcuda.so`). = ONLY NVIDIA GPU in project.
H3: toolchain = Go 1.26.5 `linux/amd64`, gcc 16.1.1. immutable distro (bazzite) → ⊥ system CUDA.
H4: CUDA 12.9 = USER-SPACE pip wheels in `.venv-cuda` (`nvidia-cuda-nvcc/cublas/cuda-runtime/cccl-cu12`). backend = pure cgo + cuBLAS + nvrtc, `.cu`-string kernels compiled at runtime by nvrtc → needs headers+libs only. `scripts/cuda-pip-env.sh` sets CGO_CFLAGS/LDFLAGS+rpath + unversioned `.so` symlinks. torch-cpu+numpy in `.venv` = vendor-BLAS refs.
H4b: **nvcc IS AVAILABLE (2026-07-18, SUPERSEDES the old "toolchain-blocked" characterization).**
The `nvidia-cuda-nvcc-cu13` pip wheel ships a COMPLETE toolkit (bin/nvcc + include + nvvm +
ptxas, CUDA 13.2) — present via the vLLM venv it was pulled into; brew has NO cuda formula so
pip is the only nvcc source. CUDA 13.2 rejects gcc>15 (host is gcc 16.1) → `brew install
gcc@14` provides the host compiler, passed via `nvcc -ccbin g++-14`. VERIFIED on sm_86: a WMMA
(`mma.h`) tensor-core GEMM compiles to a cubin. `scripts/cuda-nvcc-env.sh` sets it up. This
MOVES the wall that made prefill "toolchain-blocked" (no mma.h in nvrtc) and made the Vulkan-
coopmat arc a workaround: native CUDA tensor-core / fused-FlashAttention / int8-MMQ kernels can
now be precompiled to cubin and loaded via cuModuleLoad. **Tw-FLASHATTN arc booked:** the
measured prefill gap to vLLM (10729 vs 5600 tok/s) is fused attention, not the GEMM — build a
fused FlashAttention prefill kernel in .cu (mma.h QKᵀ + online-softmax + PV), the direct lever.
H5: role = secondary worker. main dev = macbook M2 on `main`. niche = (a) amd64 f32-SIMD (host-blocked on arm64 per §T11b/§T74/§B13) + (b) NVIDIA CUDA. ⊥ touch `main`; PR + dedicated branch only. push via `gh` (git credential helper).
