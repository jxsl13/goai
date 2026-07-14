#!/usr/bin/env bash
# Export cgo flags to build/test the CUDA backend (-tags cuda) against a
# USER-SPACE CUDA 12 toolkit installed from pip wheels — no root / no system
# package manager, which is what an immutable distro (e.g. bazzite/Universal
# Blue) needs. The goai CUDA backend is pure cgo + cuBLAS with no .cu kernels,
# so it needs headers + libs only (not nvcc).
#
# One-time setup (from the repo root):
#   python3 -m venv .venv-cuda
#   .venv-cuda/bin/pip install nvidia-cuda-nvcc-cu12 nvidia-cublas-cu12 \
#                              nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
#
# Then:  source scripts/cuda-pip-env.sh  &&  go test -tags cuda ./backend/cuda/
#
# The nvidia-cuda-nvcc wheel supplies crt/host_config.h (pulled in by
# cuda_runtime.h) and ptxas; cublas pulls in nvrtc. Wheels ship versioned libs
# (libcublas.so.12); this script makes the unversioned symlinks cgo's
# -lcublas/-lcudart expect, and sets an rpath so the test binary finds them.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NV="$(echo "$REPO"/.venv-cuda/lib/python*/site-packages/nvidia)"
if [ ! -d "$NV" ]; then
	echo "cuda-pip-env: no CUDA wheels under $REPO/.venv-cuda — run the one-time setup above" >&2
	return 1 2>/dev/null || exit 1
fi

for pair in "cuda_runtime/lib:libcudart" "cublas/lib:libcublas" "cuda_nvrtc/lib:libnvrtc"; do
	d="$NV/${pair%%:*}"
	base="${pair##*:}"
	so="$(ls "$d/$base".so.* 2>/dev/null | head -1)"
	if [ -n "$so" ] && [ ! -e "$d/$base.so" ]; then ln -s "$(basename "$so")" "$d/$base.so"; fi
done

export CGO_ENABLED=1
export CGO_CFLAGS="-I$NV/cuda_runtime/include -I$NV/cublas/include -I$NV/cuda_cccl/include -I$NV/cuda_nvcc/include"
export CGO_LDFLAGS="-L$NV/cuda_runtime/lib -L$NV/cublas/lib -Wl,-rpath,$NV/cuda_runtime/lib -Wl,-rpath,$NV/cublas/lib -Wl,-rpath,$NV/cuda_nvrtc/lib"
export LD_LIBRARY_PATH="$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cuda_nvrtc/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"
echo "cuda-pip-env: CGO flags set for $NV"
