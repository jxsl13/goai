# cuda-nvcc-env.sh — makes a REAL nvcc usable on the immutable pip-only host, so native CUDA
# .cu kernels (mma.h / WMMA tensor cores, int8 MMQ) can be compiled to cubin/fatbin — the path
# nvrtc could not take (no mma.h). Companion to cuda-pip-env.sh (which sets up nvrtc+cuBLAS).
#
# The two pieces (2026-07-18):
#   1. nvcc itself: the pip wheel `nvidia-cuda-nvcc-cu13` ships a COMPLETE toolkit
#      (bin/nvcc + include/ + nvvm/ + ptxas), CUDA 13.2. Brew has NO cuda formula — pip is the
#      only nvcc source here. Install into a venv:  pip install nvidia-cuda-nvcc-cu13
#      (already present in the vLLM venv it was pulled into as a torch/vLLM dep).
#   2. a host compiler nvcc accepts: CUDA 13.2 rejects gcc > 15, and the system is gcc 16.1.
#      `brew install gcc@14` provides g++-14; pass it via nvcc -ccbin.
#
# Verified 2026-07-18 on the RTX 3060 (sm_86): a WMMA (mma.h) tensor-core GEMM compiles to a
# cubin — the "prefill is toolchain-blocked / needs mma.h which nvrtc lacks" characterization
# (SPEC §H4, the whole reason for the Vulkan-coopmat workaround) is SUPERSEDED.

# Locate a pip nvcc-cu13 tree (prefer goai's .venv-cuda, else the vLLM venv, else any site).
_find_nvcc_home() {
  for base in \
    "$HOME/Development/goai/.venv-cuda" \
    "$HOME/.local/share/goai-vllm/vllm-venv" ; do
    local h
    h=$(find "$base" -type d -path '*/nvidia/cu13' 2>/dev/null | head -1)
    [ -n "$h" ] && { echo "$h"; return 0; }
  done
  return 1
}

CUDA_NVCC_HOME=$(_find_nvcc_home)
if [ -z "$CUDA_NVCC_HOME" ]; then
  echo "cuda-nvcc-env: no nvidia/cu13 tree found — run: pip install nvidia-cuda-nvcc-cu13" >&2
  return 1 2>/dev/null || exit 1
fi

CCBIN=/home/linuxbrew/.linuxbrew/bin/g++-14
[ -x "$CCBIN" ] || echo "cuda-nvcc-env: g++-14 missing — run: brew install gcc@14" >&2

export CUDA_HOME="$CUDA_NVCC_HOME"
export PATH="$CUDA_NVCC_HOME/bin:$PATH"
export NVCC_CCBIN="$CCBIN"
# Convenience: nvcc wrapper that always uses the supported host compiler + sm_86.
nvcc_86() { "$CUDA_NVCC_HOME/bin/nvcc" -ccbin "$CCBIN" -I"$CUDA_NVCC_HOME/include" -arch=sm_86 "$@"; }
export -f nvcc_86 2>/dev/null || true
echo "cuda-nvcc-env: nvcc $("$CUDA_NVCC_HOME/bin/nvcc" --version | grep -oE 'release [0-9.]+' | head -1), ccbin g++-14, CUDA_HOME=$CUDA_NVCC_HOME" >&2
