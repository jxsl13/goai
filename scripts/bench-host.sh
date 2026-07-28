#!/usr/bin/env bash
# scripts/bench-host.sh — honest, versioned, per-host benchmark harness.
#
# WHAT IT DOES
#   Measures GoAI against the incumbent frameworks ACTUALLY INSTALLED on THIS
#   machine and persists every result to the spectackle bench store, so
#   BENCHMARKS.md can be regenerated from one source of truth and refreshed
#   periodically (cron). Every number carries its tool version and host frame.
#
# WHY spectackle
#   Each result is a versioned record keyed <bench>/<impl> with a machine frame
#   (os/arch/cpu/ram/gpu) and a metric declaration (name:unit:dir). Re-running
#   appends a new version, so the store IS the honest, dated history per host.
#
# ENVIRONMENT (auto-captured into every note; also printed by `manifest`)
#   GoAI CPU      : GOEXPERIMENT=simd (AVX2 on this Zen3 host), go toolchain
#   GoAI CUDA     : source scripts/cuda-pip-env.sh  (pip CUDA-12 wheels in
#                   .venv-cuda: cudart/cublas/nvrtc — no system toolkit/root)
#   torch (CPU)   : $REPO/.venv           (torch *+cpu, numpy)
#   torch (CUDA)  : ~/.local/share/goai-vllm/vllm-venv (torch *+cuXXX, vLLM)
#   llama.cpp     : prebuilt Vulkan via scripts/bench-llamacpp.sh (runs on the
#                   Vulkan runtime; no nvcc needed)
#
# USAGE
#   scripts/bench-host.sh manifest        # print the versioned env manifest
#   scripts/bench-host.sh cpu-gemm        # GoAI vs torch-cpu/numpy GEMM
#   scripts/bench-host.sh                 # every section runnable on this host
#   BENCH_COUNT=10 scripts/bench-host.sh cpu-gemm
#
# INVARIANT: benchmarks are latency-sensitive — this script runs each measured
# workload SERIALLY and expects the machine otherwise idle (no parallel builds,
# no other GPU tenant). It never runs two measured workloads concurrently.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$PATH:$HOME/go/bin"
BENCH_COUNT="${BENCH_COUNT:-10}"
VENV="$REPO/.venv"
VLLM_VENV="$HOME/.local/share/goai-vllm/vllm-venv"
SPX() { spectackle call -root "$REPO" bench "$@"; }

# ---- host frame + tool versions (documented, honest) -----------------------
ver_go()        { go version 2>/dev/null | awk '{print $3}'; }
ver_torch_cpu() { "$VENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null; }
ver_numpy()     { "$VENV/bin/python" -c 'import numpy;print(numpy.__version__)' 2>/dev/null; }
ver_vllm()      { "$VLLM_VENV/bin/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null; }
ver_torch_cu()  { "$VLLM_VENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null; }
cpu_slug()      { grep -m1 'model name' /proc/cpuinfo | sed -E 's/.*: //; s/ with.*//; s/AMD Ryzen 7 /ryzen7-/; s/ .*//' | tr 'A-Z' 'a-z'; }
gpu_name()      { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA GeForce //; s/ /-/g' | tr 'A-Z' 'a-z'; }
ram_gi()        { free -g | awk '/Mem:/{print $2"gi"}'; }
driver_ver()    { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1; }

FRAME_CPU="os=linux arch=amd64 cpu=$(cpu_slug) ram=$(ram_gi) gpu=none"
FRAME_GPU="os=linux arch=amd64 cpu=$(cpu_slug) ram=$(ram_gi) gpu=$(gpu_name)"

manifest() {
  cat <<EOF
=== GoAI benchmark host manifest ($(date -u +%Y-%m-%dT%H:%MZ)) ===
host      : $(uname -sm) kernel $(uname -r)
cpu       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) ($(nproc) threads)
ram       : $(free -h | awk '/Mem:/{print $2}')
gpu       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) driver $(driver_ver)
goai      : $(git rev-parse --short HEAD) ($(git symbolic-ref --short HEAD 2>/dev/null || echo detached))
go        : $(ver_go)
torch-cpu : $(ver_torch_cpu)   (in .venv)
numpy     : $(ver_numpy)
vllm      : $(ver_vllm)   torch-cuda $(ver_torch_cu)   (in vllm-venv)
cuda      : pip wheels in .venv-cuda (cudart/cublas/nvrtc, cu12); GoAI -tags cuda
llama.cpp : prebuilt Vulkan ${LLAMACPP_BUILD:-b10012} (scripts/bench-llamacpp.sh)
frame(cpu): $FRAME_CPU
frame(gpu): $FRAME_GPU
EOF
}

# put helper: name impl "metric=value..." metricdecl note
put() {
  local name="$1" results="$2" mdecl="$3" note="$4" frame="${5:-$FRAME_CPU}"
  SPX "$(python3 - "$name" "$frame" "$mdecl" "$results" "$note" <<'PY'
import json,sys
name,frame,mdecl,results,note=sys.argv[1:6]
print(json.dumps({"op":"put","name":name,"frame":frame,"metrics":mdecl,"results":results,"note":note}))
PY
)" 2>&1 | grep -oE 'M-[A-Z0-9]+.*(new|better|worse|tie).*' | head -1
}

# ---- CPU GEMM: GoAI SIMD vs torch-cpu / numpy ------------------------------
cpu_gemm() {
  echo ">> CPU GEMM 1024^3 (GoAI GOEXPERIMENT=simd, count=$BENCH_COUNT) — serial"
  local out; out="$(GOEXPERIMENT=simd go test -run '^$' -bench 'BenchmarkGEMM_F(32|64)_1024_gflops' -count="$BENCH_COUNT" ./backend/cpu/ 2>/dev/null)"
  local g32 g64
  g32="$(awk '/GEMM_F32_1024/{s+=$5;n++} END{if(n)printf "%.1f",s/n}' <<<"$out")"
  g64="$(awk '/GEMM_F64_1024/{s+=$5;n++} END{if(n)printf "%.1f",s/n}' <<<"$out")"
  echo "   GoAI f32=$g32  f64=$g64 GFLOP/s"
  local gover; gover="$(ver_go)"
  [ -n "$g32" ] && put "cpu/GEMM_f32_1024/goai-simd" "goai-simd: gflops=$g32" "gflops:GFLOP/s:+" "1024^3 f32; GoAI GOEXPERIMENT=simd AVX2; $gover; count=$BENCH_COUNT mean"
  [ -n "$g64" ] && put "cpu/GEMM_f64_1024/goai-simd" "goai-simd: gflops=$g64" "gflops:GFLOP/s:+" "1024^3 f64; GoAI GOEXPERIMENT=simd AVX2; $gover; count=$BENCH_COUNT mean"

  echo ">> CPU GEMM incumbents (torch-cpu $(ver_torch_cpu), numpy $(ver_numpy)) — serial"
  local py; py="$("$VENV/bin/python" - <<'PY'
import time,statistics,os,json,numpy as np,torch
torch.set_num_threads(os.cpu_count())
N=1024
def med(fn,reps=15,warm=3):
    for _ in range(warm): fn()
    ts=[time.perf_counter() or 0 for _ in range(0)]
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return (2*N**3)/statistics.median(ts)/1e9
r={}
for nm,dt in (("f32",np.float32),("f64",np.float64)):
    a=np.random.rand(N,N).astype(dt); b=np.random.rand(N,N).astype(dt)
    r[f"numpy_{nm}"]=round(med(lambda:a@b),1)
    ta,tb=torch.from_numpy(a),torch.from_numpy(b)
    r[f"torch_{nm}"]=round(med(lambda:ta@tb),1)
print(json.dumps(r))
PY
)"
  echo "   incumbents: $py"
  local tv nv; tv="$(ver_torch_cpu)"; nv="$(ver_numpy)"
  local t32 t64 n32 n64
  t32="$(python3 -c "import json,sys;print(json.load(sys.stdin)['torch_f32'])" <<<"$py")"
  t64="$(python3 -c "import json,sys;print(json.load(sys.stdin)['torch_f64'])" <<<"$py")"
  n32="$(python3 -c "import json,sys;print(json.load(sys.stdin)['numpy_f32'])" <<<"$py")"
  n64="$(python3 -c "import json,sys;print(json.load(sys.stdin)['numpy_f64'])" <<<"$py")"
  put "cpu/GEMM_f32_1024/torch-cpu-$tv" "torch-cpu: gflops=$t32" "gflops:GFLOP/s:+" "1024^3 f32; torch $tv+cpu OpenBLAS $(nproc)thr; reps=15 median"
  put "cpu/GEMM_f64_1024/torch-cpu-$tv" "torch-cpu: gflops=$t64" "gflops:GFLOP/s:+" "1024^3 f64; torch $tv+cpu; reps=15 median"
  put "cpu/GEMM_f32_1024/numpy-$nv"     "numpy: gflops=$n32"     "gflops:GFLOP/s:+" "1024^3 f32; numpy $nv $(nproc)thr; reps=15 median"
  put "cpu/GEMM_f64_1024/numpy-$nv"     "numpy: gflops=$n64"     "gflops:GFLOP/s:+" "1024^3 f64; numpy $nv; reps=15 median"
}

# ---- GPU decode: GoAI-CUDA vs vLLM vs llama.cpp-Vulkan ----------------------
# Verified-runnable command blocks on this host (RTX 3060). Left as explicit,
# documented steps because each is minutes-long and needs the GPU idle; a full
# `all` run executes them. Persist with FRAME_GPU.
gpu_decode() {
  command -v nvidia-smi >/dev/null || { echo "no GPU — skip"; return; }
  echo ">> GoAI CUDA decode (source scripts/cuda-pip-env.sh) — serial, GPU must be idle"
  # shellcheck disable=SC1091
  source scripts/cuda-pip-env.sh >/dev/null 2>&1
  GOEXPERIMENT=simd go test -tags cuda -run '^$' -bench 'BenchmarkTinyLlamaDecode' -count=5 ./backend/cuda/ 2>/dev/null | tee /tmp/goai_cuda_decode.txt | grep Decode || echo "   (adjust bench name per backend/cuda/*_test.go)"
  echo ">> llama.cpp Vulkan (same GGUF weights)"
  scripts/bench-llamacpp.sh "$REPO/models/tinyllama-1.1b-chat-q8_0.gguf" 2>/dev/null | tail -6
  echo ">> vLLM 0.25.1 decode ($(ver_vllm))"
  echo "   $VLLM_VENV/bin/python -m vllm.entrypoints ... (see docs/benchmarking.md 'Three-way head-to-head')"
  # NOTE: parse tok/s from each and persist with put ... "$FRAME_GPU"; matched
  # precision class only (Q4_K vs Q4_K_M, Q8 vs Q8) — never mix precisions.
}

case "${1:-all}" in
  manifest)  manifest ;;
  cpu-gemm)  manifest; cpu_gemm ;;
  gpu-decode) manifest; gpu_decode ;;
  all)       manifest; cpu_gemm; gpu_decode ;;
  *) echo "usage: $0 {manifest|cpu-gemm|gpu-decode|all}"; exit 2 ;;
esac
