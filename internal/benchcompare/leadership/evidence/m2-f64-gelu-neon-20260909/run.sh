#!/usr/bin/env bash
# Three isolated campaigns over prebuilt binaries with an identical harness.
set -euo pipefail

if [[ $# != 2 && $# != 3 ]]; then
  echo "usage: bash run.sh OLD_BINARY NEW_BINARY [controls]" >&2
  exit 2
fi
old_binary=$1
new_binary=$2
[[ -x "$old_binary" && -x "$new_binary" ]]
benchmark='^BenchmarkVGELUF64NeonBoundary$'
fixtures='forward/backward; active/mixed; n2048/n262144'
if [[ $# == 3 ]]; then
  [[ $3 == controls ]] || { echo 'third argument must be controls' >&2; exit 2; }
  benchmark='^Benchmark(SigmoidF64_64K_cpu|SoftplusF64_256K_cpu|SiLUBackwardF64_256K_cpu)$'
  fixtures='non-target F64 Sigmoid/Softplus/SiLUBackward controls'
fi

echo 'protocol: three alternating count-seven campaigns; one-second samples'
echo 'boundary: leaf is preallocated; Execute includes output allocation'
echo "fixtures: $fixtures"
echo 'warmup: one second per arm and cell before each campaign; excluded'
echo "old-sha256: $(shasum -a 256 "$old_binary" | awk '{print $1}')"
echo "new-sha256: $(shasum -a 256 "$new_binary" | awk '{print $1}')"

for campaign in 1 2 3; do
  if (( campaign % 2 )); then
    procs_order=(1 12)
    warm_order=(old new)
  else
    procs_order=(12 1)
    warm_order=(new old)
  fi
  for procs in "${procs_order[@]}"; do
    for arm in "${warm_order[@]}"; do
      binary=$old_binary
      [[ $arm != new ]] || binary=$new_binary
      GOMAXPROCS=$procs "$binary" -test.run '^$' -test.bench "$benchmark" \
        -test.benchtime=1s -test.count=1 >/dev/null
    done
  done
  for pair in 1 2 3 4 5 6 7; do
    if (( (pair + campaign) % 2 )); then
      arms=(new old)
    else
      arms=(old new)
    fi
    for procs in "${procs_order[@]}"; do
      for arm in "${arms[@]}"; do
        binary=$old_binary
        [[ $arm != new ]] || binary=$new_binary
        echo "campaign: $campaign"
        echo "pair: $pair"
        echo "procs: $procs"
        echo "arm: $arm"
        GOMAXPROCS=$procs "$binary" -test.run '^$' -test.bench "$benchmark" \
          -test.benchtime=1s -test.count=1
      done
    done
  done
done
