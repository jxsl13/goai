#!/usr/bin/env bash
# Diagnostic only: short, paired large public-operation measurements.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: bash pilot.sh OLD_BINARY NEW_BINARY' >&2; exit 2; }
old_binary=$1
new_binary=$2
[[ -x "$old_binary" && -x "$new_binary" ]]
echo 'protocol: diagnostic only; two alternating pairs; 200ms samples; no promotion claim'
echo 'ambient: macOS background services observed consuming multiple CPU cores'
echo "old-sha256: $(shasum -a 256 "$old_binary" | awk '{print $1}')"
echo "new-sha256: $(shasum -a 256 "$new_binary" | awk '{print $1}')"
for pair in 1 2; do
  arms=(old new)
  (( pair == 1 )) || arms=(new old)
  for procs in 1 12; do
    for arm in "${arms[@]}"; do
      binary=$old_binary
      [[ $arm != new ]] || binary=$new_binary
      echo "pair: $pair"
      echo "procs: $procs"
      echo "arm: $arm"
      GOMAXPROCS=$procs "$binary" -test.run '^$' \
        -test.bench '^BenchmarkVGELUF64NeonBoundary$/^execute$/.*/.*/^n262144$' \
        -test.benchtime=200ms -test.count=1
    done
  done
done
