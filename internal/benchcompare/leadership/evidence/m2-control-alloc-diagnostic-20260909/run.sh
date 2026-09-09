#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
  echo "usage: run.sh old-old|old-v2 OLD_BINARY V2_BINARY OLD_SHA256 V2_SHA256" >&2
  exit 2
fi
phase=$1
old_binary=$2
v2_binary=$3
old_expected=$4
v2_expected=$5
case "$phase" in old-old|old-v2) ;; *) echo "invalid phase" >&2; exit 2 ;; esac
for digest in "$old_expected" "$v2_expected"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid expected SHA-256" >&2; exit 2; }
done
for binary in "$old_binary" "$v2_binary"; do
  [[ "$binary" = /* && -f "$binary" && -x "$binary" ]] || { echo "expected absolute executable path" >&2; exit 2; }
done
old_actual=$(shasum -a 256 "$old_binary" | awk '{print $1}')
v2_actual=$(shasum -a 256 "$v2_binary" | awk '{print $1}')
[[ "$old_actual" == "$old_expected" && "$v2_actual" == "$v2_expected" ]] || { echo "frozen binary SHA mismatch" >&2; exit 2; }

printf 'protocol: control-alloc-fixed1024-v1\nphase: %s\nold-sha256: %s\nv2-sha256: %s\n' "$phase" "$old_actual" "$v2_actual"
printf 'old-path: %s\nv2-path: %s\n' "$old_binary" "$v2_binary"
for campaign in 1 2 3; do
  if (( campaign % 2 )); then process_order="1 12"; else process_order="12 1"; fi
  for pair in 1 2 3 4 5 6 7; do
    for procs in $process_order; do
      if (( (pair + campaign) % 2 )); then arm_order="B A"; else arm_order="A B"; fi
      for arm in $arm_order; do
        binary=$old_binary
        binary_hash=$old_actual
        if [[ "$phase" == old-v2 && "$arm" == B ]]; then
          binary=$v2_binary
          binary_hash=$v2_actual
        fi
        printf '\ncampaign: %s\npair: %s\nprocs: %s\narm: %s\nbinary-sha256: %s\n' "$campaign" "$pair" "$procs" "$arm" "$binary_hash"
        GOMAXPROCS="$procs" GOAI_ALLOC_DIAGNOSTIC=1 "$binary" \
          -test.run='^TestCPUControlAllocationDiagnostics$' \
          -test.benchtime=1024x -test.count=1 -test.v -test.timeout=5m
      done
    done
  done
done
