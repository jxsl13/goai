#!/usr/bin/env bash
# scaling_sweep.sh — find SERIAL SPINES by measurement instead of by reading code.
#
# Runs each benchmark at GOMAXPROCS=1 and at the machine's full width, and reports the
# ratio. A benchmark with substantial ns/op and a ratio near 1.0 does not parallelize at
# all: whatever dominates it is serial. That is a candidate, not a defect — plenty of
# work is legitimately serial — but it is the cheapest way to find the ones that are not.
#
# WHY THIS IS A SCRIPT AND NOT A PERFSCAN RULE: proving that a loop's iterations are
# independent is a dataflow question, and perfscan is AST-only. A static rule that
# guessed at independence would advise data races. Scaling is an observation, so it is
# measured rather than inferred.
#
# Found in one sweep: classic GBM histogram (1.01x at 334ms, since fixed to 1.57x),
# GMM full-covariance fit (1.00x at 77ms), MLA VJP (0.99x at 20ms), Cholesky VJP
# (0.88x at 4.7ms — slower with more cores). The quantized prefill path was found the
# same way and is now 2.52x.
#
# Usage:  internal/perfscan/tools/scaling_sweep.sh <pkg> [benchtime] [regexp]
#   e.g.  internal/perfscan/tools/scaling_sweep.sh ./classic 6x '^BenchmarkGBM'
set -uo pipefail

pkg="${1:?usage: scaling_sweep.sh <pkg> [benchtime] [regexp]}"
bt="${2:-10x}"
filter="${3:-.}"

# Anything under this is too quick for the ratio to mean much — the measurement noise
# and the pool's own dispatch cost dominate.
MIN_NS=2000000

names=$(go test "$pkg" -run XXX -bench . -benchtime 1x -count 1 2>/dev/null \
        | grep -oE '^Benchmark[A-Za-z0-9_]+' | sort -u | grep -E "$filter")
[ -z "$names" ] && { echo "no benchmarks matching $filter in $pkg"; exit 1; }

printf '%-44s %12s %12s %9s\n' BENCHMARK 1P FULL SPEEDUP
for b in $names; do
  one=$(GOMAXPROCS=1 go test "$pkg" -run XXX -bench "^${b}\$" -benchtime "$bt" -count 1 2>/dev/null \
        | awk '/^Benchmark/{print $3}')
  many=$(go test "$pkg" -run XXX -bench "^${b}\$" -benchtime "$bt" -count 1 2>/dev/null \
        | awk '/^Benchmark/{print $3}')
  [ -z "$one" ] || [ -z "$many" ] || [ "$many" = "0" ] && continue
  awk -v n="$b" -v o="$one" -v m="$many" -v min="$MIN_NS" 'BEGIN{
    r = o/m
    flag = (r < 1.25 && o > min) ? "  <-- SERIAL SPINE" : ""
    printf "%-44s %10.2fms %10.2fms %8.2fx%s\n", n, o/1e6, m/1e6, r, flag
  }'
done
