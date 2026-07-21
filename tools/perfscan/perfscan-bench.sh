#!/usr/bin/env bash
# perfscan-bench.sh — pattern P4: flag "optimized" benchmarks that barely beat
# their naive/slow twin (the tell of a missed optimization, see PATTERNS.md).
#
# For each BenchmarkX{Fast,Opt} that has a matching BenchmarkX{Slow,Naive}, it
# runs both and reports the ratio; ratios below THRESHOLD (default 2.0) are
# flagged as candidates worth investigating.
#
# Usage:  tools/perfscan/perfscan-bench.sh <pkg> [threshold] [benchtime]
#   e.g.  tools/perfscan/perfscan-bench.sh ./nn 2.0 500x
set -u
PKG="${1:-./...}"
THRESHOLD="${2:-2.0}"
BENCHTIME="${3:-1000x}"

# One pass: run all *Fast/*Opt and *Slow/*Naive benchmarks, capture ns/op.
out="$(go test "$PKG" -run '^$' -bench '(Fast|Opt|Slow|Naive)$' -benchtime "$BENCHTIME" 2>/dev/null)"
if [ -z "$out" ]; then
	echo "perfscan-bench: no Fast/Slow benchmark pairs in $PKG"
	exit 0
fi

# name -> ns/op
declare -A NS
while read -r name _n nsop _rest; do
	[ -z "${nsop:-}" ] && continue
	# strip the -NN cpu suffix from the benchmark name
	base="${name%-*}"
	NS["$base"]="$nsop"
done < <(echo "$out" | awk '/ns\/op/ {print $1, $2, $3}')

flagged=0
checked=0
for name in "${!NS[@]}"; do
	case "$name" in
		*Fast|*Opt) ;;
		*) continue ;;
	esac
	stem="${name%Fast}"; stem="${stem%Opt}"
	slow=""
	for cand in "${stem}Slow" "${stem}Naive"; do
		if [ -n "${NS[$cand]:-}" ]; then slow="$cand"; break; fi
	done
	[ -z "$slow" ] && continue
	fast_ns="${NS[$name]}"; slow_ns="${NS[$slow]}"
	ratio="$(awk -v s="$slow_ns" -v f="$fast_ns" 'BEGIN{ if (f>0) printf "%.2f", s/f; else print "inf" }')"
	checked=$((checked+1))
	below="$(awk -v r="$ratio" -v t="$THRESHOLD" 'BEGIN{ print (r+0 < t+0) ? 1 : 0 }')"
	if [ "$below" = "1" ]; then
		echo "FLAG  $name ${ratio}x over $slow  (${fast_ns} vs ${slow_ns} ns/op) — investigate: the fast path still carries an avoidable cost"
		flagged=$((flagged+1))
	else
		echo "ok    $name ${ratio}x over $slow"
	fi
done

echo ""
echo "perfscan-bench: $checked pair(s) checked, $flagged flagged below ${THRESHOLD}x"
