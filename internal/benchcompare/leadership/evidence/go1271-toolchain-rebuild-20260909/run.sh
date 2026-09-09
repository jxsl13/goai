#!/usr/bin/env bash
# Run prebuilt, same-source binaries; compilation must finish before measuring.
set -euo pipefail
bindir=${1:?usage: bash run.sh DIRECTORY_CONTAINING_FROZEN_BINARIES}
mode=${2:-suite}
case "$mode" in
	suite) duration=500ms; warmup=1x ;;
	moe) duration=1s; warmup=1s ;;
	*) echo 'mode must be suite or moe' >&2; exit 2 ;;
esac

run_arm() {
	local version=$1 duration=$2
	if [ "$mode" = moe ]; then
		GOMAXPROCS=12 "$bindir/autograd-go$version.test" -test.run '^$' \
			-test.bench '^BenchmarkMoECombineBackward$' -test.benchtime="$duration" \
			-test.count=1 -test.benchmem
		return
	fi
	GOMAXPROCS=12 "$bindir/linalg-go$version.test" -test.run '^$' \
		-test.bench '^BenchmarkSVD_128x128$' -test.benchtime="$duration" -test.count=1 -test.benchmem
	for procs in 1 12; do
		GOMAXPROCS=$procs "$bindir/autograd-go$version.test" -test.run '^$' \
			-test.bench '^(BenchmarkEighVJP_128|BenchmarkMoECombineBackward)$' \
			-test.benchtime="$duration" -test.count=1 -test.benchmem
	done
}

for version in 1.27.0 1.27.1; do
	run_arm "$version" "$warmup" >/dev/null
done

for pair in {1..9}; do
	versions=(1.27.0 1.27.1)
	if { [ "$mode" = suite ] && (( pair % 2 == 0 )); } ||
		{ [ "$mode" = moe ] && (( pair % 2 == 1 )); }; then
		versions=(1.27.1 1.27.0)
	fi
	for version in "${versions[@]}"; do
		printf 'pair: %s\ntoolchain: go%s\n' "$pair" "$version"
		run_arm "$version" "$duration"
	done
done
