#!/usr/bin/env bash
set -euo pipefail

config="${PERFSCAN_CONFIG:-perfscan.yaml}"
checks_file="${PERFSCAN_COMPAT_CHECKS:-perfscan-compat-checks.txt}"
checks="$(awk '/^PS[0-9][0-9][0-9][0-9]$/ { print $1 }' "$checks_file" | paste -sd, -)"

if [ "$#" -eq 0 ]; then
	set -- ./...
fi

report="$(mktemp "${TMPDIR:-/tmp}/goai-perfscan-compat.XXXXXX")"
trap 'rm -f "$report"' EXIT

if ! env CGO_ENABLED=0 go run ./internal/perfscan -config "$config" \
	-checks "$checks" "$@" >"$report"; then
	cat "$report"
	exit 1
fi

if [ "${PERFSCAN_VERBOSE:-0}" = "1" ]; then
	cat "$report"
fi

findings="$(sed -n 's/^perfscan: \([0-9][0-9]*\) candidate.*/\1/p' "$report" | tail -n 1)"
printf 'perfscan compatibility: %s advisory finding(s) across 53 internal-only checks\n' "${findings:-0}"
