#!/usr/bin/env bash
set -euo pipefail

module="${PERFSCAN_MODULE:-github.com/jxsl13/perfscan@v1.81.0}"
config="${PERFSCAN_CONFIG:-perfscan.yaml}"

if [ "$#" -eq 0 ]; then
	set -- ./...
fi

report="$(mktemp "${TMPDIR:-/tmp}/goai-perfscan.XXXXXX")"
trap 'rm -f "$report"' EXIT

if ! env GOPROXY=direct CGO_ENABLED=0 go run "$module" \
	-config "$config" -exit-zero "$@" >"$report"; then
	cat "$report"
	exit 1
fi

passthrough=0
for arg in "$@"; do
	case "$arg" in
		-fix|-diff|-json|-sarif|-list|-version|-explain|-explain=*) passthrough=1 ;;
	esac
done

if [ "${PERFSCAN_VERBOSE:-0}" = "1" ] || [ "$passthrough" = "1" ]; then
	cat "$report"
fi

if [ "$passthrough" = "1" ]; then
	exit 0
fi

findings="$(grep -Ec '\(PS[0-9][0-9][0-9][0-9] L[123]\)$' "$report" || true)"
printf 'perfscan %s: %s advisory finding(s)\n' "$module" "$findings"
