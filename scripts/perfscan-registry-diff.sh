#!/usr/bin/env bash
set -euo pipefail

module="${PERFSCAN_MODULE:-github.com/jxsl13/perfscan@v1.81.0}"
expected_file="${PERFSCAN_COMPAT_CHECKS:-perfscan-compat-checks.txt}"

legacy_ids() {
	env CGO_ENABLED=0 go run ./internal/perfscan -list |
		awk '$1 ~ /^PS[0-9][0-9][0-9][0-9]$/ { print $1 }' |
		sort -u
}

external_ids() {
	env GOPROXY=direct CGO_ENABLED=0 go run "$module" -list |
		awk '$1 ~ /^PS[0-9][0-9][0-9][0-9]$/ { print $1 }' |
		sort -u
}

actual="$(comm -23 <(legacy_ids) <(external_ids))"
expected="$(awk '/^PS[0-9][0-9][0-9][0-9]$/ { print $1 }' "$expected_file" | sort -u)"

if [ "$actual" != "$expected" ]; then
	printf '%s\n' 'perfscan registry difference changed.'
	printf '%s\n' 'Expected compatibility IDs:' "$expected"
	printf '%s\n' 'Actual internal-only IDs:' "$actual"
	printf '%s\n' 'Update the external port and perfscan-compat-checks.txt together.'
	exit 1
fi

count="$(printf '%s\n' "$actual" | awk 'NF { n++ } END { print n + 0 }')"
printf 'perfscan registry parity: %s internal-only compatibility check(s)\n' "$count"
