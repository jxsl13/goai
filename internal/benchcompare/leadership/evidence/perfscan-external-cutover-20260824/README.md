# External perfscan cutover evidence — 2026-08-24

## Outcome

This is an architecture-leverage claim, not a scanner throughput claim. GoAI's
canonical scanner moves to the shared external release while an exact registry
difference retains coverage that is not yet upstream.

| Metric | Internal | External v1.81.0 | Compatibility |
| --- | ---: | ---: | ---: |
| Stable check IDs | 146 | 375 | 53 internal-only |
| Shared IDs | 93 | 93 | — |
| Additional external IDs | — | 282 | — |
| Whole-tree advisory findings | not republished | 1,903 | 200 |
| Median warm runtime | 5.14s | 25.08s | 5.14s |

The parity audit adds 0.83s median. The staged CI sequence therefore costs about
31.05s on the validation host. The external analyzer is about 4.88 times slower
than the old AST pass in this tree; its leverage is broader shared coverage and
eliminating future forked maintenance.

## Reproducibility boundary

- Hardware: Apple M2 Pro, 32 GiB
- OS: macOS 26.5.1 (25F80)
- Go: 1.27.0 darwin/arm64
- External module: `github.com/jxsl13/perfscan@v1.81.0`
- External release time: 2026-08-22T22:20:18Z
- Resolution: `GOPROXY=direct`
- Vocabulary: `perfscan.yaml`
- Upstream gap: <https://github.com/jxsl13/perfscan/issues/877>

The historical `/perfscan` submodule path resolves only v1.71.0. The root module
is pinned because using the named historical path would downgrade the scanner.

## Gates

- External whole-tree scan: PASS, 1,903 advisory findings.
- Internal-only scan: PASS, 200 advisory findings across exactly 53 checks.
- Registry difference: PASS, exact match with `perfscan-compat-checks.txt`.
- External configuration: PASS, no unknown-key/configuration error.
- Compatibility fixtures: PASS, `go test ./internal/perfscan`.
- Change selector tests: PASS, `go test ./internal/cichange`.
- Shell syntax: PASS for all three runner/audit scripts.

The raw runtime samples are in `samples.txt`; commands are in `commands.txt`.
