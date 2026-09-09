# Go 1.27.1 patch rebuild

This is a toolchain compatibility update, with no kernel speedup claim.
The module floor and all ten CI toolchain declarations are pinned to Go 1.27.1.
The [official release](https://go.dev/doc/devel/release#go1.27.1) includes
compiler, runtime, cgo, and SIMD fixes. The source tree already contains the
complete Go 1.27 migration; no runtime algorithm changed in this update.

## Provenance

- Identical benchmark source: `640160379b3d7199dfe716e3a7dc21ae17f4a809`.
- Apple M2 Pro, darwin/arm64, macOS 26.5.1 (25F80).
- Go 1.27.0 and Go 1.27.1; `CGO_ENABLED=0`, no `GOEXPERIMENT`,
  `GOARM64=v8.0`. All binaries were compiled before measurement.
- Official `go1.27.1.darwin-arm64.tar.gz` SHA256:
  `ee215d57e0ec269c60cc9ceca68e6bda321ba9ee5afe24f4b0988703c2d87d12`.
- benchstat: `golang.org/x/perf@v0.0.0-20260709024250-82a0b07e230d`.

Frozen test-binary SHA256 values:

| Binary | SHA256 |
| --- | --- |
| linalg, Go 1.27.0 | 6968501822c3ee16fe4d3d2ba05b5ee36a18c5793f70d7e98301724a19e23020 |
| linalg, Go 1.27.1 | 3d414f8d77bad164f3ea32a4a9f24649cadb1955a03d5d21727b467e8eaf4826 |
| autograd, Go 1.27.0 | ef7fb8df4d703f98c9899a30f49c261896d8750d0f3f61f60807ed8f2fe856d7 |
| autograd, Go 1.27.1 | f06dc11fdf3ef57c18e5c79ecc1803a6a59ae04e047e62ef1603fbb0a80ccd86 |

## Measurements

`raw.txt` contains nine alternating AB/BA pairs, with a discarded one-iteration
warmup per cell and 500 ms measurement targets. The table reports independent
medians, with pair direction as an additional check. No builds or test sweeps
ran concurrently with these measurements.

| Workload | GOMAXPROCS | Go 1.27.0 ms | Go 1.27.1 ms | 1.27.1 pair wins | benchstat time p |
| --- | ---: | ---: | ---: | ---: | ---: |
| SVD 128x128 | 12 | 25.617222 | 25.728093 | 3/9 | 0.258 |
| Eigh VJP 128 | 1 | 4.117274 | 4.134350 | 4/9 | 0.796 |
| MoECombine backward | 1 | 19.451841 | 19.360705 | 5/9 | 0.730 |
| Eigh VJP 128 | 12 | 1.327511 | 1.597536 | 4/9 | 0.436 |
| MoECombine backward | 12 | 4.680187 | 5.709938 | 1/9 | 0.024 |

The initial parallel MoE result is **22.00% slower** by independent medians.
It triggered a focused confirmation, retained in `moe-confirm.txt`: nine new
pairs, reversed initial arm order, a discarded one-second warmup per binary,
and one-second measurement targets. Medians were 4.056082 and 4.202007 ms
(3.60% slower, p=0.489, 3/9 candidate wins; paired old/new median 0.932872).
The initial regression therefore did not reproduce as a significant result;
this does not prove equivalence or exclude a smaller regression. Parallel
cells remain sensitive to runtime scheduling on the heterogeneous M2 cores.
Both campaigns are retained; neither is selected as a compiler speedup.

Median allocations per operation are unchanged in all five cells: 144, 153,
35, 228, and 60 respectively. Median allocated bytes show no significant change.
The SVD bit-identity test passes under both toolchains.

## Reproduction

Use a detached checkout of the source commit above, whose `go.mod` still
allows both compilers. Compile `linalg` and `autograd` test binaries there
with each exact toolchain, using `GOTOOLCHAIN=local CGO_ENABLED=0` and no
`GOEXPERIMENT`. Name the outputs `linalg-go1.27.0.test`,
`linalg-go1.27.1.test`, `autograd-go1.27.0.test`, and
`autograd-go1.27.1.test` in one directory. For example:

```sh
GOTOOLCHAIN=local CGO_ENABLED=0 /path/to/go1.27.0/bin/go test -c ./linalg \
  -o /path/to/binaries/linalg-go1.27.0.test
bash run.sh /path/to/binaries > raw.txt
bash run.sh /path/to/binaries moe > moe-confirm.txt
```

Run on an otherwise quiet host. `run.sh` executes prebuilt binaries only.
Split each retained file by its `toolchain:` marker before passing the two
sets of benchmark lines to benchstat. Keep the `pair:` markers when computing
pair ratios or wins. Do not combine the five-cell and focused campaigns.

## Acceptance and scope

Local verification covers the whole pure-Go build, vet, and established
short-suite gate (all buildable packages except the existing unrelated
`internal/mdlint` debt), plus the full Linux AMD64 SIMD build and test-binary
compilation for `internal/simd`, `backend/cpu`, and `format/gguf`.
Native SIMD build and all tests in `internal/simd`, `backend/cpu`, and
`format/gguf` pass. The M2 capability probe confirms Metal and MoltenVK device
access; the Metal and llamagpu short suites pass with native GPU access.
The Vulkan-tagged full-tree build, llamagpu vet, and MoltenVK backend tests
also pass, as do Windows AMD64 and Linux ARM64 pure-Go cross-builds.
External perfscan v1.81.0 passes the 53-check compatibility audit, its fixture
tests, and complete-tree scan (1,890 advisory findings; compatibility scan:
232 advisory findings). `GOPROXY=direct` is enforced by the integration.
Tidy, gofmt, changed-document lint, and parsing/counting all ten CI selectors
also pass.

Independent final verification and every PR CI lane must pass before merge.
In particular,
inspect the actual SIMD job and step conclusions: `continue-on-error` means
the overall workflow conclusion alone is insufficient.

Spectackle 0.10.0 EARS lint reports zero errors and 136 inherited prose
warnings. Independent review repaired six rules' path-only `applies` lists
through the server, preserving all rule text and rationales; all 13 unresolved
anchors are cleared and the paginated check reports zero drift. Two inherited
`CTX E` diagnostics remain: `performance` and `testing` store 35 global
contracts without source anchors. Both bundles match the base commit.
Declaring them logical is correctly refused by this CLI because the rules are
unbound; arbitrary source anchors would misrepresent their scope. These are
explicit existing spec-context debt, so the whole check is not reported as
clean. The supported server has no rule/context relocation operation.

This update supersedes stale PR #1246, which targeted an older feature branch
and duplicated already merged changes. Its `ReadRawFile` returned tensor
views after unmapping their backing bytes, and its benchmark never consumed
those bytes. Current main already supplies the safe, explicitly owned
`OpenRaw` API; the invalid addition is not carried forward.

Compiler fingerprinting and attribution are tracked in
[perfscan issue #912](https://github.com/jxsl13/perfscan/issues/912).
The full leadership matrix still requires separate incumbent measurements;
these same-source compiler comparisons do not establish an incumbent win.
