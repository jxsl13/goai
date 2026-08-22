# M2 native Metal IQ3 evidence

Date: 2026-08-22
Base: origin/main a3a55ab7d087d5bbb063c5d06cca8b2f66fa1ded
Candidate: this evidence directory's containing commit
Hardware: Apple M2 Pro, 12 CPU cores, 32 GiB unified memory
Software: macOS 26.5.1 (25F80), Go 1.26.6 darwin/arm64, Spectackle 0.10.0

## Claim boundary

This tranche adds exact native Metal kernels and end-to-end loader/recorder support for GGUF type 18
IQ3_XXS and type 21 IQ3_S. It publishes two separate dispatch decisions:

- Generic host input/output remains on the fused ARM64 CPU implementation because standalone Metal
  submissions regress several format/shape cells.
- Device-resident decoding uploads each compressed weight once and records a cooperative IQ3 matvec
  in the whole-token command buffer. This is the promoted production path.

Both the scalar control and cooperative production kernels use the same persistent codebook buffer.
The codebooks are reconstructed once through the public exact GGUF decoder, preventing an independent
hand-maintained copy of the IQ3 tables.

## Correctness and reachability

`correctness.txt` records arbitrary packed-block cross-reference tests, scalar/cooperative equality,
Inf/NaN class preservation, validation, immutable inputs, and resident recorder execution. The worst
relative differences were 1.566e-5 against the GGUF reference and 2.373e-5 cooperative versus scalar.

`loader.txt` proves compressed IQ3 bytes survive GGUF admission and execute a forward projection.
`recorder-integration.txt` proves both types select the recorder-only resident upload route.

`whole-model.txt` records a six-layer TinyLlama-shaped generation test. Scalar and cooperative modes
produced identical tokens. Three whole-token timing samples per mode gave:

| Format | Scalar | Cooperative | Gain |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 34.42 tok/s | 154.22 tok/s | 4.480x |
| IQ3_S | 19.77 tok/s | 135.75 tok/s | 6.868x |

## Three fresh-process campaigns

Each campaign ran seven samples per cell in a fresh process. Cooperative tests used 16 distinct
resident weights, scalar-A/cooperative/scalar-B ordering, Metal command-buffer timestamps, and host
wall timing. Host-route tests used eight distinct weights and equal input/output transfer boundaries.
The four M=1 decoder shapes were N512/K2048, N2048/K2048, N5632/K2048, and N2048/K5632.

All raw observations, including outliers, are in `campaign-1.txt` through `campaign-3.txt`; the 24
campaign medians are in `aggregate.tsv`. Conservative floors across both formats, all shapes, and all
three campaigns are:

- cooperative versus scalar GPU time: 5.592x
- cooperative versus scalar recorder wall time: 4.232x
- resident Metal versus fused ARM64 CPU wall time: 2.060x

The unchanged scalar-A/scalar-B median control is 1.000x to 1.072x. The large retained margins are not
explained by control drift.

Direct host Metal fails the 1.10x all-cell gate, reaching only 0.4114x CPU in the worst median. Generic
`Backend.UploadQuant` therefore returns `backend.ErrQuantUnsupported` for IQ3, while `llamagpu` uses
the explicit recorder-only resident upload APIs.

## Static and repository gates

The pinned external `github.com/jxsl13/perfscan/perfscan@v1.71.0` ran with `GOPROXY=direct` against the
exact base and candidate. Production counts are 778 versus 778; test-inclusive counts are 1,811 versus
1,811, with identical per-rule multisets and no IQ3 finding. See `perfscan.txt`.

`make preflight` passed formatting, pure-Go build, vet, module drift, and the complete short suite.
`make preflight-metal` passed the cgo/Metal suites for `backend/metal` and `llamagpu`.

Spectackle check emitted no drift record for this tranche. The repository retains unrelated legacy
warnings and 16 stale anchors. Spectackle 0.10.0's Homebrew binary is built with Go 1.25, so typed-call
indexing cannot load this Go 1.26 module; the reproduction is recorded on Spectackle issue 274.

The generalizable cooperative serial-K transformation, persistent codebook residency, and measured
host-route boundary are reported on perfscan issue 565:
https://github.com/jxsl13/perfscan/issues/565#issuecomment-5379985510

## Reproduction

```sh
GOTOOLCHAIN=go1.26.6 go test -c ./backend/metal -o /private/tmp/goai-metal-iq3-metal.test
/private/tmp/goai-metal-iq3-metal.test -test.run '^$' -test.bench '^BenchmarkMetalIQ3(Cooperative|HostRoute)Campaign$' -test.benchtime=1x -test.count=7
GOTOOLCHAIN=go1.26.6 go test -c ./llamagpu -o /private/tmp/goai-metal-iq3-llamagpu.test
/private/tmp/goai-metal-iq3-llamagpu.test -test.run '^TestMetalIQ3CooperativeEndToEnd$' -test.v
GOPROXY=direct GONOSUMDB=github.com/jxsl13/perfscan GOTOOLCHAIN=go1.26.6 go run github.com/jxsl13/perfscan/perfscan@v1.71.0 -config internal/perfscan/perfscan.json -tests ./backend/metal ./nlp ./llamagpu
```

The test-binary filters are intentional: the repository contract forbids `go test -run`.
