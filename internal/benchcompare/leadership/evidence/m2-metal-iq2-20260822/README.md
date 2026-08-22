# M2 native Metal IQ2 evidence

Date: 2026-08-22
Base: origin/main 368ac4ca1081c6d7ea799e61b0bc26b551468c8e
Candidate: this evidence directory's containing commit
Hardware: Apple M2 Pro, 12 CPU cores, 32 GiB unified memory
Software: macOS 26.5.1 (25F80), Go 1.26.6 darwin/arm64, Spectackle 0.10.0

## Claim boundary

This tranche adds exact native Metal kernels and end-to-end loader/recorder support for GGUF type 16
IQ2_XXS and type 17 IQ2_XS. It publishes two separate dispatch decisions:

- Generic host input/output remains on the fused ARM64 CPU implementation because standalone Metal
  submissions lose several format/shape cells.
- Device-resident decoding uploads each compressed weight once and records a cooperative IQ2 matvec
  in the whole-token command buffer. This is the promoted production path.

Both scalar controls and cooperative production kernels use the same persistent codebook buffer.
The 256x8 and 512x8 codebooks are reconstructed once through the public exact GGUF decoder, avoiding
an independent hand-maintained Metal copy of the IQ2 tables.

## Correctness and reachability

`correctness.txt` records arbitrary packed-block cross-reference tests, scalar/cooperative equality,
Inf/NaN class preservation, validation, immutable inputs, and resident recorder execution. The worst
relative difference was 1.510e-5, below the 1e-4 cross-backend contract.

`loader.txt` proves compressed IQ2 bytes survive GGUF admission and execute a forward projection.
`recorder-integration.txt` proves both types select the explicit recorder-only resident upload route.
The Phi-3 block geometry and byte slicing are separately pinned by
`TestPhi3IQ2RowSlicingPreservesCompressedBytes`.

`whole-model.txt` records a six-layer TinyLlama-shaped generation test. Scalar and cooperative modes
produced identical tokens. Three whole-token timing samples per mode gave:

| Format | Scalar | Cooperative | Gain |
| --- | ---: | ---: | ---: |
| IQ2_XXS | 35.85 tok/s | 188.61 tok/s | 5.262x |
| IQ2_XS | 29.65 tok/s | 175.94 tok/s | 5.933x |

## Three fresh-process campaigns

Each campaign ran seven samples per cell in a fresh process. Cooperative tests used 16 distinct
resident weights, scalar-A/cooperative/scalar-B ordering, Metal command-buffer timestamps, and host
wall timing. Host-route tests used eight distinct weights and equal input/output transfer boundaries.
The four M=1 decoder shapes were N512/K2048, N2048/K2048, N5632/K2048, and N2048/K5632.

All raw observations are in `campaign-1.txt` through `campaign-3.txt`; the 24 campaign medians are
in `aggregate.tsv`. Conservative floors across both formats, all shapes, and all campaigns are:

- cooperative versus scalar GPU time: 5.164x
- cooperative versus scalar recorder wall time: 4.009x
- resident Metal versus fused ARM64 CPU wall time: 1.368x

The unchanged scalar-A/scalar-B median control stays between 1.000x and 1.007x. Direct host Metal
fails the 1.10x all-cell gate, reaching only 0.2824x CPU in the worst campaign median. Generic
`Backend.UploadQuant` therefore returns `backend.ErrQuantUnsupported` for IQ2, while `llamagpu`
uses explicit recorder-only resident upload APIs.

## Static and repository gates

The pinned external `github.com/jxsl13/perfscan/perfscan@v1.71.0` ran with `GOPROXY=direct` against
the exact base and candidate. Production counts are 778 versus 778; test-inclusive counts are 1811
versus 1811, with zero candidate additions. See `perfscan.txt`.

`make preflight` passed formatting, pure-Go build, vet, module drift, and the complete short suite.
`make preflight-metal` passed the cgo/Metal suites for `backend/metal` and `llamagpu`.

The generalizable cooperative serial-K transformation, persistent codebook residency, and measured
host-route boundary are reported on perfscan issue 565:
https://github.com/jxsl13/perfscan/issues/565#issuecomment-5380389629

## Reproduction

```sh
GOTOOLCHAIN=go1.26.6 go test -c ./backend/metal -o /private/tmp/goai-metal-iq2-metal.test
/private/tmp/goai-metal-iq2-metal.test -test.run '^$' -test.bench '^BenchmarkMetalIQ2(Cooperative|HostRoute)Campaign$' -test.benchtime=1x -test.count=7
GOTOOLCHAIN=go1.26.6 go test -c ./llamagpu -o /private/tmp/goai-metal-iq2-llamagpu.test
/private/tmp/goai-metal-iq2-llamagpu.test -test.run '^TestMetalIQ2CooperativeEndToEnd$' -test.v
GOPROXY=direct GONOSUMDB=github.com/jxsl13/perfscan GOTOOLCHAIN=go1.26.6 go run github.com/jxsl13/perfscan/perfscan@v1.71.0 -config internal/perfscan/perfscan.json -tests ./backend/metal ./nlp ./llamagpu
```

The test-binary filters are intentional: the repository contract forbids `go test -run`.
