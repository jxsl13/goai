# M2 Metal GPT dispatch-fusion evidence — 2026-08-24

Status: **promoted** for the declared Apple M2 Pro GPT-2-small F32 cell.

## Claim cell and pins

- Base: `3c2ded37ac1886c64e8163d90062f8d860ee2363` (`origin/main` after PR 1204).
- Hardware: MacBook Pro Mac14,10, Apple M2 Pro, 12 CPU cores, 32 GB unified memory.
- Software: macOS 26.5.1 (25F80), Go 1.27.0 darwin/arm64, Spectackle 0.10.0.
- Model geometry: GPT-2 small, F32, vocab 50,257, context 1,024, width 768,
  12 heads, 12 blocks, FFN width 3,072, batch 1.
- Decode boundary: public `GPTDecoder.Step`, 16-token prefill excluded, sequential
  cache positions, complete LM-head projection and 50,257-logit download included.
- Prefill boundary: public `GPTDecoder.StepNLast`, complete transformer and cache
  population, final-row LM-head projection and download included.

Frozen measurement binaries:

| Binary | SHA-256 |
|---|---|
| Pre-fusion public-Step control | `3f583b2a12d58e35c994c11c889767e36253616807c99f8d6033ed9a6a168f1f` |
| Residual paired campaign | `b206383c56ff9f756aed60a1760cc3d59d3e02d486ac68fef44bf22f59275743` |
| QKV paired campaign | `40470c5ec85a365d38fd12c11a98e69050de053ab6a3724316c74ee2f9f9cf06` |
| Clean production candidate | `6577de2803fd40832b05635e456530d1413983a7c364f317d5f4b8a9c1e80798` |

## Design promoted

1. Attention-output and FFN-down projections accumulate into the running
   residual with `recordAdd`; the standalone residual `Binary` dispatches are gone.
2. Metal stores one row-major `[D,3D]` QKV weight per block. Single-token decode
   performs one grouped matrix multiplication and directly blits its K and V bands.
3. Prefills below 64 rows perform one grouped matrix multiplication and three
   output-band copies. Prefills at 64 rows or above execute three MPS matrix
   multiplications over strided, zero-copy views of the same resident weight.
4. Grouped-output scratch is bounded to `min(context,63)*3*D` floats. At this
   geometry it is 580,608 bytes; no split QKV weight copy remains. The rejected
   dual-resident prototype would have added 84,934,656 bytes, approximately 17%
   of GPT-2-small's F32 parameter storage.
5. F32 `recordAdd` does not consume a projection scratch, so the obsolete
   context-sized attention-output and FFN-output buffers are removed. This saves
   another 6,291,456 bytes at the declared geometry.
6. Vulkan and CUDA constructors do not opt in and retain the existing three-weight path.

## Paired single-token campaigns

Each slice was isolated in one process and measured over 21 aligned pairs. Each
arm executes 128 sequential public `Step` calls at identical positions. The first
campaign compares split projection plus split residual against fused residual.
The second keeps fused residual in both arms and compares three QKV projections
against one grouped projection. All 42 pairs are wins.

| Pair | Residual control ns/token | Residual candidate ns/token | Speedup | QKV control ns/token | QKV candidate ns/token | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18,747,032 | 18,475,518 | 1.014696x | 17,464,332 | 15,575,618 | 1.121261x |
| 2 | 19,321,083 | 17,509,546 | 1.103460x | 17,618,391 | 16,202,589 | 1.087381x |
| 3 | 18,848,505 | 17,750,378 | 1.061865x | 18,095,012 | 15,844,066 | 1.142069x |
| 4 | 19,395,354 | 18,512,089 | 1.047713x | 17,766,924 | 15,851,032 | 1.120869x |
| 5 | 19,159,614 | 17,749,468 | 1.079447x | 17,253,339 | 15,787,000 | 1.092883x |
| 6 | 19,389,502 | 17,760,146 | 1.091742x | 17,206,394 | 16,908,432 | 1.017622x |
| 7 | 19,646,896 | 18,453,113 | 1.064693x | 20,303,533 | 17,373,363 | 1.168659x |
| 8 | 19,601,538 | 18,276,775 | 1.072483x | 18,746,404 | 16,234,196 | 1.154748x |
| 9 | 19,310,481 | 17,824,611 | 1.083361x | 18,164,467 | 16,307,220 | 1.113891x |
| 10 | 18,539,222 | 17,219,475 | 1.076643x | 18,245,353 | 15,950,390 | 1.143881x |
| 11 | 18,476,316 | 17,086,115 | 1.081364x | 16,455,456 | 15,383,229 | 1.069701x |
| 12 | 18,355,759 | 16,370,376 | 1.121279x | 16,874,875 | 15,287,557 | 1.103831x |
| 13 | 17,614,364 | 16,930,338 | 1.040402x | 16,643,431 | 15,379,842 | 1.082159x |
| 14 | 18,481,279 | 16,170,391 | 1.142909x | 16,640,522 | 15,200,867 | 1.094709x |
| 15 | 17,664,016 | 16,192,539 | 1.090874x | 16,752,183 | 14,845,388 | 1.128444x |
| 16 | 17,526,906 | 16,434,952 | 1.066441x | 16,087,791 | 14,352,544 | 1.120902x |
| 17 | 18,575,250 | 17,485,968 | 1.062295x | 15,594,956 | 14,096,468 | 1.106302x |
| 18 | 19,056,505 | 17,229,556 | 1.106036x | 18,394,668 | 15,695,251 | 1.171989x |
| 19 | 18,564,260 | 18,026,607 | 1.029826x | 16,295,652 | 15,508,158 | 1.050779x |
| 20 | 19,156,921 | 18,095,003 | 1.058686x | 15,998,579 | 14,364,407 | 1.113765x |
| 21 | 19,270,872 | 17,822,869 | 1.081244x | 16,211,385 | 14,604,109 | 1.110056x |

| Slice | Paired median | Weakest pair | Winning pairs | Gate |
|---|---:|---:|---:|---:|
| Residual epilogues | **1.076643x** | 1.014696x | 21/21 | median >=1.03x, no regression |
| Grouped QKV decode | **1.113765x** | 1.017622x | 21/21 | median >=1.03x, no regression |

Commands:

```text
GOAI_GPT_RESIDUAL_CAMPAIGN=1 /private/tmp/goai-gpt2-residual-campaign-20260824.test -test.run '^TestGPTResidualFusionCampaign$' -test.count=1 -test.v -test.timeout=10m
GOAI_GPT_QKV_CAMPAIGN=1 /private/tmp/goai-gpt2-qkv-campaign-20260824.test -test.run '^TestGPTQKVFusionCampaign$' -test.count=1 -test.v -test.timeout=10m
```

## Cumulative production boundary

A separate seven-pair campaign alternated process lead order and compared the
frozen pre-fusion public-Step binary with the clean production candidate. Each
process ran exactly 64 sequential steps after untimed model upload, prefill, and
pipeline warm-up. The candidate won all pairs, with identical 217,808 B/op and
12 allocs/op.

| Pair | Lead | Control ns/op | Candidate ns/op | Speedup |
|---:|---|---:|---:|---:|
| 1 | control | 16,719,552 | 13,640,411 | 1.225737x |
| 2 | candidate | 16,768,069 | 13,617,054 | 1.231402x |
| 3 | control | 16,973,908 | 13,703,077 | 1.238693x |
| 4 | candidate | 16,559,807 | 13,638,247 | 1.214218x |
| 5 | control | 16,905,290 | 13,632,697 | 1.240055x |
| 6 | candidate | 16,769,554 | 13,736,004 | 1.220847x |
| 7 | control | 16,628,372 | 13,802,316 | 1.204752x |

Paired median: **1.225737x**. Worst pair: **1.204752x**. Winning pairs: **7/7**.

Standalone candidate cohorts ranged from 55.01 to 73.40 tok/s median, showing
why no claim compares independent process cohorts. All promoted speedups above
come from same-session or order-alternated paired campaigns.

## Prefill guard cells

The same-binary cumulative prefill campaign compared the merged-main-equivalent
split design with the final single-resident-weight design:

| Prompt | Split median | Candidate median | Speedup |
|---|---:|---:|---:|
| 16 | 18.163 ms | 15.677 ms | **1.158576x** |
| 64 | 18.066 ms | 17.010 ms | **1.062081x** |
| 256 | 23.076 ms | 22.034 ms | **1.047291x** |

The strided-view slice alone was approximately 1% slower at 64 rows and 2%
slower at 256 rows than retaining duplicate split weights, but it preserves the
larger cumulative win while avoiding 84.9 MB of duplicate device storage.

Final production commands:

```text
/private/tmp/goai-llamagpu-qkv-20260824.test -test.run '^$' -test.bench '^BenchmarkGPTDecodeStepMetal$' -test.benchtime=64x -test.count=5 -test.timeout=10m
/private/tmp/goai-llamagpu-qkv-20260824.test -test.run '^$' -test.bench '^BenchmarkGPTPrefillLastMetal$' -test.benchtime=3x -test.count=5 -test.timeout=10m
```

## Correctness, portability, and rejected alternatives

- `TestRecorderMatMulStridedB` passed count 3 against an exact CPU result.
- `TestGPTDecoderMatchesReference` passed count 3 with a 2e-3 relative/absolute gate.
- `TestGPT2ScalePipeline` generated 256 tokens exactly equal to the analysis-path
  full forward; its subsequent 64-token sample measured 69.6 tok/s.
- The ownership and portable-fallback white-box tests passed count 3.
- `go test -short -count=1 ./backend/metal ./llamagpu` and focused `go vet` passed.
- The Vulkan-tagged `llamagpu` test binary compiled, and the portable ownership
  contract passed count 3. CUDA remains protected by its unchanged opt-out path
  and the repository's Linux CUDA compile/link CI lane.
- Persistent/async encoding was rejected: prior research `R-01M01Z6AC8F42`
  reduced host encode approximately 5x without throughput gain because encoding
  already overlaps GPU execution.
- Retaining fused and split QKV weights was rejected despite slightly faster large
  prefills because it duplicates approximately 17% of F32 GPT-2-small weights.

Generalizable findings:

- [perfscan #881: recognize recorder matmul followed by a fusible residual add](https://github.com/jxsl13/perfscan/issues/881)
- [perfscan #882: detect execution-policy drift between sibling decoders](https://github.com/jxsl13/perfscan/issues/882)
- [perfscan #883: detect fused-weight optimizations that retain duplicate unfused weights](https://github.com/jxsl13/perfscan/issues/883)
- [perfscan #884: detect dead projection scratch retained after accumulate-epilogue fusion](https://github.com/jxsl13/perfscan/issues/884)

This is not a universal leadership claim. It applies only to the declared
hardware, model geometry, F32 representation, batch, context, and timed boundaries.
