# Apple M2 AMX generation experiment evidence

This directory publishes the source-only evidence for the Apple M2 Pro qualification of the first AMX generation-dispatch candidate. It does not publish executables, capture metadata, private paths, or raw command output. `candidate1.csv` is a lossless extraction of the numeric tokens printed by the frozen benchmark binaries.

Candidate 1 is rejected. In each of the three complete campaigns, both small cases increased from 272 to 288 B/op (`p=0.001`, seven measured observations per arm); allocations remained 4 allocs/op. Campaign 1 reported isolated timing differences for the 256 cube (-1.22%, `p=0.026`) and 1024 cube (-7.23%, `p=0.017`), but neither reproduced in campaigns 2 or 3. There is no reproducible timing improvement and this evidence makes no speed claim.

Candidate 2 has no completed benchmark campaign. The builder therefore fails closed and no `candidate2.csv` is published. Native Apple M1 qualification remains pending; this evidence makes neither an M1 speed claim nor an equivalence claim.

## Protocol boundary

Each campaign used Go 1.27.1 with `CGO_ENABLED=0` and the SIMD experiment, eight interleaved baseline/candidate pairs with alternating arm order, fixed 500 ms benchmark time, `GOMAXPROCS=12`, `GOGC=100`, `GOMEMLIMIT=off`, and an empty `GODEBUG`. Both arms used the same frozen full-GEMM harness. Packing and the existing worker dispatch were inside the benchmark timer; fixture construction and parity warmup were outside it. Pair 0 is a retained warmup and is excluded from analysis. Pairs 1 through 7 are repeated warm measurements, giving `n=7` per arm, case, and campaign; they are neither cold-start measurements nor M1 results. The CSV contains all five cases and both arms for all pairs: 240 rows total, including 30 warmup rows and 210 measured rows.

The reported significance checks used benchstat's Mann-Whitney test on the two independent arms within each campaign. Interleaving controls order but does not turn that test into a paired analysis. No cross-campaign pooling, filtering, outlier removal, rounding, or normalization was applied.

## Measured latency summary

Each cell is `median [min, max]` in ns/op over measured pairs 1 through 7. These descriptive values are generated from `candidate1.csv`; they are not a replacement for the campaign-level statistical decision.

| Campaign | Case | Baseline | Candidate |
| ---: | --- | ---: | ---: |
| 1 | m32_k64_n32 | 2,358 [2,316, 2,384] | 2,335 [2,284, 2,402] |
| 1 | m64_k64_n64 | 6,526 [6,363, 7,936] | 6,432 [6,319, 6,630] |
| 1 | m256_k256_n256 | 61,244 [60,532, 68,915] | 60,496 [60,174, 61,143] |
| 1 | m1024_k1024_n1024 | 1,042,954 [960,037, 1,140,660] | 967,527 [943,941, 1,003,747] |
| 1 | m512_k2048_n4096 | 5,319,538 [4,885,211, 7,692,362] | 5,409,125 [5,069,703, 5,534,565] |
| 2 | m32_k64_n32 | 2,328 [2,288, 2,354] | 2,341 [2,323, 2,367] |
| 2 | m64_k64_n64 | 6,440 [6,349, 7,924] | 6,340 [6,228, 7,573] |
| 2 | m256_k256_n256 | 60,967 [59,924, 62,877] | 61,738 [60,612, 71,562] |
| 2 | m1024_k1024_n1024 | 965,544 [941,149, 1,074,217] | 947,174 [934,965, 1,151,313] |
| 2 | m512_k2048_n4096 | 5,160,536 [5,028,277, 5,756,693] | 5,243,622 [4,837,386, 6,265,817] |
| 3 | m32_k64_n32 | 2,351 [2,328, 2,410] | 2,377 [2,330, 2,588] |
| 3 | m64_k64_n64 | 6,507 [6,311, 8,491] | 7,644 [6,358, 11,423] |
| 3 | m256_k256_n256 | 61,922 [60,617, 70,104] | 62,247 [60,373, 158,297] |
| 3 | m1024_k1024_n1024 | 985,600 [961,690, 1,750,698] | 1,033,483 [974,239, 1,401,160] |
| 3 | m512_k2048_n4096 | 5,604,540 [5,216,569, 7,881,661] | 5,491,398 [5,106,147, 7,691,460] |

Across all three campaigns, the small-case baseline/candidate median memory cells are 272/288 B/op and 4/4 allocs/op. The larger cases retain their measured per-run memory values in the CSV and are summarized by the validator on demand.

## Reproduction and validation

The public builder accepts only a named frozen plan, a private artifact root, and a new output path:

```text
ruby build_samples.rb candidate1 PRIVATE_ARTIFACT_ROOT OUTPUT_CSV
ruby build_samples.rb candidate2 PRIVATE_ARTIFACT_ROOT OUTPUT_CSV
ruby verify_samples.rb OUTPUT_CSV
ruby verify_samples.rb --summary OUTPUT_CSV
ruby self_test.rb
```

It verifies the three exact manifests, frozen runner and binaries, every command/result record, capture-helper identity, stream size and digest, empty stderr, successful exits, strict timeline and preflight freshness, exact benchmark matrix, and the alternating schedule before creating output. Existing output paths, including dangling symlinks, are refused. The shared parser independently validates the completed CSV matrix and the self-test exercises hermetic positive and adversarial fixtures. There is no public bypass, synthetic mode, or skip-verification option.

## Frozen identities

The experiment used Go 1.27.1 on `darwin/arm64` with the SIMD experiment enabled.

- Harness source: `081e2bc27e702a06605f9380161bac0db722724849eedd8a602bc2447b80d41d`
- Candidate 1 Go source: `d5da2e4b9147ca0da219cc48c616ae65cb263f86144d506e5a93a142dbde3d8f`
- Candidate 2 Go source: `1ab174178d3cb8ab559769af22e6824dedad04efbd10b0dbcdee11a48d8bb142`
- Shared baseline binary: `1e4af073226b5342bd0c03a5e4aac3f9ea7f7906870a5417b13afb009509743d`
- Candidate 1 binary: `305c818eed157aab881155c602daf8ad3da1b6ae85a56d78a6e43e5c92514860`
- Candidate 1 runner: `e1dedd1b056efbe821193bdf1c4b22c205fadf76a6179399210007e406e07ccf`
- Candidate 2 binary: `c10dc8a6683bff3606f9873fc74c4271a1c6c78b78efdb4c32b295e8df27fe46`
- Candidate 2 runner: `bfb8d5315a8600db582b68606a8e2fdaed6880542f36410dabeec9e90f9f2386`
- Capture helper: `69b09028b284504adc871e3531ba7a0b1f165486ae8ce5075d001d222f8ed691`

Tracking context: [issue 987](https://github.com/jxsl13/perfscan/issues/987) and [issue 989](https://github.com/jxsl13/perfscan/issues/989).
