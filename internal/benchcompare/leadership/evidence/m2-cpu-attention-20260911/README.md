# M2 CPU attention rejected pilot (2026-09-11)

This directory preserves a minimal, sanitized record of a withdrawn ARM64 SIMD causal-attention backward prototype. It is evidence of a rejected experiment, not a performance promotion or an external/model speedup. `candidate.patch` is **WITHDRAWN / NOT FOR PROMOTION** and is not applied to the runtime.

## Reproduction and data boundary

The private captures contain binaries, per-invocation command/result metadata, stdout/stderr, profiles, and local paths. They remain outside Git; private documents and books likewise remain local. `samples.csv` contains only enum and numeric fields, retaining every warmup and measured result without outlier filtering. Invalid `pilot-01` is preserved privately in full but excluded here because unrelated compiler/indexer contention was discovered before review. Planned `confirmation-02` and `confirmation-03` were **NOT RUN** after the first valid confirmation made the fixed three-campaign qualification impossible; no campaign was replaced or resampled.

The two included campaigns each used one excluded warmup pair plus seven measured alternating-order pairs, two arms, and nine cases: seven backward cells, one forward control, and one whole GPT training step. `pilot-02` used 500 ms for attention cases and 2 s for GPT; `confirmation-01` used 2 s for every case. The fixed environment was Apple M2 Pro (12 cores), Go 1.27.1, `CGO_ENABLED=0`, `GOEXPERIMENT=simd`, `GOMAXPROCS=12`, `GOGC=100`, and no memory limit. The builder validates both manifests, all 48 invocations per campaign, direct exit/PASS/stderr status, stdout byte/SHA integrity, exact cases and metrics, and the binary contents before exporting 288 rows.

Pins:

- baseline source commit: `2bc5836f5112f34772a0bbc6790da79fc86c953c`; restored `backend/cpu/mha.go`: `bfeb5d479c52bccd774782e376b32fbcbb4c596a400aefa34a429f0658c4a43c`
- shared benchmark harness `internal/benchcompare/cpu_attention_test.go`: `d2516455ddd0a5611dcee0ea00778a48b3c9c884eee77c7834faa1d6c484376c`
- baseline binary: `072377ad1163bf0544245f653fce75d88bb5a515968ed5de736ebb257ed9f035`; candidate binary: `efd5eb4eedcf58804000d43fac106406b2f12086a0a35566c2ca7beeff373f35`
- confirmation capture runner: `f0c39bb6523b911ded99d2a547a988bab342f155be339b98d701382ed65782a7`
- withdrawn reconstruction: `mha.go` `ecb83935f8539c2e8dc21df547ca9b370eea920a0282ce852f2d3558d95c3b59`, ARM64 helper `b1307191c9cbe2fb17c71c49a97017579c6a10bd3610ebe1fe105d22937f2ee4`, default stub `1099b99ce816e46cb61c27985abaf4c0ebd3d7447d9c01529e0df2e7cd7f9709`

From this directory, build, verify, self-test, and summarize with:

```sh
ruby build_samples.rb <private-artifact-root> samples.csv
ruby verify_samples.rb samples.csv
ruby verify_samples.rb --self-test
ruby verify_samples.rb --summary samples.csv
```

Determinism is checked by generating to a second output path and comparing it byte-for-byte with `samples.csv`. To audit the withdrawn source, run `git apply --check candidate.patch` at source commit `2bc5836f5112f34772a0bbc6790da79fc86c953c`, apply it only in a disposable private tree, and compare the three reconstructed SHA-256 values above. Do not apply it to a live/runtime checkout.

## Result and disposition

Medians are over the seven measured samples per arm; p-values are two-sided Mann–Whitney U results from benchstat, which treats the samples as independent. Negative latency deltas favor the candidate.

| Campaign | Case/metric | Baseline median | Candidate median | Delta | p | Interpretation |
|---|---|---:|---:|---:|---:|---|
| pilot-02 | causal S256 latency | 874,784 ns/op | 796,323 ns/op | -8.97% | .007 | Candidate spread was 11.15%, failing the required near-5% spread for a sub-10% claim. |
| pilot-02 | causal S512 latency | 3,093,938 ns/op | 2,512,238 ns/op | -18.80% | .001 | Preliminary operator signal only. |
| pilot-02 | GPT latency | 60,138,834 ns/op | 58,742,589 ns/op | -2.32% | .073 | Not significant. |
| confirmation-01 | causal S256 latency | 872,829 ns/op | 787,617 ns/op | -9.76% | .535 | Failed significance; baseline/candidate spreads were 10.39%/31.10%. |
| confirmation-01 | causal S512 latency | 3,039,470 ns/op | 2,503,479 ns/op | -17.63% | .017 | Did not rescue the campaign-wide failed gate. |
| confirmation-01 | GPT latency | 60,932,114 ns/op | 59,333,009 ns/op | -2.62% | .805 | Not significant. |
| confirmation-01 | GPT B/op | 380,304,171 | 387,948,399 | +2.01% | .007 | Significant measured allocation-byte regression. |
| confirmation-01 | GPT allocs/op | 2,744 | 2,750 | +0.22% | .005 | Significant measured allocation-count regression. |

The causal backward targets remained at 18 allocs/op in every baseline and candidate sample. The observed GPT allocation increase has no proved source-level cause; the `+6` count must not be interpreted as one allocation per layer. `confirmation-01` was valid but non-qualifying, so continuing the predeclared three-campaign sequence could not produce three qualifying confirmations without forbidden replacement. The conservative futility stop therefore rejected and withdrew the prototype.

The original runtime is restored. Root verification exercised the 44-case frozen oracle and full short CPU package through the new CI guard, but native exact-head execution on all three CI operating systems remains pending. Broader validation is not all green: the unchanged baseline still has the known NLP diffusion/PRM failures and ten markdown-lint findings. No tolerance, threshold, or validation result was relaxed.
