# Parallelizing the host loops (§V22-measured sweep, 2026-07)

Techniques applied to the CPU-side loops in `classic/`, `nn/`, `linalg/` and
`format/gguf`, and the allocation work that followed. A/B numbers observed on
darwin/arm64 (M-series, GOMAXPROCS=12), interleaved, minimum of 3 runs per arm.
Every change kept a **bit-identical** result, proven by an explicit gate rather
than argued from the code.

This doc is about *where the parallelism was missing and how to tell*, not about
the compute kernels — those are at the BLAS/SIMD ceiling and out of scope.

## 1. Find serial spines by measurement, not by reading

Run a benchmark at `GOMAXPROCS=1` and at full width and divide. Substantial
ns/op with a ratio near **1.0** means nothing in it parallelizes. Shipped as
`internal/perfscan/tools/scaling_sweep.sh`.

This is a script and **not** a perfscan rule on purpose: proving a loop's
iterations are independent is a dataflow question, and perfscan is AST-only. A
static rule guessing at independence would advise data races.

| benchmark | before | after | |
|---|---|---|---|
| `EncodeAQLM_256x256` | 990 ms | 223 ms | 4.44× *(composed with an unroll from main)* |
| `MuonStepOnly` | 193 ms | 41 ms | 4.63× |
| `KNNPredict` | 96.6 ms | 13.8 ms | 7.00× |
| `GMMFitFull` | 76.5 ms | 18.8 ms | 4.09× |
| `GBMHist_exact_80k` | 1865 ms | 627 ms | 2.80× |
| quantized decode (7 types) | 525–1082 µs | 182–322 µs | — |
| `OrthogonalFast` (Householder QR) | 48.3 ms | 35.9 ms | 1.37× |
| `GBMHist_hist_80k` | 337 ms | 214 ms | 1.57× |
| SOAP / Shampoo step | 8.1 / 5.9 ms | 6.1 / 5.3 ms | 1.32× / 1.10× |

## 2. What decides whether parallelizing pays

**The enclosing parallelism, not the loop.** A random forest already
parallelizes across trees (7.85×), so splitting the CART sweep *inside* each
tree is inert — nested dispatches find the pool busy and run inline. Boosting is
sequential across trees, so the identical inner loop was worth 2.80× there. Same
code, opposite verdict, decided entirely by what surrounds it.

**Amdahl bounds the answer before you start.** QR stops improving past 4 Ps
because it peels a column per step, so later reflectors fall under the work
threshold. GBM flattens because `partition` was still serial. Both are correct
behaviour, not defects.

**A perfect null is usually a threshold, not a transform.** The GMM E-step
measured exactly 1.00× until the work estimate was corrected: a full-covariance
Mahalanobis is O(d²) per component, and passing `k` alone understated it by a
factor of `d`, so the fit never crossed the crossover. It is 1.93× once the
guard fires.

## 3. Keeping it bit-identical

**Split the search from the fold.** When an expensive independent search feeds
an order-dependent reduction — k-means assignment writing `sums[b]`, an E-step
accumulating a log-likelihood — run the *search* in parallel into an array and
fold *sequentially* afterwards. Chunked partials reassociate the sums. The wrong
version is quicker to write and passes any test that checks reproducibility
rather than preservation. Detected by `PS6007`.

**Per-chunk scratch, hoisted.** A shared scratch buffer is what usually blocks a
loop, and it appeared four times in unrelated code (`logGaussian`'s solve
buffer, `gbmBuilder.vals`, `gbmBuilder.part`, AQLM's residual `target`).
Detected by `PS6006`. But allocating the replacement *inside* the parallel body
is only safe when the dispatch is infrequent — see §5.

**Combine an argmax in source order.** Recording one candidate per feature and
folding them with the same strict `>` in ascending order reproduces the serial
scan exactly, including which feature a tie selects. Any other order silently
grows a different tree.

## 4. The gates were the hard part

**Determinism tests prove reproducibility, not preservation.** `TestGMMDeterminism`,
`TestAQLMDeterministic` and `TestGBMHistogramDeterministic` all fit twice with the
*same* code and compare. A change that is deterministic and wrong passes every
one of them. Each needed a frozen bit-level reference captured from the
pre-change implementation.

**A differential gate covers only what differs between its arms.** A one-ulp
perturbation of `linalg.applyReflector` turned *nothing* red — including
`nn.TestOrthogonalBitIdenticalToSlowPath`, which looks like exactly the right
guard. Both arms call `applyReflector`, so perturbing shared code moves them
identically. Cast as `NUM-DIFFERENTIAL-GATE-001`.

**Comparison-decided output needs a constructed tie.** Random data never
produces two bit-equal gains or two bit-equidistant points, so tie-breaks are
untestable by fixture. Three separate cases needed a deliberately built tie: the
GBM split argmax, the AQLM ICM argmin, and KNN's `(dist, idx)` order — where
inverting the comparison left every existing test green. Cast as
`NUM-ARGMAX-TIEBREAK-001`.

**Probe the site you are about to change, not the file.** An earlier session
removed five bit-identity oracles after concluding each was "guarded elsewhere".
For `applyReflector` that conclusion did not hold.

## 5. Allocation is a separate axis, and it bites

Sweeping the *same* benchmarks by allocs/op immediately found a **31× memory
regression that had shipped** behind a 2.80× speedup: the GBM parallelization
allocated per-chunk scratch inside the parallel body, and that body runs once per
**tree node**.

| dispatch frequency | site | memory |
|---|---|---|
| once per encode pass | AQLM ICM | 49 → 51 MB — fine |
| once per EM iteration | GMM M-step | 4 → 4 MB — fine |
| **once per tree node** | GBM exact | **64 → 2007 MB** |

Identical code shape, three orders of magnitude apart. Detected by `PS6008`;
fixed with `parallel.RowsIdx`, which passes the chunk index so buffers live on
the caller's struct. Report allocs and bytes alongside every speedup
(`PROC-BENCH-MEMAXIS-001`) — a ns/op-only A/B would never have shown it.

The same axis found `sort.Slice` reaching its swap through
`reflectlite.Swapper`, which allocates on **every call** regardless of length:
`ForestFit` 1,095,700 → 352,027 allocs (3.11×), the KNN sites 1.50×. Triage by
**call frequency, not slice length** — the KNN sorts handle k results and still
paid. Detected by `PS6009`.

## 6. Measurement hygiene, learned the hard way

**A ratio of two noisy numbers is noisier than either.** A single run at
`-benchtime 12x` reported `CholeskyVJP_128` at 0.88× — *slower with more cores*,
which reads as false sharing. It is 1.09×. The sweep now defaults to a generous
benchtime and takes the minimum of 3 runs per arm
(`PROC-BENCH-MINOFN-001`).

**A mostly-`cond_wait` profile is not evidence of overhead.** Parked time is
attributed but free. Replacing the parking with a bounded spin measured **2–4%
slower** across three callers and was reverted. The scaling curve is the
evidence (`PROC-PROFILE-PARKED-001`).

**Do not parse `-benchmem` by field index.** A benchmark calling `SetBytes`
emits an extra `MB/s` column that shifts everything after `ns/op`, turning
`GPT2Encode`'s 37 allocs/op into a reported 10,565,506.

## 7. What was deliberately not done

- **CART sweep** (`classic/tree.go`) — a genuine spine at 1.00×, but forests
  already saturate the machine at the tree level, so it is inert there and worth
  only a lone 10 ms tree fit.
- **SVD one-sided Jacobi** — rotations mutate the columns later pairs read;
  parallel orderings change the rotation sequence, which is a different
  algorithm, not a rounding difference.
- **MLA VJP** — three of five gradient outputs accumulate across the loop index,
  and `dkRrot` is shared across heads because MLA decouples the RoPE key.
- **GPT/T5/CLA decode** — the bottleneck is `backend/cpu`'s dispatch, not the
  callers. GPT is fastest at **8** Ps and 14% *slower* at 12; that regression,
  not the profile's sync share, is what needs explaining.

## See also

- `internal/perfscan/PATTERNS.md` — PS6003, PS6010–PS6009, and the sweep method
- `docs/perf-notes-training.md` — the per-element host-loop sweep this follows
- `docs/perf-notes-lowlevel.md` — bounds checks, dispatch, and the SIMD floor
