# Allocation sweep (Go heap, `-memprofile`)

A CPU profile and an allocation profile rank different things, and this repo's
CPU side was already worked over when this sweep ran. Everything below came
from `go test -bench . -memprofile` on a package, then
`pprof -sample_index=alloc_space -list` on the top own-code entries.

## Shipped

| site | change | allocs/op | time |
|---|---|---|---|
| `nlp` TurboQuant KV cache `rows`/`reconstruct` | output rows from one slab; temporaries to per-chunk scratch; scratch pooled | 3100 → 38 (**−98.8%**) at 128×512; 6172 → 40 at 256×1024 | **−66%** |
| `classic` `GaussianMixture.PredictProba` output rows | one slab, capacity-capped views | 605 → 94 (−84.5%); 2141 → 94 at 2048×8 | −11.2% at k=4 |
| `classic` GMM per-worker density scratch (3 closures) | one `sync.Pool` | 94 → 34 (−63%); GMMFit 2652 → 2192 | ~ (−0.6%) |

The last two are the same transform at different positions in the loop nest and
they answer the speed question differently — see PS2008's entry, which was
qualified as a result.

## Declined, with reasons — do not re-derive these

- **`nlp.rowLogits` (9.89GB, the largest single site).** 99.79% of it comes from
  `benchRowLogits`; the real `Generate` paths contribute 0.2%. Optimizing it
  moves the benchmark that exists to measure it.
- **`nlp.(*Sampler).Dist` (1.48GB).** Every caller in the profile is a
  `Benchmark` function, and the allocation is inherent to returning a fresh
  distribution — which is why `distInto` already exists for callers that do not
  want one.
- **`nlp.MLMMaskExcluding`, `Watermark.BiasLogits`.** API-inherent: they return
  freshly allocated slices.
- **`tensor.heapAllocator.Alloc` (89.6% of `tensor`, 51.5% of `classic`).** Every
  backend op allocates its output tensor. Pooling them is a question about
  ownership and lifetime across the backend API, not a local fix.
- **`tensor.(*Tensor).Permute` / `.Slice`.** Already optimal: `copyShapeStrides`
  packs shape and strides into a single `make([]int, 2*n)` with capped views. What
  remains is the `&Tensor{}` view struct itself, ~1.3% of the package's
  allocations; folding it into one allocation means embedding a fixed-size dim
  array in a foundational type, which is not worth 1.3%.
- **`classic.SoftmaxRegression.PredictProba`'s input tensor (1.39GB).**
  `tensor.New` per call. Same ownership question as the allocator entry above.
- **`classic/tree.go` (`buildIdx`, `DecisionTreeClassifier.Predict`).** Not this
  branch's lane.

## Method notes that cost time to learn

- **Check the callers before the line.** Two of the three largest sites in `nlp`
  were benchmark scaffolding. `pprof -peek` on the symbol answers this in one
  command and should come before reading any code.
- **A returned row carved from a slab needs its own test.** Value-level tests pass
  either way — the numbers are identical — so nothing notices if rows start
  sharing storage or if a view can be appended past its end into its neighbour.
  Both shipped slabs above hand rows to the caller, and both have a
  mutation-verified independence test.
- **Pool sets must be sized on `get`.** A process can hold models or caches of
  different geometry; a set recycled from a smaller one has to be grown, not
  silently reused too short.

## Other axes swept, and found clean

Recorded so they are not re-run: each of these was profiled across the lanes
above and produced nothing actionable.

- **Blocking (`-blockprofile`, vision).** 83% of all delay is `runtime.chanrecv`,
  and 61% of the total is pool workers parked on their task channel — idle
  workers waiting for work, not contention. The actual barrier cost, the
  caller's `sync.WaitGroup.Wait` inside `parallelWork`, is 2.86% of delay. A
  block profile of any pooled program is dominated by expected idleness; read
  the `WaitGroup` line, not the total.
- **Mutex contention (`-mutexprofile`, vision and classic).** ~0.42s of delay in
  each, of which 78–85% is `runtime.unlock` — runtime-internal locks from the
  allocator, GC and scheduler, not application mutexes. `sync.(*Mutex).Unlock`
  is 62ms in vision and 8ms in classic. There is no lock contention to fix, and
  the runtime-lock share is itself reduced by the allocation work above rather
  than by anything lock-specific.
- **`backend/cpu` allocations.** The package allocates almost nothing of its own:
  `getF64` and `getF64Raw` together are 1.1GB of 91.9GB (1.2%), and those are
  pool MISSES, i.e. the pools working. Its large cumulative figures — `binOp` at
  12.3GB, `addBiasKernel` at 6.6GB — are the per-op output tensors, the same
  ownership question listed under "Declined" above.

The pattern across all four axes is that the remaining allocation mass in this
repo is output-tensor construction at the backend boundary. That is one design
decision, not a set of local fixes.
