---
schema: v1
---

## T-01KYJQ3QRHFRSTCA1Q521E99DX Make triangular back-substitution walk the solution contiguously
kind: task
state: draft
created: 2026-07-27

SITE: linalg/linalg.go:118 in (*LU).Solve; identically linalg/cholesky.go:78 CholSolve and linalg/qr.go:102 Lstsq — the same out[j*cols+c] shape verbatim at all three.

WHY HOT: this is the hottest in-repo linalg path and it is user-visible. nn/gptq.go:106 and nn/sparsegpt.go:133 call linalg.Inverse on an [in,in] Hessian PER QUANTIZED LINEAR LAYER, where in is the layer input width (768-4096 in practice). Inverse (linalg.go:130) is Solve(tensor.Eye(n)), i.e. cols = n. Factor is about n^3/3 flops; Solve with n right-hand sides is about n^3 — so the SUBSTITUTION phase, not the factorization, dominates Inverse.

DEFECT: the back-substitution accumulator reads the already-computed part of the solution as out[j*cols+c] with c (the outer column loop variable) fixed and j (the innermost variable) striding by cols. For cols = n = 512, consecutive inner iterations are 4KB apart: every read touches a new cache line and a new page, on a buffer of 2MB at n=512 and 134MB at n=4096, far larger than L1/L2. The row-major f.lu[i][j] factor read next to it is perfectly contiguous, which makes the asymmetry stark.

FIX: per column c, solve into a contiguous local x := make([]float64, n) (reuse y's allocation — allocate both once outside the c loop), i.e. s -= f.lu[i][j] * x[j], then scatter out[i*cols+c] = x[i] in one pass after the substitution. n*8 bytes stays L1-resident. Apply the identical edit at all three sites. OPTIONAL FOLLOW-UP, keep separate: for Inverse specifically the identity RHS lets forward substitution skip leading zeros (start at the first i with piv[i] == c), cutting the forward half roughly in half — but that one DOES change which terms are summed, so it must not be bundled with the layout fix.

VALIDATION GATE (benchmark only): linalg has ZERO benchmarks today. Write BenchmarkLUSolveMat over n in {64, 256, 512, 768} with a diagonally-dominant random [n,n] and Factor HOISTED OUT of b.Loop (that hoist is what isolates the substitution change from the factorization), BenchmarkInverse over the same n set doing full Factor+Solve (the GPTQ-shaped call), and BenchmarkCholSolveMat / BenchmarkLstsqMat for the sibling sites.

EXPECTED: 2-5x on the substitution phase for n >= 256; about 1.6-3x on Inverse end to end. Medium-high confidence — the mechanism is unambiguous, the constant is host-dependent, and M-series 128-byte lines make the waste ratio worse, which favors the fix.

BIT-IDENTITY BAR: bit-identical, and this is the point worth stressing — the fix reads THE SAME VALUES IN THE SAME ORDER into the same accumulator, changing only where those values live in memory. The numeric-parity tests in linalg_test.go / cholesky_test.go / qr_test.go hold unchanged and no RNG is involved anywhere. Still add a tolerance-0 golden comparison for one fixed matrix per site.

PERFSCAN RULE REQUIRED: major-axis-innermost traversal of a flat row-major buffer. AST shape: inside the innermost ForStmt, an IndexExpr on a []T whose index is a BinaryExpr of the form i*S + c where i is the INNERMOST loop's induction variable and c is an OUTER loop's, with S loop-invariant and not equal to the innermost trip count. Emit when the same buffer is also written at c*S + i order elsewhere, confirming row-major intent. Cheap, syntactic, high precision. A second shape belongs to the same class: an IndexExpr chain m[i][j] on a [][]T where i resolves to the innermost loop and j to an enclosing one — the loop nest order is the transpose of the storage order (live at linalg/qr.go:150-156 and :173-182, and svd.go:43-47/:68-72). Attach a warning to the suggested rewrite that any a*b*c reassociation must preserve the original grouping.

## R-01KYN45MA3FA2A0QPA4582JR0H DECLINED: SVD one-sided Jacobi cannot be parallelized bit-identically — the structure, not the effort, is the blocker
kind: research
state: draft
created: 2026-07-28

BenchmarkSVDPCA is a genuine serial spine (34.93ms, 1.00x across GOMAXPROCS 1..12, min of 3 per point) and linalg.SVD is 90% flat / 95% cumulative of it. Declined after analysis; recording which structural facts rule out each route so this is not re-derived.

SHAPE: m=2000, n=50, one-sided Jacobi. Per sweep, n(n-1)/2 = 1225 (i,j) column pairs; roughly 10 sweeps to converge.

ROUTE 1, ACROSS (i,j) PAIRS — ruled out by dependency. Rotating pair (i,j) MUTATES columns i and j, which later pairs in the same sweep read. Jacobi admits a parallel ordering (round-robin pairing of disjoint pairs), but that changes the SEQUENCE of rotations, hence the arithmetic, hence the result. Not a rounding difference to argue about: a different rotation order is a different algorithm.

ROUTE 2, INSIDE THE HOT LOOP — ruled out by size and by reduction. Line-level profile: the alpha/beta/gamma accumulation is 870ms of 1170ms flat (74%) and the column rotation 280ms (24%). The accumulation is a REDUCTION over k, so splitting it reassociates. The rotation is elementwise and WOULD be bit-identical under partition, but at m=2000 each rotation is about 8000 work units — an order of magnitude under the 1<<15 crossover, and correctly so: dispatch would cost more than the work. Parallelizing 24% of the time in chunks that small is a loss, not a small win.

ROUTE 3, ALGORITHMIC CACHING — ruled out by exactness, and this is the one worth knowing about. alpha = ||c_i||^2 is recomputed for every j even when c_i has not changed since the last pair (in converged sweeps most pairs skip rotation via the rel <= tol test). Caching it IS bit-identical, since recomputing the same data in the same order yields the same bits. But it saves only the alpha accumulation, roughly 250ms of 870ms, while the loads of c_i[k] are still needed for gamma — and in early sweeps nearly every pair rotates, invalidating the cache and adding overhead. Complexity high, payoff about 1.15x, benefit concentrated where the algorithm already spends least. Not attempted.

The classic Jacobi norm-update identities (alpha' = alpha - t*gamma) are NOT bit-identical to recomputing the sum and are therefore not available at all under this repo's tolerance-0 convention.

OBSERVATION FOR ANYONE REVISITING: line 46 (beta over c_j) costs 570ms against line 45 (alpha over c_i) at 250ms, for identical arithmetic. That 2.3x is cache behavior — c_i stays hot across the whole j loop while c_j streams. A blocked pair ordering would fix it and would also change the rotation sequence, so it runs into ROUTE 1.

Also swept and NOT flagged: vision scales well already (MLPMixer 4.26x, Swin 3.29x, ViT 4.40x), Pinv_64x32 is 1.00x but only 0.62ms, LUSolve_128x128 is 1.01x at 2.38ms.

## R-01KYN818R3F3W97NM7MN0EHWNG A fast-vs-slow bit-identity gate cannot guard code both arms share — applyReflector was blind under one that looked exactly right
kind: research
state: draft
created: 2026-07-28

Found while parallelizing Householder QR, and the gate finding matters more than the 1.37x.

THE MEASUREMENT: BenchmarkOrthogonalFast was 45.9ms at 0.97x. applyReflector is 52.8% cumulative; parallelizing its column loop gives 48.2-49.1ms -> 35.2-36.1ms, 1.37x, scaling 48.4 / 34.5 / 34.9ms at 1 / 4 / 12 Ps. It stops improving after 4 Ps because QR peels a column per step, so the trailing submatrix shrinks and later reflectors fall below the work threshold — correct behavior, not a defect.

THE GATE HOLE: a one-ulp perturbation of applyReflector's update turned NO test red. Not linalg's, and not nn.TestOrthogonalBitIdenticalToSlowPath — which looks like exactly the guard for this and structurally cannot be. It compares the FAST path against the SLOW path, and BOTH call applyReflector, so perturbing shared code moves both arms identically and the comparison still passes. A DIFFERENTIAL GATE COVERS ONLY WHAT DIFFERS BETWEEN ITS ARMS. TestQRReconstruct checks Q*R against A within a tolerance and passes a one-ulp shift comfortably.

This is the same family as the self-fulfilling-oracle failure already known here (a guard that reads the table it checks proves the table equals itself), but the differential form is harder to see: the test is genuinely useful, genuinely tolerance-0, and genuinely catches divergence between the two implementations. It just says nothing about their common subroutine, and the name gives no hint of that boundary. Cast as NUM-DIFFERENTIAL-GATE-001.

CORRECTION TO AN EARLIER SESSION: commit e7b27ee8 removed five bit-identity oracles including qr's, concluding each was "guarded elsewhere" after probing. For applyReflector that conclusion does not hold, whatever site was probed at the time. A frozen FNV hash over every element of Q and R replaces it — red under the exact mutation the existing suite missed, while TestQRReconstruct stays green. Worth noting the removal was itself a correction of over-testing, so the lesson is not "never remove an oracle" but "probe the SITE you are about to change, not the file".

STILL SERIAL: householder (34.7%) computes one reflector from a column norm — a reduction, so not bit-identically splittable, and the vectors shrink as QR proceeds.

## R-01KYQ8DP51EWHRPK3S54912JW4 PS6011 linalg sweep: SymEig 1.89x at n=128; rule corrected for hoisted-stride transposes (120->71)
kind: research
state: draft
created: 2026-07-29

Continued the PS6011 sweep into linalg and nlp. Two outcomes: one shipped optimization and
one rule correction that mattered more than the optimization.

SHIPPED — SymEig eigenvector accumulator stored transposed (internal/linalg/eigen.go):
n=64 2.91ms -> 2.72ms (1.04-1.09x), n=128 42.96ms -> 22.70ms (1.87-1.90x). M2 Pro
darwin/arm64, interleaved over 3 alternations, min of 3 runs of 5x per arm. The Jacobi
accumulator is ONLY ever column-rotated, so transposing its storage makes the walk
contiguous with identical arithmetic; the final extraction of an eigenvector (a column of v)
becomes a single copy for the same reason. SymEig had no benchmark despite backing SOAP,
Shampoo, GaLore, classic PCA and GMM, and backend/ref's eigh. Bit-identity held by the
existing oracle test that replays the original row-of-slices implementation.

NOT addressed: the m rotation's column loop. m is both row- and column-rotated in the same
step, so a transposed copy would have to be maintained in parallel — the traffic moves
rather than disappears.

THE SIZE THRESHOLD IS THE LESSON. At n=64 the matrix is 32KB and cache-resident, so the
optimization measures at 1.05x — indistinguishable from noise, and a benchmark sized only
there would have rejected a 1.9x win. Cast as PROC-BENCH-CACHE-THRESHOLD-001: a benchmark
validating a memory-access optimization must report at least two sizes, one exceeding cache.
This is the same residency principle as PERF-ACCUM-RESIDENCY-001 seen from the measurement
side rather than the design side.

RULE CORRECTION — PS6011 was flagging ALREADY-CORRECT code. The transpose exclusion is
syntactic and loses the stride once it is hoisted: with `row := i*b` outside the inner loop,
`dst[j*a+i] = src[row+j]` shows no multiplication of the outer variable. nlp's gguf weight
transposes are already tiled, the correct form, and were being flagged anyway — the worst
kind of finding, since it advises rewriting code that is right. Now excludes permutation
copies generally (one indexed write fed by one read, with or without a conversion or
accessor), including the generic AtF64 fallback arms. Tree-wide findings 120 -> 71, a 41%
noise cut, with all three original motivating cases still caught on replay. Kept honest by a
test asserting the check STILL fires on a strided accumulation written in the same
assignment shape.

REMAINING PS6011 after the correction: 71 tree-wide. backend/* (the parallel worker's lane)
holds the bulk. In linalg, the cholesky/qr/linalg.go pairs at adjacent lines are symmetric
fills and transposes; svd.go and derived.go are unswept and have partial benchmark coverage
(BenchmarkSVDPCA, BenchmarkPinv). nlp is now clean.

## R-01KYQBJ7BREF4SA4JX1G8VAV56 Rejected: SymEig column-walk cannot use row reads — Jacobi asymmetry accumulates across all rotated pairs
kind: research
state: draft
created: 2026-07-29

REJECTED after implementation and measurement: replacing the column reads in SymEig's Jacobi
rotation with contiguous row reads. The transform is invalid for bit-identity, and the reason
is worth recording because it is not visible from the code.

THE IDEA: the rotation reads m[k][p] and m[k][q] down two COLUMNS at stride n (PS6011, the
last unfixed strided walk in this function, and SymEig is the bottleneck of the whole
GaLore/SOAP/Shampoo family). m is documented as symmetric, so those values also sit in rows p
and q, which are contiguous. Snapshot the two rows, read from the snapshots, keep the strided
writes — halving the strided traffic in the loop.

WHY IT FAILS: m is NOT exactly symmetric during the sweep, even though the input is. For every
entry outside rows and columns p and q, symmetry survives exactly, because m[p][k] and m[k][p]
receive identical expression trees on identical operands — that part of the reasoning holds.
The four entries where p and q intersect do not: m[p][q] and m[q][p] emerge from the two
rotation loops through DIFFERENT orders of the same operations (one is c*(sA+cB) - s*(sC+cD),
the other s*(cA-sB) + c*(cC-sD)), so they can differ in the last bit.

The first attempt corrected exactly those two entries from the column and still failed the
oracle. That is the real lesson: the asymmetry does not stay local. Every rotation leaves a
possible one-ulp asymmetry at ITS (p,q) pair, and a cyclic sweep visits every pair, so by the
second sweep the asymmetry is scattered across the matrix. A later rotation reading row p at
index k lands on some earlier pair's damaged entry. Correcting the current pair is necessary
and nowhere near sufficient.

CONSEQUENCE: the column walk in the m rotation stays. It cannot be removed by exploiting
symmetry while the bit-identity oracle stands, and that oracle is the contract — it replays
the original row-of-slices implementation and is what makes every other change to this
function safe. The eigenvector accumulator's walk WAS removable (it is only ever
column-rotated, so transposing its storage is a pure relayout) and shipped 1.89x at n=128.

IF MORE SPEED IS WANTED HERE, attack the algorithm rather than the layout: block the Jacobi
sweep, or use a different eigensolver for the sizes that matter. Both change the arithmetic
and would need the oracle relaxed deliberately, with an ADR, rather than by accident.
