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
