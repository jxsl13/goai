---
schema: v1
prefix: NUM
---

## NUM-DIFFERENTIAL-GATE-001
IF a bit-identity gate compares two implementations against each other, THEN the gate SHALL be documented as covering only what DIFFERS between its arms; a subroutine both arms call stays unguarded, as applyReflector was.

## TEST-GOLDEN-BELOW-THRESHOLD-001
IF a bit-identity golden runs at a geometry below the parallel threshold of the code it covers, THEN the reader SHALL treat the parallel arm as UNCOVERED and add a two-arm diff selected by the threshold.

Rationale: Every bit-identity golden in linalg ran far under factorParThreshold — QR at 24x16, 384 elements against a 65536-element bound — so no test had ever entered a parallel branch, while the names and comments read as though both paths were pinned. Demonstrated rather than argued: perturbing only the parallel branch of the reflector apply by one ulp left TestQRBitIdenticalToGolden PASSING. The fix is a differential whose two arms are THE SAME SOURCE selected only by the threshold (made a var for this), so any difference is attributable to the partition and not to a reference written in a different expression shape — a separately written reference contracts to FMA differently and fails by an ulp on arithmetic that never changed.

## PERF-LOCALITY-SUBSUMES-PARALLELISM-001
IF a loop is both cache-strided and serial and two branches fix one deficiency each, THEN the implementing agent SHALL land the locality fix first and re-measure the parallel gain, expecting under 1.05x.

Rationale: A conflict in qr.go paired a column-parallel reflector apply on the strided rm[i][j] layout against a serial i-outer rank-1 update on a flat row-major layout. After taking the contiguous rewrite, re-applying parallelism to the apply half yielded only 1.03x (LstsqMat/512 -2.80%, /768 -2.25%), against an Amdahl estimate of about 1.85x over the parallelized fraction. The contiguous form is bandwidth-bound and one core already saturates much of the bandwidth, so the extra cores buy almost nothing. The large win the strided layout appeared to give parallelism was a cache-locality deficit that parallelism partly masked; the locality fix collects it directly. Two apparently competing alternatives were largely the same win counted twice, which is why measuring the second one after the first is what reveals the real remaining headroom.

## PROC-AUDIT-EACH-FINDING-001
WHEN a check reports a class of findings and the whole class looks actionable, the implementing agent SHALL probe each site before writing code, since 8 findings split into 2 already-covered, 1 unfixable and 2 real.

Rationale: PS6004 listed eight unverified dual paths in linalg and acting on the list as a list would have produced two redundant tests, one impossible one, and a wrong belief about what was covered. Probing each site by mutation split them: NormFro and NormInf were already gated by TestNormFlatArmsBitIdentical, so a one-ulp perturbation of their fast paths reddened an existing test and any new test would be duplicate maintenance. Pinv guards its fast path on SVD OWN outputs, always contiguous f64, so its accessor arm is unreachable — panicking in it left the whole suite green — and no test can gate it, so it takes an ignore directive naming the reason. Only toColMajor and toFlat were genuinely uncovered, and each new gate is the only test that reddens on a perturbation of its arm. The probe is cheap and specific: break the arm the finding names, then read WHICH tests fail. That answers already-covered, uncovered, and unreachable in one step, which reading the finding cannot.

## PERF-FUSED-PASS-BEATS-CACHING-001
IF a value looks redundantly recomputed inside a loop that already loads its operand, THEN the implementing agent SHALL leave it: hoisting one of 3 fused accumulators cost 54% on SVDPCA and 19% on Pinv.

Rationale: The one-sided Jacobi sweep in linalg SVD computes alpha, beta and gamma in a single pass over columns i and j. alpha is column i squared norm and is invariant across the whole j loop, so it reads as recomputed n-i-1 times for nothing — 69% of the function by line-level profile at GOMAXPROCS=1, and SVDPCA scales 1.01x so it is entirely serial. Hoisting alpha out of the j loop and refreshing it only after a rotation is exactly bit-identical, the same ascending-k sum over the same values, and it measured 54.34% SLOWER on SVDPCA and 19.44% slower on Pinv_64x32. The framing was wrong: alpha is not redundant work, it is FREE work riding a traversal that already loads ci[k], and the three independent accumulators give the loop its instruction-level parallelism. Splitting them trades one pass with 3 FMAs for one pass with 2 plus an extra full pass per rotation, and the rotation rate here is 71.7% of pairs, so nearly every pair pays it. Two consequences. A load already in flight makes an extra arithmetic op on it close to free, so counting arithmetic without counting traversals misranks these rewrites. And bit-identical parallelism is NOT available for this sweep at all: each rotation changes columns later pairs read, and the classic parallel Jacobi round-robin reorders the rotations, producing a valid but different factorization.
