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
