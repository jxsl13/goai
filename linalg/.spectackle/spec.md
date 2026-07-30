---
schema: v1
prefix: NUM
---

## NUM-DIFFERENTIAL-GATE-001
IF a bit-identity gate compares two implementations against each other, THEN the gate SHALL be documented as covering only what DIFFERS between its arms; a subroutine both arms call stays unguarded, as applyReflector was.

## TEST-GOLDEN-BELOW-THRESHOLD-001
IF a bit-identity golden runs at a geometry below the parallel threshold of the code it covers, THEN the reader SHALL treat the parallel arm as UNCOVERED and add a two-arm diff selected by the threshold.

Rationale: Every bit-identity golden in linalg ran far under factorParThreshold — QR at 24x16, 384 elements against a 65536-element bound — so no test had ever entered a parallel branch, while the names and comments read as though both paths were pinned. Demonstrated rather than argued: perturbing only the parallel branch of the reflector apply by one ulp left TestQRBitIdenticalToGolden PASSING. The fix is a differential whose two arms are THE SAME SOURCE selected only by the threshold (made a var for this), so any difference is attributable to the partition and not to a reference written in a different expression shape — a separately written reference contracts to FMA differently and fails by an ulp on arithmetic that never changed.
