package linalg

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestQRParallelArmBitIdenticalToSerial is the bit-identity gate for the PARALLEL reflector
// application. It exists because no other test reached that code: every golden in this package
// runs at a geometry far below factorParThreshold — QR's is 24x16, i.e. 384 elements against a
// 65536-element bound — so the parallel branch was never entered and the goldens were pinning
// only the serial path while reading as though they covered both.
//
// The reference is not a hand-written reimplementation. A reference rewritten for readability
// contracts to FMA differently than the code under test and fails by an ulp on arithmetic that
// never changed, so the two arms here are THE SAME SOURCE, selected only by the threshold: raise
// it above the geometry and the identical loop runs serially. Any difference is therefore
// attributable to the partition and to nothing else.
func TestQRParallelArmBitIdenticalToSerial(t *testing.T) {
	const m, n = 300, 256 // (m-k)*(n-k) = 76800 at k=0, above the 65536 default
	a := tensor.New(tensor.F64, tensor.Shape{m, n})
	x := uint64(0x2545F4914F6CDD1D)
	for i := range m {
		for j := range n {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			a.SetF64(float64(int64(x))/(1<<62), i, j)
		}
	}
	if m*n <= factorParThreshold {
		t.Fatalf("test geometry no longer reaches the parallel arm: %d vs threshold %d",
			m*n, factorParThreshold)
	}

	bits := func() []uint64 {
		q, r, err := QR(a)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]uint64, 0, m*n+n*n)
		for i := range m {
			for j := range n {
				out = append(out, math.Float64bits(q.AtF64(i, j)))
			}
		}
		for i := range n {
			for j := range n {
				out = append(out, math.Float64bits(r.AtF64(i, j)))
			}
		}
		return out
	}

	par := bits() // parallel arm (default threshold)

	saved := factorParThreshold
	factorParThreshold = 1 << 30 // force every step serial
	ser := bits()
	factorParThreshold = saved

	if len(par) != len(ser) {
		t.Fatalf("%d values from the parallel arm, %d from the serial arm", len(par), len(ser))
	}
	for i := range ser {
		if par[i] != ser[i] {
			t.Fatalf("value %d: parallel %016x != serial %016x — the row partition moved a bit",
				i, par[i], ser[i])
		}
	}
}
