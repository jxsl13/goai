//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAQ8MoeGateParity validates the MoE-gated Q8 GEMV (QMatMulMoeInto): (1) with an all-nonzero
// gate it is BIT-IDENTICAL to the ungated QMatMulInto (the gate only adds an early-return, never alters
// the arithmetic), and (2) with a per-row zero gate the gated-out rows are exactly 0 while the rest match
// the ungated result — which is what makes the sparse MoE eval bit-identical to the dense eval (whose
// non-selected experts are zeroed by RowAxpy weight=0), at ~E/K the decode cost.
func TestCUDAQ8MoeGateParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const K, N, M = 256, 128, 96
	rng := rand.New(rand.NewSource(9182))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.5
	}
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	r, err := cuda.NewResidentBQ8(w)
	must(t, err)
	defer r.Free()
	da, err := cuda.NewDeviceF32(M, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))

	// ungated reference
	dense, err := cuda.NewDeviceF32(M, N)
	must(t, err)
	defer dense.Free()
	must(t, r.QMatMulInto(da, dense))
	ref := make([]float32, M*N)
	must(t, dense.DownloadF32(ref))

	// gate buffer [M,1]
	gate, err := cuda.NewDeviceF32(M, 1)
	must(t, err)
	defer gate.Free()

	// (1) all-nonzero gate → BIT-IDENTICAL to ungated
	allOn := make([]float32, M)
	for i := range allOn {
		allOn[i] = 0.3 // any nonzero
	}
	must(t, gate.UploadF32(allOn))
	g1, err := cuda.NewDeviceF32(M, N)
	must(t, err)
	defer g1.Free()
	must(t, r.QMatMulMoeInto(da, g1, gate, 1, 0))
	got1 := make([]float32, M*N)
	must(t, g1.DownloadF32(got1))
	for i := range ref {
		if got1[i] != ref[i] {
			t.Fatalf("all-nonzero gate not bit-identical to ungated at %d: %v vs %v", i, got1[i], ref[i])
		}
	}

	// (2) per-row zero gate → gated rows exactly 0, others == ungated
	mask := make([]float32, M)
	for i := range mask {
		if i%2 == 0 {
			mask[i] = 0 // gated out
		} else {
			mask[i] = 1
		}
	}
	must(t, gate.UploadF32(mask))
	g2, err := cuda.NewDeviceF32(M, N)
	must(t, err)
	defer g2.Free()
	must(t, r.QMatMulMoeInto(da, g2, gate, 1, 0))
	got2 := make([]float32, M*N)
	must(t, g2.DownloadF32(got2))
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			v := got2[m*N+n]
			if mask[m] == 0 {
				if v != 0 {
					t.Fatalf("gated-out row %d col %d not zero: %v", m, n, v)
				}
			} else if v != ref[m*N+n] {
				t.Fatalf("selected row %d col %d != ungated: %v vs %v", m, n, v, ref[m*N+n])
			}
		}
	}
	t.Logf("Q8 MoE-gated GEMV bit-identical: all-on == ungated, zero-gate rows == 0, selected rows == ungated (M=%d N=%d)", M, N)
}
