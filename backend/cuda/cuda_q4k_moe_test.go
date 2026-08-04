//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAQ4KMoeGateParity — Q4_K twin of TestCUDAQ8MoeGateParity: all-nonzero gate == ungated
// bit-for-bit, zero-gate rows == 0, selected rows == ungated. K%256==0 (Q4_K requirement).
func TestCUDAQ4KMoeGateParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const K, N, M = 512, 128, 96
	rng := rand.New(rand.NewSource(3141))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.4
	}
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	qi, err := quantQ4K(w)
	must(t, err)
	r := qi.(*cuda.ResidentBQ4K)
	defer r.Free()
	da, err := cuda.NewDeviceF32(M, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	dense, err := cuda.NewDeviceF32(M, N)
	must(t, err)
	defer dense.Free()
	must(t, r.QMatMulInto(da, dense))
	ref := make([]float32, M*N)
	must(t, dense.DownloadF32(ref))
	gate, err := cuda.NewDeviceF32(M, 1)
	must(t, err)
	defer gate.Free()
	// all-nonzero → bit-identical
	on := make([]float32, M)
	for i := range on {
		on[i] = 0.3
	}
	must(t, gate.UploadF32(on))
	g1, err := cuda.NewDeviceF32(M, N)
	must(t, err)
	defer g1.Free()
	must(t, r.QMatMulMoeInto(da, g1, gate, M, 1, 0))
	got1 := make([]float32, M*N)
	must(t, g1.DownloadF32(got1))
	for i := range ref {
		if got1[i] != ref[i] {
			t.Fatalf("Q4_K all-nonzero gate not bit-identical at %d: %v vs %v", i, got1[i], ref[i])
		}
	}
	// per-row zero gate → gated rows 0, others == ungated
	mask := make([]float32, M)
	for i := range mask {
		if i%2 == 0 {
			mask[i] = 0
		} else {
			mask[i] = 1
		}
	}
	must(t, gate.UploadF32(mask))
	g2, err := cuda.NewDeviceF32(M, N)
	must(t, err)
	defer g2.Free()
	must(t, r.QMatMulMoeInto(da, g2, gate, M, 1, 0))
	got2 := make([]float32, M*N)
	must(t, g2.DownloadF32(got2))
	for mm := 0; mm < M; mm++ {
		for n := 0; n < N; n++ {
			v := got2[mm*N+n]
			if mask[mm] == 0 {
				if v != 0 {
					t.Fatalf("Q4_K gated-out row %d col %d not zero: %v", mm, n, v)
				}
			} else if v != ref[mm*N+n] {
				t.Fatalf("Q4_K selected row %d col %d != ungated: %v vs %v", mm, n, v, ref[mm*N+n])
			}
		}
	}
	t.Logf("Q4_K MoE-gated GEMV bit-identical (M=%d N=%d K=%d)", M, N, K)
}
