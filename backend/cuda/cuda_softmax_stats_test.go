//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestSoftmaxStatsN: the on-device (maxLogit, Zexp) reduction must match a host f64 reference within a
// tight relative tolerance (the device accumulates Zexp in double; only the tree-vs-sequential sum order
// and f32 __expf differ). Exercised at TinyLlama (32k) and Llama-3 (128k) vocab sizes.
func TestSoftmaxStatsN(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	for _, n := range []int{32000, 128256} {
		for _, T := range []float64{0.7, 1.0} {
			xt := bench.RandF32(tensor.Shape{1, n}, uint64(n)+7)
			xs := xt.Storage().F32()
			// host reference: max then Σ exp((x-max)/T) in sequential f64
			maxL := math.Inf(-1)
			for _, v := range xs {
				if float64(v) > maxL {
					maxL = float64(v)
				}
			}
			var zref float64
			for _, v := range xs {
				zref += math.Exp((float64(v) - maxL) / T)
			}
			d, _ := cuda.NewDeviceF32(1, n)
			d.UploadF32(xs)
			defer d.Free()
			gotMax, gotZ, err := d.SoftmaxStatsN(n, T)
			if err != nil {
				t.Fatalf("n=%d T=%v: %v", n, T, err)
			}
			if gotMax != maxL {
				t.Errorf("n=%d T=%v: max got %v want %v", n, T, gotMax, maxL)
			}
			if rel := math.Abs(gotZ-zref) / zref; rel > 1e-4 {
				t.Errorf("n=%d T=%v: Zexp got %v want %v (rel %.2e > 1e-4)", n, T, gotZ, zref, rel)
			}
		}
	}
}
