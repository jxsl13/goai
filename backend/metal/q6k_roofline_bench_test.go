//go:build darwin && cgo

package metal

import (
	"fmt"
	"sort"
	"testing"
)

// TestQ6KDecodeRoofline measures resident Q6_K decode with the command buffer's GPU timestamps.
// The three production cells cover TinyLlama attention and feed-forward projections; the final
// cache-busting cell streams enough weight data to approximate the model-wide DRAM regime.
func TestQ6KDecodeRoofline(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	if testing.Short() {
		t.Skip("Q6_K roofline allocates a cache-busting resident weight")
	}

	for _, shape := range []struct {
		k, n int
	}{
		{2048, 2048},
		{2048, 5632},
		{5632, 2048},
		{2048, 131072},
	} {
		k, n := shape.k, shape.n
		weightBytes := n * (k / 256) * 210
		w := make([]byte, weightBytes)
		for i := range w {
			w[i] = byte(i*31 + 7)
		}
		raw, err := Backend{}.UploadQuant(w, 14, n, k)
		if err != nil {
			t.Skip(err)
		}
		rq := raw.(*ResidentQWeight)
		xb, err := NewDeviceBufferF32(make([]float32, k))
		if err != nil {
			t.Fatal(err)
		}
		ob, err := NewDeviceBufferF32(make([]float32, n))
		if err != nil {
			t.Fatal(err)
		}

		reps := 128
		if weightBytes > 20<<20 {
			reps = 16
		}
		samples := make([]float64, 0, 9)
		for run := range 13 {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			for range reps {
				if err := r.QMatMulResident(xb, rq, ob, 1); err != nil {
					r.Free()
					t.Fatal(err)
				}
			}
			if err := r.Commit(); err != nil {
				r.Free()
				t.Fatal(err)
			}
			if err := r.Wait(); err != nil {
				r.Free()
				t.Fatal(err)
			}
			perOp := LastGPUSeconds() / float64(reps)
			r.Free()
			if run >= 4 {
				samples = append(samples, perOp)
			}
		}
		sort.Float64s(samples)
		perOp := samples[len(samples)/2]
		gbs := float64(weightBytes) / perOp / 1e9
		fmt.Printf("Q6K roofline K=%d N=%d weight=%.1fMB per-op=%.1fus %.1f GB/s\n",
			k, n, float64(weightBytes)/(1<<20), perOp*1e6, gbs)
		if gbs < 20 {
			t.Errorf("K=%d N=%d: %.1f GB/s — the decode kernel has fallen off the bandwidth roofline entirely", k, n, gbs)
		}

		rq.Close()
		xb.Release()
		ob.Release()
	}
}
