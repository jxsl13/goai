//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// refMoEGate is the reference top-k routing: weight[e] = softmax over the top-k logits (renormalized
// combine weights), 0 for unselected — matching nn.TopKGating / the Mixtral combine.
func refMoEGate(logits []float32, e, k int) []float32 {
	w := make([]float32, e)
	sel := make([]bool, e)
	for p := 0; p < k && p < e; p++ {
		best, bv := -1, math.Inf(-1)
		for i := 0; i < e; i++ {
			if !sel[i] && float64(logits[i]) > bv {
				bv, best = float64(logits[i]), i
			}
		}
		if best >= 0 {
			sel[best] = true
		}
	}
	mx := math.Inf(-1)
	for i := 0; i < e; i++ {
		if sel[i] && float64(logits[i]) > mx {
			mx = float64(logits[i])
		}
	}
	sum := 0.0
	for i := 0; i < e; i++ {
		if sel[i] {
			sum += math.Exp(float64(logits[i]) - mx)
		}
	}
	for i := 0; i < e; i++ {
		if sel[i] {
			w[i] = float32(math.Exp(float64(logits[i])-mx) / sum)
		}
	}
	return w
}

// TestCUDAMoEGate checks cu_moe_gate (Recorder.MoEGate) produces the Mixtral top-k renormalized
// combine weights — the routing primitive for a GPU sparse-MoE decoder. It covers several
// (rows, experts, top-k) shapes including top-k == experts (dense) and k=1 (argmax hard routing).
func TestCUDAMoEGate(t *testing.T) {
	skipNoGPU(t)
	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	cases := []struct {
		rows, e, k int
		logits     []float32
	}{
		{1, 8, 2, []float32{0.2, -1.1, 3.4, 0.0, 2.1, -0.7, 1.5, 0.9}},
		{1, 4, 1, []float32{-2, 5, 1, 4}},
		{1, 3, 3, []float32{0.5, 0.5, -0.5}}, // top-k == experts (dense softmax)
		{3, 6, 2, []float32{
			1, 2, 3, 4, 5, 6,
			6, 5, 4, 3, 2, 1,
			0, 0, 9, 0, 0, 3,
		}},
	}
	for ci, c := range cases {
		dlog, err := cuda.NewDeviceBufferF32(c.logits)
		if err != nil {
			t.Fatalf("case %d upload logits: %v", ci, err)
		}
		dw, err := cuda.NewDeviceF32(c.rows, c.e)
		if err != nil {
			t.Fatalf("case %d alloc weights: %v", ci, err)
		}
		if err := rec.MoEGate(dlog, dw, c.rows, c.e, c.k, 0, 1); err != nil {
			t.Fatalf("case %d MoEGate: %v", ci, err)
		}
		if err := rec.Wait(); err != nil {
			t.Fatalf("case %d wait: %v", ci, err)
		}
		got := make([]float32, c.rows*c.e)
		if err := dw.DownloadF32(got); err != nil {
			t.Fatalf("case %d download: %v", ci, err)
		}
		dlog.Free()
		dw.Free()

		for r := 0; r < c.rows; r++ {
			want := refMoEGate(c.logits[r*c.e:(r+1)*c.e], c.e, c.k)
			var wsum float64
			for j := 0; j < c.e; j++ {
				g := got[r*c.e+j]
				if math.IsNaN(float64(g)) || math.Abs(float64(g)-float64(want[j])) > 1e-6 {
					t.Fatalf("case %d row %d expert %d: gpu %v vs ref %v", ci, r, j, g, want[j])
				}
				wsum += float64(g)
			}
			if math.Abs(wsum-1) > 1e-5 { // selected weights renormalize to 1
				t.Fatalf("case %d row %d: weights sum to %v, want 1", ci, r, wsum)
			}
		}
	}
	t.Logf("cu_moe_gate matches the Mixtral top-k renormalized combine weights across %d shapes", len(cases))
}

// TestCUDARowAxpy checks cu_row_axpy (Recorder.RowAxpy): dst[r,:] += arow[r]·src[r,:] — the MoE
// weighted-combine primitive. Covers the decode case (1 row) and a multi-row batch.
func TestCUDARowAxpy(t *testing.T) {
	skipNoGPU(t)
	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	cases := []struct {
		rows, cols int
		dst0, src  []float32
		arow       []float32
	}{
		{1, 4, []float32{1, 2, 3, 4}, []float32{10, 20, 30, 40}, []float32{0.5}},
		{3, 2, []float32{0, 0, 1, 1, 2, 2}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 0, -1}},
	}
	for ci, c := range cases {
		dd, _ := cuda.NewDeviceBufferF32(c.dst0)
		ds, _ := cuda.NewDeviceBufferF32(c.src)
		da, _ := cuda.NewDeviceBufferF32(c.arow)
		if err := rec.RowAxpy(dd, ds, da, c.rows, c.cols); err != nil {
			t.Fatalf("case %d RowAxpy: %v", ci, err)
		}
		if err := rec.Wait(); err != nil {
			t.Fatalf("case %d wait: %v", ci, err)
		}
		got := make([]float32, c.rows*c.cols)
		if err := dd.DownloadF32(got); err != nil {
			t.Fatalf("case %d download: %v", ci, err)
		}
		dd.Free()
		ds.Free()
		da.Free()
		for r := 0; r < c.rows; r++ {
			for j := 0; j < c.cols; j++ {
				want := c.dst0[r*c.cols+j] + c.arow[r]*c.src[r*c.cols+j]
				if g := got[r*c.cols+j]; math.Abs(float64(g-want)) > 1e-6 {
					t.Fatalf("case %d [%d,%d]: gpu %v vs ref %v", ci, r, j, g, want)
				}
			}
		}
	}
	t.Logf("cu_row_axpy matches dst += diag(arow)·src across %d shapes", len(cases))
}
