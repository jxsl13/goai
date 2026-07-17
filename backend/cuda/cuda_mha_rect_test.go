//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// refMHARect computes causal multi-head attention with rectangular head dims (q/k width dqk, v width
// dv, no GQA) on the host, for validating cu MHARect. Q [sq, heads·dqk], K [sk, heads·dqk],
// V [sk, heads·dv] → out [sq, heads·dv]. Query i attends to keys j ≤ i.
func refMHARect(q, k, v []float32, sq, sk, heads, dqk, dv int, scale float64, causal bool) []float32 {
	out := make([]float32, sq*heads*dv)
	for h := 0; h < heads; h++ {
		for i := 0; i < sq; i++ {
			lim := sk - 1 // non-causal: attend to every key
			if causal {
				lim = i + (sk - sq) // query i (abs sk-sq+i) attends to keys ≤ its position
			}
			sc := make([]float64, lim+1)
			mx := math.Inf(-1)
			for j := 0; j <= lim; j++ {
				var d float64
				for c := 0; c < dqk; c++ {
					d += float64(q[i*heads*dqk+h*dqk+c]) * float64(k[j*heads*dqk+h*dqk+c])
				}
				d *= scale
				sc[j] = d
				if d > mx {
					mx = d
				}
			}
			var sum float64
			for j := 0; j <= lim; j++ {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for c := 0; c < dv; c++ {
				var acc float64
				for j := 0; j <= lim; j++ {
					acc += sc[j] / sum * float64(v[j*heads*dv+h*dv+c])
				}
				out[i*heads*dv+h*dv+c] = float32(acc)
			}
		}
	}
	return out
}

// TestCUDAMHARect checks cu MHARect: multi-head attention where the value head width (dv) differs
// from the query/key head width (dqk) — the rectangular geometry DeepSeek-V2 MLA needs. Covers dv<dqk
// and dv>dqk with a couple of head counts.
func TestCUDAMHARect(t *testing.T) {
	skipNoGPU(t)
	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	gen := func(n, seed int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((((i+seed)*7)%23)-11) * 0.03
		}
		return x
	}
	cases := []struct {
		sq, heads, dqk, dv int
		causal             bool
	}{
		{5, 2, 6, 4, true},  // dv < dqk (MLA: VHead < QKNope+QKRope)
		{4, 3, 4, 8, true},  // dv > dqk
		{6, 1, 8, 8, true},  // square (degenerate == plain MHA)
		{5, 2, 6, 4, false}, // non-causal (offset=sk branch): every query sees every key
	}
	for ci, c := range cases {
		sk := c.sq
		q := gen(c.sq*c.heads*c.dqk, 1)
		k := gen(sk*c.heads*c.dqk, 2)
		v := gen(sk*c.heads*c.dv, 3)
		scale := 1.0 / math.Sqrt(float64(c.dqk))
		causalFlag := 0
		if c.causal {
			causalFlag = 1
		}

		dq, _ := cuda.NewDeviceBufferF32(q)
		dk, _ := cuda.NewDeviceBufferF32(k)
		dv, _ := cuda.NewDeviceBufferF32(v)
		do, _ := cuda.NewDeviceF32(c.sq, c.heads*c.dv)
		if err := rec.MHARect(dq, dk, dv, do, c.sq, sk, c.heads, c.heads, c.dqk, c.dv, causalFlag, 0, float32(scale)); err != nil {
			t.Fatalf("case %d MHARect: %v", ci, err)
		}
		if err := rec.Wait(); err != nil {
			t.Fatalf("case %d wait: %v", ci, err)
		}
		got := make([]float32, c.sq*c.heads*c.dv)
		if err := do.DownloadF32(got); err != nil {
			t.Fatalf("case %d download: %v", ci, err)
		}
		dq.Free()
		dk.Free()
		dv.Free()
		do.Free()

		want := refMHARect(q, k, v, c.sq, sk, c.heads, c.dqk, c.dv, scale, c.causal)
		for idx := range want {
			if math.IsNaN(float64(got[idx])) || math.Abs(float64(got[idx]-want[idx])) > 1e-4 {
				t.Fatalf("case %d (sq=%d heads=%d dqk=%d dv=%d) [%d]: gpu %v vs ref %v",
					ci, c.sq, c.heads, c.dqk, c.dv, idx, got[idx], want[idx])
			}
		}
	}
	t.Logf("cu MHARect matches host rectangular multi-head attention across %d shapes (dv≠dqk)", len(cases))
}
