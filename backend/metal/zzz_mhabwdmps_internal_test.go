//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
	"time"
)

// §T399: attention backward as MPS matmuls (mtl_mha_backward_mps) — the backward analog of §T394's
// forward. §T399 measured the hand flash-backward at 70ms (54× the 1.3ms forward), the GPT training
// bottleneck. Cross-validated against the ref-validated flash-backward, then A/B'd at 512×8×64.
func TestMHABackwardMPSCrossReference(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	for _, tc := range []struct {
		name                string
		seq, heads, kvH, dk int
	}{
		{"mha-small", 8, 2, 2, 8},
		{"mha-mid", 32, 4, 4, 16},
		{"gqa", 16, 8, 2, 8},
		{"mqa", 12, 4, 1, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			seq, heads, kvHeads, dk := tc.seq, tc.heads, tc.kvH, tc.dk
			dm := heads * dk
			kvLen := seq * kvHeads * dk
			scale := float32(1.0 / math.Sqrt(float64(dk)))
			q := make([]float32, seq*dm)
			k := make([]float32, kvLen)
			v := make([]float32, kvLen)
			dO := make([]float32, seq*dm)
			for i := range q {
				q[i] = float32((i%17)-8) * 0.05
				dO[i] = float32((i%7)-3) * 0.04
			}
			for i := range k {
				k[i] = float32((i%13)-6) * 0.06
				v[i] = float32((i%11)-5) * 0.07
			}
			dq, dk_, dv, err := mhaBackwardMPS(q, k, v, dO, seq, dm, heads, dk, 1 /*causal*/, kvHeads, scale)
			if err != nil {
				t.Fatal(err)
			}
			wq, wk, wv, err := mhaBackwardFlashHost(q, k, v, dO, seq, dm, heads, dk, 1, kvHeads, scale)
			if err != nil {
				t.Fatal(err)
			}
			cmp := func(name string, got, want []float32) {
				for i := range want {
					if math.Abs(float64(got[i]-want[i])) > 1e-3*math.Max(1, math.Abs(float64(want[i]))) {
						t.Fatalf("%s[%d]: mps %v vs flash %v", name, i, got[i], want[i])
					}
				}
			}
			cmp("dQ", dq, wq)
			cmp("dK", dk_, wk)
			cmp("dV", dv, wv)
		})
	}
}

func TestMHABackwardMPSvsFlashBench(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const seq, heads, dk = 512, 8, 64
	const dm = heads * dk
	scale := float32(1.0 / math.Sqrt(float64(dk)))
	q := make([]float32, seq*dm)
	k := make([]float32, seq*dm)
	v := make([]float32, seq*dm)
	dO := make([]float32, seq*dm)
	for i := range q {
		q[i] = float32((i%17)-8) * 0.01
		k[i] = float32((i%13)-6) * 0.01
		v[i] = float32((i%11)-5) * 0.01
		dO[i] = float32((i%7)-3) * 0.01
	}
	timeit := func(fn func() error) float64 {
		if err := fn(); err != nil {
			t.Fatal(err)
		}
		const N = 15
		s := time.Now()
		for range N {
			if err := fn(); err != nil {
				t.Fatal(err)
			}
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	flash := timeit(func() error {
		_, _, _, e := mhaBackwardFlashHost(q, k, v, dO, seq, dm, heads, dk, 1, heads, scale)
		return e
	})
	mps := timeit(func() error { _, _, _, e := mhaBackwardMPS(q, k, v, dO, seq, dm, heads, dk, 1, heads, scale); return e })
	t.Logf("attention BACKWARD %d×%d×%d causal: flash-backward %.2f ms | MPS matmuls %.2f ms | speedup %.1fx",
		seq, heads, dk, flash, mps, flash/mps)
}
