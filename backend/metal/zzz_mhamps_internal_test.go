//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
	"time"
)

// §T394: MPS-matmul attention forward (mtl_mha_mps) — the reformulation §T393's floor measurement
// showed is ~15× faster than the hand flash kernel. Cross-validated against a host causal
// softmax-attention reference, then A/B'd against the flash kernel at the real 512×8×64 shape.
func TestMHAMPSCrossReference(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const seq, heads, dk = 8, 2, 8
	const dm = heads * dk
	const kvHeads = heads
	scale := float32(1.0 / math.Sqrt(float64(dk)))
	q := make([]float32, seq*dm)
	k := make([]float32, seq*kvHeads*dk)
	v := make([]float32, seq*kvHeads*dk)
	for i := range q {
		q[i] = float32((i%17)-8) * 0.05
	}
	for i := range k {
		k[i] = float32((i%13)-6) * 0.06
		v[i] = float32((i%11)-5) * 0.07
	}
	got, err := mhaMPS(q, k, v, seq, dm, heads, dk, 1 /*causal*/, kvHeads, scale)
	if err != nil {
		t.Fatal(err)
	}
	// host reference: per head, per query row i, causal softmax over j<=i.
	dkv := kvHeads * dk
	for h := range heads {
		off := h * dk
		for i := range seq {
			scores := make([]float64, i+1)
			mx := math.Inf(-1)
			for j := 0; j <= i; j++ {
				var s float64
				for d := range dk {
					s += float64(q[i*dm+off+d]) * float64(k[j*dkv+off+d])
				}
				s *= float64(scale)
				scores[j] = s
				if s > mx {
					mx = s
				}
			}
			var l float64
			for j := 0; j <= i; j++ {
				scores[j] = math.Exp(scores[j] - mx)
				l += scores[j]
			}
			for d := range dk {
				var acc float64
				for j := 0; j <= i; j++ {
					acc += scores[j] * float64(v[j*dkv+off+d])
				}
				want := acc / l
				g := float64(got[i*dm+off+d])
				if math.Abs(g-want) > 1e-4*math.Max(1, math.Abs(want)) {
					t.Fatalf("h=%d i=%d d=%d: %v vs want %v", h, i, d, g, want)
				}
			}
		}
	}
}

func TestMHAMPSvsFlashBench(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const seq, heads, dk = 512, 8, 64
	const dm = heads * dk
	const kvHeads = heads
	scale := float32(1.0 / math.Sqrt(float64(dk)))
	q := make([]float32, seq*dm)
	k := make([]float32, seq*kvHeads*dk)
	v := make([]float32, seq*kvHeads*dk)
	for i := range q {
		q[i] = float32((i%17)-8) * 0.01
	}
	for i := range k {
		k[i] = float32((i%13)-6) * 0.01
		v[i] = float32((i%11)-5) * 0.01
	}
	timeit := func(fn func() ([]float32, error)) float64 {
		if _, err := fn(); err != nil { // warmup
			t.Fatal(err)
		}
		const N = 20
		s := time.Now()
		for range N {
			if _, err := fn(); err != nil {
				t.Fatal(err)
			}
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	flash := timeit(func() ([]float32, error) { return flashAttnHost(q, k, v, seq, dm, heads, dk, 1, kvHeads, scale) })
	mps := timeit(func() ([]float32, error) { return mhaMPS(q, k, v, seq, dm, heads, dk, 1, kvHeads, scale) })
	t.Logf("attention fwd %d×%d×%d causal: flash-kernel %.2f ms | MPS(Q·Kᵀ)→softmax→MPS(P·V) %.2f ms | speedup %.1fx (torch-mps ~0.45ms)",
		seq, heads, dk, flash, mps, flash/mps)
}
