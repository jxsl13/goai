//go:build vulkan && cgo

package vulkan

import (
	"math"
	"testing"
	"time"
)

// §T397: Vulkan MPS-analog attention forward (mhaMatmul) — per head strided matmul S=scale·Q·Kᵀ →
// causal softmax → strided matmul S·V, all in one Recorder command buffer. Cross-validated against a
// host causal softmax-attention reference, then A/B'd against the flash kernel at 512×8×64.
func TestVulkanMHAMatmulCrossReference(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	for _, tc := range []struct {
		name                string
		seq, heads, kvH, dk int
	}{
		{"mha", 8, 2, 2, 8},
		{"gqa", 12, 4, 2, 8},
		{"mqa", 10, 4, 1, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			seq, heads, kvHeads, dk := tc.seq, tc.heads, tc.kvH, tc.dk
			dm := heads * dk
			rep := heads / kvHeads
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
			got, err := mhaMatmul(q, k, v, seq, dm, heads, dk, 1 /*causal*/, kvHeads, scale)
			if err != nil {
				t.Fatal(err)
			}
			dkv := kvHeads * dk
			for h := range heads {
				qOff := h * dk
				kvOff := (h / rep) * dk
				for i := range seq {
					sc := make([]float64, i+1)
					mx := math.Inf(-1)
					for j := 0; j <= i; j++ {
						var s float64
						for d := range dk {
							s += float64(q[i*dm+qOff+d]) * float64(k[j*dkv+kvOff+d])
						}
						s *= float64(scale)
						sc[j] = s
						if s > mx {
							mx = s
						}
					}
					var l float64
					for j := 0; j <= i; j++ {
						sc[j] = math.Exp(sc[j] - mx)
						l += sc[j]
					}
					for d := range dk {
						var acc float64
						for j := 0; j <= i; j++ {
							acc += sc[j] * float64(v[j*dkv+kvOff+d])
						}
						want := acc / l
						g := float64(got[i*dm+qOff+d])
						if math.Abs(g-want) > 1e-4*math.Max(1, math.Abs(want)) {
							t.Fatalf("h=%d i=%d d=%d: %v vs want %v", h, i, d, g, want)
						}
					}
				}
			}
		})
	}
}

func TestVulkanMHAMatmulvsFlashBench(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
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
		if _, err := fn(); err != nil {
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
	mm := timeit(func() ([]float32, error) { return mhaMatmul(q, k, v, seq, dm, heads, dk, 1, kvHeads, scale) })
	t.Logf("vulkan attention fwd %d×%d×%d causal: flash-kernel %.2f ms | strided-matmul→softmax→matmul %.2f ms | speedup %.1fx",
		seq, heads, dk, flash, mm, flash/mm)
}
