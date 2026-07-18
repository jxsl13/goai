//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// The fused nvcc WMMA attention must match a plain f64 causal-softmax-attention reference within
// the f16-input envelope. This is the correctness gate for the kernel that targets vLLM's
// prefill lead (fused attention, not the GEMM).
func TestCUDAWMMAAttn(t *testing.T) {
	skipNoGPU(t)
	const heads, seq, hd = 2, 64, 64
	rng := rand.New(rand.NewSource(23))
	n := heads * seq * hd
	q := make([]float32, n)
	k := make([]float32, n)
	v := make([]float32, n)
	o := make([]float32, n)
	for i := range q {
		q[i] = float32(rng.NormFloat64()) * 0.25
		k[i] = float32(rng.NormFloat64()) * 0.25
		v[i] = float32(rng.NormFloat64()) * 0.25
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	if err := cuda.WMMAAttn(q, k, v, o, heads, seq, hd, scale); err != nil {
		t.Fatal(err)
	}
	// reference (f64, causal)
	var num, den float64
	for h := 0; h < heads; h++ {
		base := h * seq * hd
		for i := 0; i < seq; i++ {
			// scores row
			sc := make([]float64, i+1)
			mx := -1e300
			for j := 0; j <= i; j++ {
				var d float64
				for c := 0; c < hd; c++ {
					d += float64(q[base+i*hd+c]) * float64(k[base+j*hd+c])
				}
				d *= float64(scale)
				sc[j] = d
				if d > mx {
					mx = d
				}
			}
			var sum float64
			for j := 0; j <= i; j++ {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for c := 0; c < hd; c++ {
				var acc float64
				for j := 0; j <= i; j++ {
					acc += sc[j] / sum * float64(v[base+j*hd+c])
				}
				got := float64(o[base+i*hd+c])
				num += (got - acc) * (got - acc)
				den += acc * acc
			}
		}
	}
	relRMS := math.Sqrt(num / den)
	t.Logf("fused WMMA attention vs f64 causal ref: norm-rel-RMS %.3e", relRMS)
	if relRMS > 1e-2 {
		t.Fatalf("fused WMMA attention off reference: norm-rel-RMS %.3e", relRMS)
	}
}
