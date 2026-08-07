//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
)

// The f16-cuBLAS prefill path uses f16 weights (higher precision than the int8 Q8 weight), so its
// prefill logits should be CLOSE to the int8-MMQ path (both approximate the f32 reference). A wild
// divergence would signal a layout/kernel bug in the routing.
func TestQ8PrefillF16Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no cuda")
	}
	prompt := make([]int, 64)
	for i := range prompt {
		prompt[i] = (i*7)%31991 + 1
	}
	get := func(f16 bool) []float32 {
		cuda.Q8PrefillF16 = f16
		defer func() { cuda.Q8PrefillF16 = false }()
		dec, err := llamagpu.NewLlamaQ8CUDA(tinyQ8(t))
		if err != nil {
			t.Fatal(err)
		}
		defer dec.Release()
		lg, err := dec.StepNLast(prompt, 0)
		if err != nil {
			t.Fatal(err)
		}
		return lg
	}
	mmq := get(false)
	f16 := get(true)
	if len(mmq) != len(f16) {
		t.Fatalf("len %d != %d", len(mmq), len(f16))
	}
	var maxAbs, sumSq, sumSqRef float64
	amx, amxRef := math.Inf(-1), math.Inf(-1)
	var im, if16 int
	for i := range mmq {
		d := math.Abs(float64(mmq[i] - f16[i]))
		if d > maxAbs {
			maxAbs = d
		}
		sumSq += d * d
		sumSqRef += float64(mmq[i]) * float64(mmq[i])
		if float64(mmq[i]) > amx {
			amx = float64(mmq[i])
			im = i
		}
		if float64(f16[i]) > amxRef {
			amxRef = float64(f16[i])
			if16 = i
		}
	}
	relL2 := math.Sqrt(sumSq / sumSqRef)
	t.Logf("prefill logits mmq vs f16: maxAbs=%.4f relL2=%.4f  argmax mmq=%d f16=%d", maxAbs, relL2, im, if16)
	if im != if16 {
		t.Logf("NOTE argmax differs (mmq %d vs f16 %d) — greedy first token would diverge", im, if16)
	}
	if relL2 > 0.05 {
		t.Fatalf("f16 prefill logits diverge from mmq: relL2=%.4f (>0.05 suggests a bug)", relL2)
	}
}
