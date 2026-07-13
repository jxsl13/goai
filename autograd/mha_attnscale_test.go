package autograd_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// refScaledAttn is an independent single-head softmax attention with the YaRN
// attention-temperature factor: scores = (Q·Kᵀ/√d)·attnScale, softmax, ·V.
func refScaledAttn(q, k, v *tensor.Tensor, attnScale float64, causal bool) *tensor.Tensor {
	seq, d := q.Shape()[0], q.Shape()[1]
	scale := attnScale / math.Sqrt(float64(d))
	out := tensor.New(tensor.F64, tensor.Shape{seq, d})
	s := make([]float64, seq)
	for i := range seq {
		jmax := seq
		if causal {
			jmax = i + 1
		}
		m := math.Inf(-1)
		for j := range jmax {
			var dot float64
			for e := range d {
				dot += q.AtF64(i, e) * k.AtF64(j, e)
			}
			s[j] = dot * scale
			if s[j] > m {
				m = s[j]
			}
		}
		var sum float64
		for j := range jmax {
			s[j] = math.Exp(s[j] - m)
			sum += s[j]
		}
		for e := range d {
			var acc float64
			for j := range jmax {
				acc += s[j] / sum * v.AtF64(j, e)
			}
			out.SetF64(acc, i, e)
		}
	}
	return out
}

// §V16 tier-1: OpMHA with an attn_scale attr matches an independent scaled-score
// attention at f64 1e-12, for scaled and (backward-compatible) unscaled cases.
func TestMHAAttnScaleParity(t *testing.T) {
	q, k, v := randFA(tensor.Shape{6, 4}, 1), randFA(tensor.Shape{6, 4}, 2), randFA(tensor.Shape{6, 4}, 3)
	for _, s := range []float64{1.0, 1.1386, 2.0} {
		for _, causal := range []bool{false, true} {
			out, err := backend.Execute(backend.NewContext(), backend.OpMHA,
				[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: 1, Causal: causal, Scale: s})
			if err != nil {
				t.Fatal(err)
			}
			want := refScaledAttn(q, k, v, s, causal)
			for idx := range out[0].Numel() {
				id := tensor.Unravel(idx, out[0].Shape())
				if a, b := out[0].AtF64(id...), want.AtF64(id...); math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(b)) {
					t.Fatalf("attn_scale=%v causal=%v [%d]: %.14g vs %.14g", s, causal, idx, a, b)
				}
			}
		}
	}
}

// §V2: the attention-temperature-scaled MHA is differentiable — finite differences
// match the analytic gradients into Q, K, V.
func TestMHAAttnScaleGradCheck(t *testing.T) {
	attrs := backend.AttnAttrs{Heads: 2, Causal: true, Scale: 1.4}
	ins := []*tensor.Tensor{
		randFA(tensor.Shape{5, 8}, 4), randFA(tensor.Shape{5, 8}, 5), randFA(tensor.Shape{5, 8}, 6),
	}
	gradCheckOp(t, backend.OpMHA, attrs, ins)
}

// YaRN extends a model's context by rescaling RoPE frequencies AND raising the
// attention temperature so the softmax stays appropriately sharp over the longer
// sequence. backend.YaRNAttnScale gives that temperature (0.1·ln(s)+1); pass it as
// the OpMHA "attn_scale" attr. Here a 4× extension gives temperature ≈1.14, and the
// attention still maps a [4,4] batch to [4,4].
func Example_yarnAttentionTemperature() {
	q := randFA(tensor.Shape{4, 4}, 1)
	k := randFA(tensor.Shape{4, 4}, 2)
	v := randFA(tensor.Shape{4, 4}, 3)
	temp := backend.YaRNAttnScale(4)
	out, _ := backend.Execute(backend.NewContext(), backend.OpMHA,
		[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: 1, Causal: true, Scale: temp})
	fmt.Printf("temperature=%.4f out=%v\n", temp, out[0].Shape())
	// Output: temperature=1.1386 out=(4, 4)
}
