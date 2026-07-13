//go:build cgo

package llamagpu_test

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// synthGPT2HF builds a REAL-SIZE (124M-parameter) HuggingFace-shaped GPT-2 tensor
// map with deterministic random weights — the exact names and layouts GPT2FromHF
// consumes (fused c_attn [d,3d]+bias, Conv1D layout, wte/wpe/ln_f). Random weights:
// this validates the CONVERTER MECHANICS and the at-scale pipeline, not language
// quality (real weights need a download the loop is not permitted to make).
func synthGPT2HF(vocab, ctx, d, layers int) map[string]*tensor.Tensor {
	rng := rand.New(rand.NewPCG(7, 0x69f2))
	mk := func(shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape(shape))
		s := t.Storage().F32()
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.02
		}
		return t
	}
	ones := func(n int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape{n})
		s := t.Storage().F32()
		for i := range s {
			s[i] = 1
		}
		return t
	}
	zeros := func(n int) *tensor.Tensor { return tensor.New(tensor.F32, tensor.Shape{n}) }
	ts := map[string]*tensor.Tensor{
		"wte.weight": mk(vocab, d), "wpe.weight": mk(ctx, d),
		"ln_f.weight": ones(d), "ln_f.bias": zeros(d),
	}
	for l := range layers {
		p := fmt.Sprintf("h.%d.", l)
		ts[p+"ln_1.weight"], ts[p+"ln_1.bias"] = ones(d), zeros(d)
		ts[p+"attn.c_attn.weight"], ts[p+"attn.c_attn.bias"] = mk(d, 3*d), zeros(3*d)
		ts[p+"attn.c_proj.weight"], ts[p+"attn.c_proj.bias"] = mk(d, d), zeros(d)
		ts[p+"ln_2.weight"], ts[p+"ln_2.bias"] = ones(d), zeros(d)
		ts[p+"mlp.c_fc.weight"], ts[p+"mlp.c_fc.bias"] = mk(d, 4*d), zeros(4*d)
		ts[p+"mlp.c_proj.weight"], ts[p+"mlp.c_proj.bias"] = mk(4*d, d), zeros(d)
	}
	return ts
}
