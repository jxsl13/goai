package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// ApplyLoRAGPT attaches rank-r LoRA adapters (Hu et al. 2021) to every attention
// projection (q, k, v, o) of every block (§T504) and returns the adapters'
// trainable tensors — the PEFT workflow for the built-in GPT: freeze the model
// (do NOT pass its Params to the optimizer), train ONLY the returned adapter
// params, and the base weights stay bit-identical. B is zero-initialized, so
// attaching adapters does not change the model's output until training. The
// adapters live on the blocks' MHA.LoRA maps; detach by clearing those maps.
func ApplyLoRAGPT(g *GPT, r int, alpha float64, seed uint64) ([]*tensor.Tensor, error) {
	if r <= 0 {
		return nil, fmt.Errorf("nlp: ApplyLoRAGPT rank must be > 0, got %d", r)
	}
	var params []*tensor.Tensor
	for bi, b := range g.Blocks {
		weights := map[string]*tensor.Tensor{"q": b.Attn.Wq, "k": b.Attn.Wk, "v": b.Attn.Wv, "o": b.Attn.Wo}
		b.Attn.LoRA = make(map[string]*nn.LoRALinear, len(weights))
		for i, name := range []string{"q", "k", "v", "o"} {
			l, err := nn.NewLoRA(weights[name], r, alpha, seed+uint64(bi*4+i))
			if err != nil {
				return nil, err
			}
			b.Attn.LoRA[name] = l
			params = append(params, l.Params()...)
		}
	}
	return params, nil
}
