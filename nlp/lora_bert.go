package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// ApplyLoRABert attaches rank-r LoRA adapters (Hu et al. 2021) to every attention
// projection (q, k, v, o) of every encoder layer of a [Bert] (or a RoBERTa /
// DistilBERT model, which share the type) and returns the adapters' trainable
// tensors — the PEFT workflow for a loaded HF encoder: freeze the model (do NOT
// pass its Params to the optimizer), train ONLY the returned adapter params, and
// the base weights stay bit-identical. B is zero-initialized, so attaching does
// not change the output until training. Detach by clearing the layers' MHA.LoRA.
func ApplyLoRABert(m *Bert, r int, alpha float64, seed uint64) ([]*tensor.Tensor, error) {
	if r <= 0 {
		return nil, fmt.Errorf("nlp: ApplyLoRABert rank must be > 0, got %d", r)
	}
	var params []*tensor.Tensor
	for li, l := range m.Layers {
		weights := map[string]*tensor.Tensor{"q": l.Attn.Wq, "k": l.Attn.Wk, "v": l.Attn.Wv, "o": l.Attn.Wo}
		l.Attn.LoRA = make(map[string]*nn.LoRALinear, len(weights))
		for i, name := range []string{"q", "k", "v", "o"} {
			la, err := nn.NewLoRA(weights[name], r, alpha, seed+uint64(li*4+i))
			if err != nil {
				return nil, err
			}
			l.Attn.LoRA[name] = la
			params = append(params, la.Params()...)
		}
	}
	return params, nil
}
