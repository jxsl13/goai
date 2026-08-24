package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// LossAndGrad evaluates the standard causal language-model objective and
// returns one gradient for every trainable GPT parameter in Params order.
func ExampleGPT_LossAndGrad() {
	cfg := nlp.GPTConfig{Vocab: 4, Ctx: 2, Dim: 2, Heads: 1, Layers: 1, Eps: 1e-5}
	model, err := nlp.FromSafetensors(cfg, exampleGPTParameters(cfg))
	if err != nil {
		panic(err)
	}
	targets := tensor.New(tensor.F32, tensor.Shape{2})
	targets.SetF64(1, 0)
	targets.SetF64(2, 1)
	loss, grads, err := model.LossAndGrad(backend.NewContext(), []int{0, 1}, targets)
	if err != nil {
		panic(err)
	}
	fmt.Println(loss.Numel(), len(grads))
	// Output: 1 17
}

func exampleGPTParameters(cfg nlp.GPTConfig) map[string]*tensor.Tensor {
	zeros := func(shape ...int) *tensor.Tensor {
		return tensor.New(tensor.F32, tensor.Shape(shape))
	}
	return map[string]*tensor.Tensor{
		"tok_emb":            zeros(cfg.Vocab, cfg.Dim),
		"pos_emb":            zeros(cfg.Ctx, cfg.Dim),
		"blocks.0.ln1.gamma": zeros(cfg.Dim),
		"blocks.0.ln1.beta":  zeros(cfg.Dim),
		"blocks.0.attn.wq":   zeros(cfg.Dim, cfg.Dim),
		"blocks.0.attn.wk":   zeros(cfg.Dim, cfg.Dim),
		"blocks.0.attn.wv":   zeros(cfg.Dim, cfg.Dim),
		"blocks.0.attn.wo":   zeros(cfg.Dim, cfg.Dim),
		"blocks.0.ln2.gamma": zeros(cfg.Dim),
		"blocks.0.ln2.beta":  zeros(cfg.Dim),
		"blocks.0.ffn.w1":    zeros(cfg.Dim, 4*cfg.Dim),
		"blocks.0.ffn.b1":    zeros(4 * cfg.Dim),
		"blocks.0.ffn.w2":    zeros(4*cfg.Dim, cfg.Dim),
		"blocks.0.ffn.b2":    zeros(cfg.Dim),
		"lnf.gamma":          zeros(cfg.Dim),
		"lnf.beta":           zeros(cfg.Dim),
		"head":               zeros(cfg.Dim, cfg.Vocab),
	}
}
