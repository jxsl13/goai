package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// PromptTuning is a soft-prompt adapter (Lester et al. 2021, arXiv:2104.08691, §R127) — the simplest
// parameter-efficient fine-tuning method. It prepends a learned "soft prompt" P ∈ [k, d_model] of k
// continuous virtual-token embeddings to the sequence of input token embeddings X ∈ [seq, d_model]:
//
//	Forward(X) = [P ; X] ∈ [k + seq, d_model]
//
// Only P trains (k·d_model parameters) — every pretrained weight, including the token embedding
// table, stays frozen. The same input-agnostic P is prepended for all inputs at both training and
// inference. Prompt tuning is the input-only special case of prefix tuning (Li & Liang 2021), which
// instead injects learned key/value vectors at every transformer layer. Feed the block's output to
// the frozen model in place of the raw embeddings; slice off the first k output positions if needed.
type PromptTuning struct {
	P      *tensor.Tensor // [numVirtualTokens, dModel], the trainable soft prompt
	dModel int
}

// NewPromptTuning builds a soft prompt of numVirtualTokens (the prompt length k; Lester et al. sweep
// k∈{1,5,20,100,150}, 20 a strong default) over d_model, randomly initialized. Random init is the
// HuggingFace-PEFT default; initializing from sampled vocabulary embeddings converges better when a
// token-embedding table is available (not assumed by this standalone layer).
func NewPromptTuning(dtype tensor.Dtype, numVirtualTokens, dModel int, seed uint64) (*PromptTuning, error) {
	if numVirtualTokens <= 0 || dModel <= 0 {
		return nil, fmt.Errorf("nn: PromptTuning needs positive numVirtualTokens and dModel, got %d, %d", numVirtualTokens, dModel)
	}
	p := tensor.New(dtype, tensor.Shape{numVirtualTokens, dModel})
	fillUniform(p, -0.5, 0.5, seed) // small random init
	return &PromptTuning{P: p, dModel: dModel}, nil
}

// Forward prepends the soft prompt to the input embeddings: X[seq, d_model] → [k+seq, d_model].
func (p *PromptTuning) Forward(ctx *backend.Context, embeds *tensor.Tensor) (*tensor.Tensor, error) {
	if embeds.Ndim() != 2 || embeds.Shape()[1] != p.dModel {
		return nil, fmt.Errorf("nn: PromptTuning expects embeds [seq,%d], got %v", p.dModel, embeds.Shape())
	}
	out, err := backend.Execute(ctx, backend.OpConcat, []*tensor.Tensor{p.P, embeds}, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// NumVirtualTokens returns the soft-prompt length k.
func (p *PromptTuning) NumVirtualTokens() int { return p.P.Shape()[0] }

// Params returns the single trainable soft prompt (the base model is frozen).
func (p *PromptTuning) Params() []*tensor.Tensor { return []*tensor.Tensor{p.P} }
