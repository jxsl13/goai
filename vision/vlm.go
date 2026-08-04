package vision

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// VLMProjector is the LLaVA-style bridge between a vision encoder and a
// language model (Liu et al. 2023, "Visual Instruction Tuning"; the 2-layer
// GELU MLP is LLaVA-1.5's projector, §T592/§R236): it maps ViT patch features
// into the LLM's embedding space so image patches become "soft tokens" the
// decoder reads exactly like word embeddings.
//
// In plain terms: the vision model and the language model speak different
// internal languages; this small translator network converts "what the image
// encoder saw" into vectors the text model can read as if they were words —
// that is the whole trick behind chat models that can look at pictures.
//
// Further reading: Liu et al. 2023 (LLaVA) and LLaVA-1.5 (the MLP projector
// upgrade); Zhai et al. 2023 (SigLIP) for the encoder side of modern VLMs.
type VLMProjector struct {
	FC1, FC2 *nn.Linear // vitDim → gptDim GELU MLP
}

// NewVLMProjector builds the 2-layer projector from the ViT feature width to
// the language model's embedding width. Deterministic via seed.
func NewVLMProjector(dtype tensor.Dtype, vitDim, gptDim int, seed uint64) (*VLMProjector, error) {
	if vitDim <= 0 || gptDim <= 0 {
		return nil, fmt.Errorf("vision: NewVLMProjector needs positive dims")
	}
	return &VLMProjector{
		FC1: nn.NewLinear(dtype, vitDim, gptDim, seed),
		FC2: nn.NewLinear(dtype, gptDim, gptDim, seed+1),
	}, nil
}

// Params returns the projector's trainable tensors.
func (p *VLMProjector) Params() []*tensor.Tensor {
	return append(p.FC1.Params(), p.FC2.Params()...)
}

// Forward projects encoder features [T, vitDim] to soft tokens [T, gptDim].
func (p *VLMProjector) Forward(ctx *backend.Context, feats *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := p.FC1.Forward(ctx, feats)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 resource-only alloc; single GELU dispatch, kernel-dominated, time flat
	g, err := backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{h}, nil)
	if err != nil {
		return nil, err
	}
	return p.FC2.Forward(ctx, g[0])
}

// VLMLogits runs the full LLaVA-style forward (§T592): the image's ViT PATCH
// features (class row dropped) are projected into the decoder's embedding
// space, the text tokens are embedded with the decoder's token table, both are
// concatenated — image first, text after — position embeddings run over the
// COMBINED sequence, and the decoder returns logits [imgTokens+len(text),
// vocab]. Row imgTokens+i predicts the token after text[i], exactly like plain
// GPT rows do; captions are trained/read from those rows.
func VLMLogits(ctx *backend.Context, vit *ViT, proj *VLMProjector, gpt *nlp.GPT, img *tensor.Tensor, text []int) (*tensor.Tensor, error) {
	if vit == nil || proj == nil || gpt == nil {
		return nil, fmt.Errorf("vision: VLMLogits needs vit, proj and gpt")
	}
	if len(text) == 0 {
		return nil, fmt.Errorf("vision: VLMLogits needs at least one text token")
	}
	feats, err := vit.Features(ctx, img) // [N+1, vitD]
	if err != nil {
		return nil, err
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	patches, err := ex(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 1, End: feats.Shape()[0]}, feats)
	if err != nil {
		return nil, err
	}
	soft, err := proj.Forward(ctx, patches) // [N, gptD]
	if err != nil {
		return nil, err
	}
	n := soft.Shape()[0]
	total := n + len(text)
	if total > gpt.Config.Ctx {
		return nil, fmt.Errorf("vision: %d image + %d text tokens exceed the decoder context %d", n, len(text), gpt.Config.Ctx)
	}
	tokIdx := tensor.New(gpt.TokEmb.Dtype(), tensor.Shape{len(text)})
	for i, tk := range text {
		if tk < 0 || tk >= gpt.Config.Vocab {
			return nil, fmt.Errorf("vision: text token %d outside vocab %d", tk, gpt.Config.Vocab)
		}
		tokIdx.SetF64(float64(tk), i)
	}
	txtEmb, err := ex(backend.OpEmbed, nil, gpt.TokEmb, tokIdx)
	if err != nil {
		return nil, err
	}
	seq, err := ex(backend.OpConcat, backend.ConcatAttrs{Axis: 0}, soft, txtEmb)
	if err != nil {
		return nil, err
	}
	posIdx := tensor.New(gpt.PosEmb.Dtype(), tensor.Shape{total})
	for i := range total {
		posIdx.SetF64(float64(i), i)
	}
	pos, err := ex(backend.OpEmbed, nil, gpt.PosEmb, posIdx)
	if err != nil {
		return nil, err
	}
	x, err := ex(backend.OpAdd, nil, seq, pos)
	if err != nil {
		return nil, err
	}
	return gpt.ForwardFromEmbed(ctx, x)
}
