package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// RWKVDecodeState is the O(1) recurrent generation state of an [RWKV] model —
// one fixed-size [nn.RWKVState] per block. This is RWKV's defining advantage
// over attention decoders: where a transformer's KV-cache grows linearly with
// every generated token, this state never grows. Each layer keeps exactly five
// [dim] vectors (the two token-shift rows and the WKV numerator/denominator/
// max-exponent), no matter whether the model has absorbed three tokens or
// three million. There is consequently no position argument anywhere in the
// decode path and no context-length limit: time lives inside the state.
//
// For ML practitioners: this is the "RNN mode" of Peng et al. 2023
// (arXiv:2305.13048) — the parallel WKV formulation used by [RWKV.Forward]
// and this recurrence are the same computation, so stepping a prompt through
// [RWKV.DecodeStep] reproduces Forward's final-row logits exactly.
//
// For Go engineers: treat it as an opaque accumulator. Get one from
// [RWKV.NewDecodeState], pass it to every DecodeStep in order, and discard it
// to reset the sequence. It is not safe for concurrent use.
type RWKVDecodeState struct {
	Layers []*nn.RWKVState // one per block, same order as m.Blocks
}

// NewDecodeState returns the pre-first-token decode state: every layer's
// token-shift predecessor is the zero row and its WKV history is empty.
func (m *RWKV) NewDecodeState() *RWKVDecodeState {
	st := &RWKVDecodeState{Layers: make([]*nn.RWKVState, len(m.Blocks))}
	for l, b := range m.Blocks {
		st.Layers[l] = b.NewState()
	}
	return st
}

// embedOne returns the embedding of a single token, x[1,dim] = Embed[token].
func (m *RWKV) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	x := tensor.New(m.Embed.Dtype(), tensor.Shape{1, d})
	for j := range d {
		x.SetF64(m.Embed.AtF64(token, j), 0, j)
	}
	return x, nil
}

// DecodeStep advances the model one token in recurrent mode and returns the
// next-token logits [1,vocab]. The graph is the single-token twin of
// [RWKV.Forward]: embed → block-0 pre-LayerNorm (the HF "ln0", applied to the
// embedding of EVERY token before block 0, exactly as Forward normalises the
// whole embedded prompt) → per block one [nn.RWKVBlock.Step] (which carries
// the block's ln1/ln2, both residuals, the token-shift state and the WKV
// recurrence internally) → final LayerNorm → untied head. The state is
// updated in place; there is no position argument and no context limit —
// the recurrence is exact, so the logits match a full Forward over the same
// prefix bit-for-bit. Inference-only (no tape); training uses Forward.
func (m *RWKV) DecodeStep(ctx *backend.Context, st *RWKVDecodeState, token int) (*tensor.Tensor, error) {
	if len(st.Layers) != len(m.Blocks) {
		return nil, fmt.Errorf("nlp: RWKV decode state has %d layers, model has %d", len(st.Layers), len(m.Blocks))
	}
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// The pre_ln normalises the embedding before block 0 only (RWKV-4 quirk),
	// mirroring Forward's placement.
	if x, err = m.PreLN.Forward(ctx, x); err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		if x, err = b.Step(ctx, st.Layers[l], x); err != nil {
			return nil, err
		}
	}
	if x, err = m.LNOut.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Head)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the
// sampler s, running the O(1) recurrent state (one DecodeStep per token, no
// KV-cache). Prefill IS decoding here: each prompt token is stepped through
// the recurrence and the state absorbs it. Returns prompt+generated. With a
// greedy sampler the output is identical to argmax-ing a full Forward at each
// step. Unlike the attention decoders there is no context-length ceiling —
// the state is constant-size at any sequence length. The decode runs on
// backend.Default() unless WithBackend overrides it (§T361).
func (m *RWKV) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	st := m.NewDecodeState()
	out := append([]int(nil), prompt...)

	var logits *tensor.Tensor
	for _, tok := range prompt {
		l, err := m.DecodeStep(ctx, st, tok)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	for range maxNew {
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		l, err := m.DecodeStep(ctx, st, next)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	return out, nil
}
