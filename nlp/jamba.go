package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Jamba is AI21's hybrid Transformer–Mamba mixture-of-experts decoder (Lieber et
// al. 2024, arXiv:2403.19887) — the architecture that interleaves the two
// sequence-mixer families instead of picking one. Each block chooses its mixer
// and its feed-forward independently by two periodic, config-driven patterns:
//
//   - Mixer: layer i is Llama-style grouped-query ATTENTION when
//     i % attn_layer_period == attn_layer_offset, otherwise a MAMBA selective
//     scan. Jamba's attention applies NO rotary embedding (NoPE): position
//     information comes entirely from the recurrent Mamba layers, so queries
//     and keys go straight into causal attention.
//   - FFN: layer i is a sparse MoE (router + top-k SwiGLU experts) when
//     i % expert_layer_period == expert_layer_offset, otherwise a single dense
//     SwiGLU.
//
// Jamba's Mamba mixer differs from the plain [Mamba] mixer in exactly one way:
// RMSNorms on the x_proj outputs (dt_layernorm / b_layernorm / c_layernorm)
// applied to Δ_low, B and C before the Δ up-projection and the selective scan —
// they stabilise the hybrid stack at scale. Its MoE router differs from
// Mixtral's in exactly one way: the top-k softmax scores are used raw, WITHOUT
// renormalizing them to sum to 1.
//
// Every block is pre-norm residual: x += mixer(input_layernorm(x)) followed by
// x += ffn(pre_ff_layernorm(x)); final RMSNorm and an untied LM head finish the
// stack. Load a Hugging Face JambaForCausalLM checkpoint with [JambaFromHF].
type Jamba struct {
	Config JambaConfig    // geometry and the per-layer mixer/FFN pattern (see JambaConfig)
	TokEmb *tensor.Tensor // token embedding [vocab, dim]
	Layers []*JambaLayer  // the hybrid attention/Mamba + MoE/dense blocks
	Norm   *nn.RMSNorm    // model.final_layernorm
	Out    *tensor.Tensor // [dim, vocab] untied LM head (embedding transpose when tied)
}

// JambaConfig fixes the model geometry. Dim, Vocab, Layers, KVHeads and the
// per-layer types are inferred from the checkpoint by [JambaFromHF]; Heads,
// TopK and Eps come from config.json (TopK defaults to 2, Eps to 1e-6).
type JambaConfig struct {
	Vocab   int     // vocabulary size
	Dim     int     // embedding width (hidden_size)
	Heads   int     // num_attention_heads (attention layers)
	KVHeads int     // num_key_value_heads (GQA); 0 → Heads
	Layers  int     // number of hybrid blocks
	TopK    int     // num_experts_per_tok on MoE layers; 0 → 2
	Eps     float64 // rms_norm_eps
}

// JambaLayer is one pre-norm block. Exactly one of Attn/Mamba is non-nil (the
// sequence mixer) and exactly one of MoE/Dense is non-nil (the feed-forward).
type JambaLayer struct {
	InputNorm *nn.RMSNorm // pre-mixer RMSNorm (input_layernorm)
	PreFFNorm *nn.RMSNorm // pre-FFN RMSNorm (pre_ff_layernorm)

	Attn  *JambaAttention // attention mixer, or nil
	Mamba *JambaMixer     // Mamba mixer, or nil

	MoE   *JambaMoE  // sparse-MoE FFN, or nil
	Dense *nn.SwiGLU // dense SwiGLU FFN, or nil
}

// JambaAttention is Jamba's attention mixer: Llama-style grouped-query causal
// attention with bias-free projections and NO rotary embedding (NoPE) — the
// Mamba layers carry position, so transformers' JambaAttention.forward never
// rotates q/k.
type JambaAttention struct {
	Wq, Wk, Wv, Wo *tensor.Tensor // [dim, heads·hd], [dim, kv·hd], [dim, kv·hd], [heads·hd, dim]
}

// JambaMixer is Jamba's Mamba mixer: the plain selective scan of
// [nn.MambaBlock] plus Jamba's three extra RMSNorms on the x_proj outputs
// (dt/b/c layernorms). Block holds the shared weights with nn.MambaBlock field
// semantics; Forward re-wires the same ops with the norms inserted, mirroring
// transformers' JambaMambaMixer.slow_forward.
type JambaMixer struct {
	Block                *nn.MambaBlock // shared Mamba weights (in/x/dt/out projections, conv, A/D)
	DtNorm, BNorm, CNorm *nn.RMSNorm    // RMSNorms on Δ_low [dt_rank], B [N], C [N]
}

// Forward runs the mixer on u[L, dim]: in_proj split → causal depthwise conv +
// SiLU → x_proj split → RMSNorm(Δ_low/B/C) → Δ = softplus(dt_proj·Δ_low + bias)
// → selective scan with A = −exp(A_log) and the D skip → SiLU(z) gate →
// out_proj. Identical to nn.MambaBlock.Forward except the three norms.
func (m *JambaMixer) Forward(ctx *backend.Context, u *tensor.Tensor) (*tensor.Tensor, error) {
	b := m.Block
	xin, err := b.InX.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	z, err := b.InZ.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	// causal depthwise conv + SiLU
	xc, err := exec1(ctx, backend.OpConv1D, nil, xin, b.ConvW, b.ConvB)
	if err != nil {
		return nil, err
	}
	if xc, err = exec1(ctx, backend.OpSiLU, nil, xc); err != nil {
		return nil, err
	}
	// input-dependent Δ, B, C — each RMSNormed before use (Jamba's addition)
	dtLow, err := b.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if dtLow, err = m.DtNorm.Forward(ctx, dtLow); err != nil {
		return nil, err
	}
	dtPre, err := b.DtProj.Forward(ctx, dtLow)
	if err != nil {
		return nil, err
	}
	delta, err := exec1(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bMat, err := b.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if bMat, err = m.BNorm.Forward(ctx, bMat); err != nil {
		return nil, err
	}
	cMat, err := b.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if cMat, err = m.CNorm.Forward(ctx, cMat); err != nil {
		return nil, err
	}
	// A = −exp(A_log)
	expA, err := exec1(ctx, backend.OpExp, nil, b.ALog)
	if err != nil {
		return nil, err
	}
	A, err := exec1(ctx, backend.OpNeg, nil, expA)
	if err != nil {
		return nil, err
	}
	// selective scan, then gate y ⊙ SiLU(z) and down-project
	y, err := exec1(ctx, backend.OpSSM, nil, xc, delta, A, bMat, cMat, b.Dskip)
	if err != nil {
		return nil, err
	}
	gate, err := exec1(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	if y, err = exec1(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return b.OutProj.Forward(ctx, y)
}

// Params returns every trainable tensor of the mixer.
func (m *JambaMixer) Params() []*tensor.Tensor {
	ps := m.Block.Params()
	ps = append(ps, m.DtNorm.Gamma, m.BNorm.Gamma, m.CNorm.Gamma)
	return ps
}

// JambaMoE is Jamba's sparse-MoE feed-forward: a bias-free linear router over E
// SwiGLU experts, mixing the top-k with RAW softmax scores. Unlike Mixtral (and
// nn.SparseMoE's TopKGating), Jamba does NOT renormalize the selected weights —
// transformers' JambaSparseMoeBlock.route_tokens_to_experts softmaxes over all
// experts, takes top-k, and uses those probabilities as-is — so the mixture is
// computed from primitives here, exactly as [DeepSeekV2]'s moeFFN does for
// DeepSeek's un-normalized gating.
type JambaMoE struct {
	Router  *tensor.Tensor // [dim, E] router weight (feed_forward.router)
	Experts []*nn.SwiGLU   // E SwiGLU experts
	TopK    int            // experts per token
}

// Forward computes y = Σ_{i∈topk} softmax(x·Router)_i · Expert_i(x) for each
// token, with NO renormalization of the top-k weights. Only the experts some token routed to are
// evaluated: an unselected expert's contribution is its output times a weight that is EXACTLY
// zero, and adding an exact zero leaves a finite accumulator unchanged, so the result is the same
// bits as the dense sum rather than merely close (the two exceptions, a negative-zero accumulator
// sign and a NaN or Inf escaping an unrouted expert, are the trade [nn.SparseMoE.ForwardDecode]
// already makes).
func (m *JambaMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Router)
	if err != nil {
		return nil, err
	}
	scores, err := exec1(ctx, backend.OpSoftmax, nil, logits) // softmax over all E experts
	if err != nil {
		return nil, err
	}
	seq, e := scores.Shape()[0], scores.Shape()[1]

	// Per-token greedy top-k; weight = raw softmax score for selected experts,
	// 0 otherwise (no renormalization — Jamba's gating).
	weight := tensor.New(tensor.F64, tensor.Shape{seq, e})
	ws := weight.Storage().F64()
	used := make([]bool, e)
	for t := range seq {
		for _, i := range topKIndices(scores, t, e, m.TopK) {
			ws[t*e+i] = scores.AtF64(t, i)
			used[i] = true
		}
	}

	// Evaluate ONLY the experts top-k selected. An unselected expert contributes
	// expert_i(x) ⊙ 0, so dropping it leaves every surviving addend in the same relative order
	// and the same value — the two exceptions being a negative-zero accumulator sign and a NaN or
	// Inf escaping an expert nobody asked for, which is the trade [SparseMoE.ForwardDecode]
	// already makes. With TopK experts of E routed per token this is the difference between
	// streaming E sets of expert weights per step and streaming the ones that matter, and these
	// are GEMVs, so the step is bound by exactly that.
	var y *tensor.Tensor
	for i := range e {
		if !used[i] {
			continue
		}
		out, err := m.Experts[i].Forward(ctx, x) // [seq, dim]
		if err != nil {
			return nil, err
		}
		wcol, err := weight.Slice(1, i, i+1) // [seq, 1] broadcast column
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 variadic-slice alloc; expert-forward/matmul dominated, resource-only
		term, err := exec1(ctx, backend.OpMul, nil, out, wcol.Contiguous())
		if err != nil {
			return nil, err
		}
		if y == nil {
			y = term
			//perfscan:ignore PS6017 variadic-slice alloc; matmul-dominated layer loop, resource-only
		} else if y, err = exec1(ctx, backend.OpAdd, nil, y, term); err != nil {
			return nil, err
		}
	}
	if y == nil { // no tokens, so nothing was routed: keep the old shape rather than a nil
		return exec1(ctx, backend.OpMul, nil, x, tensor.New(x.Dtype(), tensor.Shape{1}))
	}
	return y, nil
}

// Params returns the router and every expert tensor.
func (m *JambaMoE) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Router}
	for _, e := range m.Experts {
		ps = append(ps, e.Params()...)
	}
	return ps
}

func (c JambaConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens: embed → per layer
// x += mixer(input_layernorm(x)); x += ffn(pre_ff_layernorm(x)) → final RMSNorm
// → untied head.
func (m *Jamba) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: Jamba prompt is empty")
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}

	cfg := m.Config
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: true}
	for _, l := range m.Layers {
		// mixer sublayer (attention or Mamba)
		xb, err := l.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var mix *tensor.Tensor
		if l.Attn != nil {
			// NoPE grouped-query causal attention: q/k are NOT rotated.
			//perfscan:ignore PS6017 variadic-slice alloc; OpMatMul dominated, resource-only no wallclock
			q, err := exec1(ctx, backend.OpMatMul, nil, xb, l.Attn.Wq)
			if err != nil {
				return nil, err
			}
			//perfscan:ignore PS6017 variadic-slice alloc; OpMatMul dominated, resource-only no wallclock
			k, err := exec1(ctx, backend.OpMatMul, nil, xb, l.Attn.Wk)
			if err != nil {
				return nil, err
			}
			//perfscan:ignore PS6017 variadic-slice alloc; OpMatMul dominated, resource-only no wallclock
			v, err := exec1(ctx, backend.OpMatMul, nil, xb, l.Attn.Wv)
			if err != nil {
				return nil, err
			}
			//perfscan:ignore PS6017 variadic-slice alloc; OpMHA dominated, resource-only no wallclock
			a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
			if err != nil {
				return nil, err
			}
			//perfscan:ignore PS6017 variadic-slice alloc; OpMatMul dominated, resource-only no wallclock
			if mix, err = exec1(ctx, backend.OpMatMul, nil, a, l.Attn.Wo); err != nil {
				return nil, err
			}
		} else {
			if mix, err = l.Mamba.Forward(ctx, xb); err != nil {
				return nil, err
			}
		}
		//perfscan:ignore PS6017 variadic-slice alloc; OpAdd on large tensor, resource-only no wallclock
		if x, err = exec1(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
		// FFN sublayer (sparse MoE or dense SwiGLU)
		xf, err := l.PreFFNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if l.MoE != nil {
			ff, err = l.MoE.Forward(ctx, xf)
		} else {
			ff, err = l.Dense.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 variadic-slice alloc; matmul-dominated layer loop, resource-only
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Params returns every trainable tensor for optimizers.
func (m *Jamba) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, l := range m.Layers {
		ps = append(ps, l.InputNorm.Gamma, l.PreFFNorm.Gamma)
		if l.Attn != nil {
			ps = append(ps, l.Attn.Wq, l.Attn.Wk, l.Attn.Wv, l.Attn.Wo)
		} else {
			ps = append(ps, l.Mamba.Params()...)
		}
		if l.MoE != nil {
			ps = append(ps, l.MoE.Params()...)
		} else {
			ps = append(ps, l.Dense.Params()...)
		}
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
