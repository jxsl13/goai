package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantJambaFromGGUF builds a [QuantJamba] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "jamba" (gguf.ReadRaw) —
// the quantized twin of [JambaFromGGUF], and the hybrid capstone of the quantized
// loaders: a llama.cpp-quantized Jamba checkpoint loads straight into QuantLinear
// projections without ever materializing full-precision weights, with each of the
// four per-layer flavors handled by the byte-preserving primitive its component
// architecture already proved out.
//
// The jamba conventions documented at [JambaFromGGUF] carry over, split by what
// stays quantized:
//
//   - The mixer interleave comes from the PER-LAYER jamba.attention.head_count_kv
//     VECTOR: n_head_kv(i) == 0 → recurrent (SSM tensors required), non-zero → GQA
//     attention (llama.cpp's is_recr_impl keying). The uniform non-zero entry
//     becomes the single GoAI KVHeads.
//   - Attention is NoPE and bias-free: attn_q/k/v/attn_output wrap DIRECTLY as
//     QuantLinear — GGUF's [out, in] row layout is exactly what QuantLinear
//     consumes, and jamba's converter applies NO q/k permute (no RoPE → nothing to
//     un-permute on the Q-block bytes, unlike [QuantMixtralFromGGUF]).
//   - The Mamba mixer is the [QuantMambaFromGGUF] convention verbatim (shared
//     [quantSSMMixerFromGGUF]: packed ssm_in / ssm_x split on the quantized bytes,
//     ssm_a kept as stored — −exp(A_log), gated strictly negative — conv and the
//     1-D tensors decoded to f32) PLUS Jamba's dedicated ssm_{dt,b,c}_norm.weight
//     gains, decoded to f32 like every norm (1-D tensors never block-quantize).
//   - The FFN type comes from TENSOR PRESENCE, llama.cpp's router probe: a present
//     blk.N.ffn_gate_inp.weight → sparse MoE with the router decoded to f32 (never
//     quantized in real files — see [QuantJambaMoE]) and the FUSED 3-D
//     ffn_{gate,up,down}_exps un-fused per expert via [quantSliceExpert], a
//     lossless byte-range copy; absent → dense ffn_gate/up/down wrapped directly.
//   - TIED head with an optional override: a present output.weight wraps as the
//     head; an absent one ties the head to token_embd (llama.cpp's
//     TENSOR_DUPLICATED fallback), which must then itself be stored in a
//     quantized-matmul format. token_embd always also feeds the f32 lookup table
//     via dequantization, whatever its storage type.
//
// Dimensions come from metadata and are cross-checked against every tensor
// (mirroring the float loader — a mismatched packed tensor would otherwise split at
// wrong row offsets and misload silently); Vocab comes from token_embd.
func QuantJambaFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantJamba, error) {
	const arch = "jamba"
	if a, _ := meta[ggufArch].(string); a != arch {
		return nil, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return nil, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return nil, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return nil, err
	}
	ffn, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return nil, err
	}
	kvVec, err := metaIntVec(meta, key(ggufHeadKV), layers)
	if err != nil {
		return nil, err
	}
	dConv, err := metaInt(meta, key(ggufSSMConvKernel))
	if err != nil {
		return nil, err
	}
	dInner, err := metaInt(meta, key(ggufSSMInnerSize))
	if err != nil {
		return nil, err
	}
	n, err := metaInt(meta, key(ggufSSMStateSize))
	if err != nil {
		return nil, err
	}
	dtRank, err := metaInt(meta, key(ggufSSMDtRank))
	if err != nil {
		return nil, err
	}

	cfg := JambaConfig{
		Dim: dim, Heads: heads, Layers: layers, TopK: 2,
		Eps: metaFloat(meta, key(ggufRMSEps), 1e-6),
	}
	if k, e := metaInt(meta, key(ggufExpertUsed)); e == nil {
		cfg.TopK = k
	}
	// The uniform non-zero kv entry is GoAI's single KVHeads; llama.cpp keeps a
	// per-layer array, but Jamba's converter only ever writes one value.
	for i, kv := range kvVec {
		if kv == 0 {
			continue
		}
		if cfg.KVHeads != 0 && cfg.KVHeads != kv {
			return nil, fmt.Errorf("nlp: Jamba GGUF head_count_kv mixes %d and %d across attention layers; GoAI's Jamba has a single KVHeads", cfg.KVHeads, kv)
		}
		cfg.KVHeads = kv
		if heads <= 0 || dim%heads != 0 {
			return nil, fmt.Errorf("nlp: Jamba GGUF dim %d not divisible by head_count %d (layer %d is attention)", dim, heads, i)
		}
	}

	get := func(name string) (gguf.QuantTensor, error) {
		qt, ok := tensors[name]
		if !ok {
			return gguf.QuantTensor{}, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		return qt, nil
	}
	// wrap a GGUF projection (quantized [out, in]) as a QuantLinear — no transpose,
	// no requant (and no permute: jamba's attention is NoPE).
	wrapQ := func(name string, qt gguf.QuantTensor) (*nn.QuantLinear, error) {
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s ggml type %d is not a quantized-matmul format", name, qt.GGType)
		}
		return &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}, nil
	}
	mkQ := func(name string) (*nn.QuantLinear, error) {
		qt, err := get(name)
		if err != nil {
			return nil, err
		}
		return wrapQ(name, qt)
	}
	// decode a REQUIRED small tensor (norm gain) to f32.
	mkF32 := func(name string) (*tensor.Tensor, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		v, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return v, nil
	}
	mkRMS := func(name string) (*nn.RMSNorm, error) {
		g, err := mkF32(name)
		if err != nil {
			return nil, err
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if len(tok.Shape) != 2 || tok.Shape[1] != dim {
		return nil, fmt.Errorf("nlp: Jamba GGUF token_embd.weight %v does not match embedding_length %d", tok.Shape, dim)
	}
	cfg.Vocab = tok.Shape[0]
	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantJamba{Config: cfg, TokEmb: emb}
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		layer := &QuantJambaLayer{}
		if layer.InputNorm, err = mkRMS(p + "attn_norm.weight"); err != nil { // HF input_layernorm
			return nil, err
		}
		if layer.PreFFNorm, err = mkRMS(p + "ffn_norm.weight"); err != nil { // HF pre_ff_layernorm
			return nil, err
		}

		// Mixer: n_head_kv(i) == 0 → recurrent (src/models/jamba.cpp), else GQA.
		if kvVec[l] > 0 {
			hd := dim / heads
			wq, err := get(p + "attn_q.weight")
			if err != nil {
				return nil, err
			}
			wk, err := get(p + "attn_k.weight")
			if err != nil {
				return nil, err
			}
			if len(wq.Shape) != 2 || wq.Shape[0] != heads*hd {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sattn_q.weight %v, want [heads·hd, dim] = [%d, %d]", p, wq.Shape, heads*hd, dim)
			}
			if len(wk.Shape) != 2 || wk.Shape[0] != kvVec[l]*hd {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sattn_k.weight %v disagrees with head_count_kv[%d]=%d (want %d rows)", p, wk.Shape, l, kvVec[l], kvVec[l]*hd)
			}
			qa := &QuantJambaAttention{}
			if qa.Wq, err = wrapQ(p+"attn_q.weight", wq); err != nil {
				return nil, err
			}
			if qa.Wk, err = wrapQ(p+"attn_k.weight", wk); err != nil {
				return nil, err
			}
			if qa.Wv, err = mkQ(p + "attn_v.weight"); err != nil {
				return nil, err
			}
			if qa.Wo, err = mkQ(p + "attn_output.weight"); err != nil {
				return nil, err
			}
			layer.Attn = qa
		} else {
			block, err := quantSSMMixerFromGGUF(tensors, p, dim, dInner, dConv, n, dtRank)
			if err != nil {
				return nil, err
			}
			dtNorm, err := mkF32(p + "ssm_dt_norm.weight")
			if err != nil {
				return nil, err
			}
			bNorm, err := mkF32(p + "ssm_b_norm.weight")
			if err != nil {
				return nil, err
			}
			cNorm, err := mkF32(p + "ssm_c_norm.weight")
			if err != nil {
				return nil, err
			}
			if dtNorm.Numel() != dtRank || bNorm.Numel() != n || cNorm.Numel() != n {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sssm_{dt,b,c}_norm sizes %d/%d/%d, want dt_rank=%d / d_state=%d", p, dtNorm.Numel(), bNorm.Numel(), cNorm.Numel(), dtRank, n)
			}
			layer.Mamba = &QuantJambaMixer{
				Block:  block,
				DtNorm: &nn.RMSNorm{Gamma: dtNorm, Eps: cfg.Eps},
				BNorm:  &nn.RMSNorm{Gamma: bNorm, Eps: cfg.Eps},
				CNorm:  &nn.RMSNorm{Gamma: cNorm, Eps: cfg.Eps},
			}
		}

		// FFN: router presence decides, like llama.cpp's TENSOR_NOT_REQUIRED probe.
		if router, ok := tensors[p+"ffn_gate_inp.weight"]; ok {
			moe, err := quantJambaMoEFromGGUF(meta, tensors, p, router, dim, ffn, cfg.TopK)
			if err != nil {
				return nil, err
			}
			layer.MoE = moe
		} else {
			gate, err := get(p + "ffn_gate.weight")
			if err != nil {
				return nil, err
			}
			if len(gate.Shape) != 2 || gate.Shape[0] != ffn {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_gate.weight width %v != jamba.feed_forward_length %d", p, gate.Shape, ffn)
			}
			d := &nn.QuantSwiGLU{}
			if d.Gate, err = wrapQ(p+"ffn_gate.weight", gate); err != nil {
				return nil, err
			}
			if d.Up, err = mkQ(p + "ffn_up.weight"); err != nil {
				return nil, err
			}
			if d.Down, err = mkQ(p + "ffn_down.weight"); err != nil {
				return nil, err
			}
			layer.Dense = d
		}
		q.Layers = append(q.Layers, layer)
	}

	if q.Norm, err = mkRMS("output_norm.weight"); err != nil {
		return nil, err
	}
	// Tied head with optional override: absent output.weight → the quantized
	// token_embd bytes (llama.cpp's TENSOR_DUPLICATED fallback); present → untied.
	if o, ok := tensors["output.weight"]; ok {
		if len(o.Shape) != 2 || o.Shape[0] != cfg.Vocab || o.Shape[1] != dim {
			return nil, fmt.Errorf("nlp: Jamba GGUF output.weight %v, want [vocab, dim] = [%d, %d]", o.Shape, cfg.Vocab, dim)
		}
		if q.Out, err = wrapQ("output.weight", o); err != nil {
			return nil, err
		}
	} else if q.Out, err = wrapQ("token_embd.weight", tok); err != nil {
		return nil, err
	}
	return q, nil
}

// quantJambaMoEFromGGUF builds one block's [QuantJambaMoE] from the GGUF router
// ([E, dim], decoded to f32 — never quantized in real files, see [QuantJambaMoE])
// and the fused quantized expert tensors (gate/up [E, ffn, dim], down [E, dim, ffn]
// — jamba.py's torch.stack, the Mixtral layout), un-fusing each expert with
// [quantSliceExpert] — a lossless byte-range copy, no dequantization — and checking
// the expert count against jamba.expert_count and the width against
// jamba.feed_forward_length.
func quantJambaMoEFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor, p string, router gguf.QuantTensor, dim, ffn, topK int) (*QuantJambaMoE, error) {
	if len(router.Shape) != 2 || router.Shape[1] != dim {
		return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_gate_inp.weight shape %v, want [experts, %d]", p, router.Shape, dim)
	}
	rt, err := router.Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF %sffn_gate_inp.weight: %w", p, err)
	}
	moe := &QuantJambaMoE{Router: f32Transpose(rt), TopK: topK} // [E,dim] → [dim,E]

	exps := make([]gguf.QuantTensor, 3)
	for i, name := range []string{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} {
		qt, ok := tensors[p+name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF %sffn_gate_inp.weight present but %s%s missing (a MoE layer needs the fused expert tensors)", p, p, name)
		}
		if len(qt.Shape) != 3 {
			return nil, fmt.Errorf("nlp: Jamba GGUF %s%s must be fused 3-D, got %v", p, name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s%s ggml type %d is not a quantized-matmul format", p, name, qt.GGType)
		}
		exps[i] = qt
	}
	gate, up, down := exps[0], exps[1], exps[2]
	e := gate.Shape[0]
	if ec, err := metaInt(meta, "jamba."+ggufExpertCnt); err == nil && ec != e {
		return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_gate_exps has %d experts, jamba.expert_count says %d", p, e, ec)
	}
	if up.Shape[0] != e || down.Shape[0] != e || router.Shape[0] != e {
		return nil, fmt.Errorf("nlp: Jamba GGUF %s expert tensors disagree on the expert count (gate %v, up %v, down %v, router %v)", p, gate.Shape, up.Shape, down.Shape, router.Shape)
	}
	if gate.Shape[1] != ffn || up.Shape[1] != ffn || down.Shape[2] != ffn || gate.Shape[2] != dim {
		return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_*_exps shapes %v/%v/%v disagree with feed_forward_length %d / dim %d", p, gate.Shape, up.Shape, down.Shape, ffn, dim)
	}
	for x := range e {
		mk := func(fused gguf.QuantTensor, name string) (*nn.QuantLinear, error) {
			s, err := quantSliceExpert(fused, x)
			if err != nil {
				return nil, fmt.Errorf("nlp: GGUF %s%s expert %d: %w", p, name, x, err)
			}
			return &nn.QuantLinear{Weight: s.Data, QT: gguf.QuantType(s.GGType), In: s.Shape[1], Out: s.Shape[0]}, nil
		}
		g, err := mk(gate, "ffn_gate_exps.weight")
		if err != nil {
			return nil, err
		}
		u, err := mk(up, "ffn_up_exps.weight")
		if err != nil {
			return nil, err
		}
		d, err := mk(down, "ffn_down_exps.weight")
		if err != nil {
			return nil, err
		}
		moe.Experts = append(moe.Experts, &nn.QuantSwiGLU{Gate: g, Up: u, Down: d})
	}
	return moe, nil
}
