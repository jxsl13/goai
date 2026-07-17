package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// JambaFromGGUF builds a [Jamba] from the metadata and (dequantized) tensor
// maps of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose
// general.architecture is "jamba" — llama.cpp's arch string for AI21's hybrid
// Transformer–Mamba MoE (JambaForCausalLM). The config comes from the jamba.*
// metadata keys and the weights from the token_embd / blk.N.* / output_norm /
// output tensors; every projection is transposed from GGUF's torch [out, in]
// layout into GoAI's [in, out].
//
// The Jamba conventions this loader implements were verified against llama.cpp
// master (conversion/jamba.py JambaModel + src/models/jamba.cpp +
// src/models/mamba-base.cpp build_mamba_layer + gguf-py constants.py /
// tensor_mapping.py + src/llama-arch.cpp):
//
//   - The mixer interleave is encoded in the PER-LAYER head_count_kv VECTOR:
//     JambaModel.set_gguf_parameters writes jamba.attention.head_count_kv as an
//     array with n_kv_head at attention positions (i % attn_layer_period ==
//     attn_layer_offset) and 0 everywhere else, and src/models/jamba.cpp keys
//     the layer type off exactly that — `is_recr_impl[i] = n_head_kv(i) == 0`,
//     then loads SSM tensors when 0 and attention tensors otherwise. This
//     loader consumes the same vector (a scalar broadcasts, llama.cpp's
//     get_key_or_arr tolerance), requires it to cover block_count, requires
//     the matching tensors per layer, and derives the single GoAI KVHeads from
//     the (necessarily uniform) non-zero entries.
//   - Attention is NoPE: JambaModel writes NO rope.* metadata and the graph
//     applies none — src/models/jamba.cpp comments "No RoPE :)" and feeds
//     q/k straight into build_attn. Projections are bias-free pure renames
//     (q/k/v/o_proj → attn_q/k/v/attn_output; jamba.py has NO permute — the
//     llama-arch q/k interleave does not apply here), with q/o square
//     [dim, dim] and k/v [kv·hd, dim], hd = dim/heads.
//   - The Mamba mixer is the [MambaFromGGUF] convention verbatim (same
//     build_mamba_layer): packed ssm_in split at d_inner into x|z, packed
//     ssm_x split at dt_rank/d_state into Δ_low|B|C, ssm_conv1d stored
//     squeezed [d_inner, d_conv] (jamba.py applies the same squeeze), the Δ
//     bias on ssm_dt.bias applied inside softplus, ssm_a/ssm_d without the
//     ".weight" suffix, and ssm_a storing A = −exp(A_log) (jamba.py
//     "A_log --> A") — inverted here to ALog = ln(−ssm_a) with strictly
//     negative elements enforced. PLUS Jamba's weighted Δ/B/C norms as
//     DEDICATED tensors: blk.N.ssm_dt_norm.weight [dt_rank],
//     ssm_b_norm.weight [d_state], ssm_c_norm.weight [d_state]
//     (dt/b/c_layernorm in tensor_mapping.py), required on every Mamba layer
//     and applied by build_mamba_layer before the Δ up-projection and the
//     scan — GoAI's JambaMixer.DtNorm/BNorm/CNorm, all with the single
//     jamba.attention.layer_norm_rms_epsilon (default 1e-6).
//   - Per-layer FFN type comes from TENSOR PRESENCE, like llama.cpp's
//     TENSOR_NOT_REQUIRED router probe: blk.N.ffn_gate_inp.weight [E, dim]
//     present → sparse MoE with FUSED 3-D experts ffn_gate_exps/ffn_up_exps
//     [E, ffn, dim] and ffn_down_exps [E, dim, ffn] (jamba.py torch.stack's
//     the per-expert tensors "using the same merged name as qwen2moe" — the
//     Mixtral fused layout, unpacked per expert slice here); absent → dense
//     SwiGLU ffn_gate/ffn_up/ffn_down. Both widths are checked against
//     jamba.feed_forward_length and the expert count against
//     jamba.expert_count. The routing llama.cpp runs (build_moe_ffn with
//     norm_w=false, softmax gating) is Jamba's raw un-renormalized top-k —
//     exactly [JambaMoE].Forward; jamba.expert_used_count → TopK (default 2).
//   - Both per-block norms are required RMSNorms: blk.N.attn_norm.weight (HF
//     input_layernorm) before the mixer and blk.N.ffn_norm.weight (HF
//     pre_ff_layernorm) before the FFN — src/models/jamba.cpp's two
//     build_norm calls in the pre-norm residual graph, the same shape as
//     [Jamba.Forward].
//   - Head: output.weight is TENSOR_NOT_REQUIRED with a token_embd fallback
//     (src/models/jamba.cpp), so an absent tensor ties Out to TokEmbᵀ —
//     mirroring [JambaFromHF]'s lm_head fallback.
//
// jamba.context_length is metadata-only here (GoAI's Jamba has no rope/ctx
// coupling; attention is NoPE and the SSM state is unbounded). Vocab comes
// from token_embd.
func JambaFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Jamba, error) {
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

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if tok.Ndim() != 2 || tok.Shape()[1] != dim {
		return nil, fmt.Errorf("nlp: Jamba GGUF token_embd.weight %v does not match embedding_length %d", tok.Shape(), dim)
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Jamba{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		in, err := g("attn_norm.weight") // HF input_layernorm
		if err != nil {
			return nil, err
		}
		ff, err := g("ffn_norm.weight") // HF pre_ff_layernorm
		if err != nil {
			return nil, err
		}
		layer := &JambaLayer{InputNorm: rmsFromGGUF(in, cfg.Eps), PreFFNorm: rmsFromGGUF(ff, cfg.Eps)}

		// Mixer: n_head_kv(i) == 0 → recurrent (src/models/jamba.cpp), else GQA.
		if kvVec[l] > 0 {
			hd := dim / heads
			wq, err := g("attn_q.weight")
			if err != nil {
				return nil, err
			}
			wk, err := g("attn_k.weight")
			if err != nil {
				return nil, err
			}
			wv, err := g("attn_v.weight")
			if err != nil {
				return nil, err
			}
			wo, err := g("attn_output.weight")
			if err != nil {
				return nil, err
			}
			if wq.Ndim() != 2 || wq.Shape()[0] != heads*hd {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sattn_q.weight %v, want [heads·hd, dim] = [%d, %d]", p, wq.Shape(), heads*hd, dim)
			}
			if wk.Ndim() != 2 || wk.Shape()[0] != kvVec[l]*hd {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sattn_k.weight %v disagrees with head_count_kv[%d]=%d (want %d rows)", p, wk.Shape(), l, kvVec[l], kvVec[l]*hd)
			}
			layer.Attn = &JambaAttention{
				Wq: transpose2D(wq), Wk: transpose2D(wk), Wv: transpose2D(wv), Wo: transpose2D(wo),
			}
		} else {
			block, err := ssmMixerFromGGUF(tensors, p, dim, dInner, dConv, n, dtRank)
			if err != nil {
				return nil, err
			}
			dtNorm, err := g("ssm_dt_norm.weight")
			if err != nil {
				return nil, err
			}
			bNorm, err := g("ssm_b_norm.weight")
			if err != nil {
				return nil, err
			}
			cNorm, err := g("ssm_c_norm.weight")
			if err != nil {
				return nil, err
			}
			if dtNorm.Numel() != dtRank || bNorm.Numel() != n || cNorm.Numel() != n {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sssm_{dt,b,c}_norm sizes %d/%d/%d, want dt_rank=%d / d_state=%d", p, dtNorm.Numel(), bNorm.Numel(), cNorm.Numel(), dtRank, n)
			}
			layer.Mamba = &JambaMixer{
				Block:  block,
				DtNorm: rmsFromGGUF(dtNorm, cfg.Eps),
				BNorm:  rmsFromGGUF(bNorm, cfg.Eps),
				CNorm:  rmsFromGGUF(cNorm, cfg.Eps),
			}
		}

		// FFN: router presence decides, like llama.cpp's TENSOR_NOT_REQUIRED probe.
		if router, ok := tensors[p+"ffn_gate_inp.weight"]; ok {
			moe, err := jambaMoEFromGGUF(meta, tensors, p, router, dim, ffn, cfg.TopK)
			if err != nil {
				return nil, err
			}
			layer.MoE = moe
		} else {
			gate, err := g("ffn_gate.weight")
			if err != nil {
				return nil, err
			}
			up, err := g("ffn_up.weight")
			if err != nil {
				return nil, err
			}
			down, err := g("ffn_down.weight")
			if err != nil {
				return nil, err
			}
			if gate.Shape()[0] != ffn {
				return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_gate.weight width %d != jamba.feed_forward_length %d", p, gate.Shape()[0], ffn)
			}
			layer.Dense = swiGLUFromGGUF(gate, up, down)
		}
		m.Layers = append(m.Layers, layer)
	}

	on, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.Norm = rmsFromGGUF(on, cfg.Eps)
	head := tok // tied fallback (TENSOR_NOT_REQUIRED + token_embd duplicate)
	if o, ok := tensors["output.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head)
	m.Config = cfg
	return m, nil
}

// jambaMoEFromGGUF builds one block's [JambaMoE] from the GGUF router
// ([E, dim]) and fused expert tensors (gate/up [E, ffn, dim], down
// [E, dim, ffn] — jamba.py's torch.stack of the per-expert HF weights, the
// Mixtral layout), checking the expert count against jamba.expert_count and
// the width against jamba.feed_forward_length.
func jambaMoEFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor, p string, router *tensor.Tensor, dim, ffn, topK int) (*JambaMoE, error) {
	gate, okG := tensors[p+"ffn_gate_exps.weight"]
	up, okU := tensors[p+"ffn_up_exps.weight"]
	down, okD := tensors[p+"ffn_down_exps.weight"]
	if !okG || !okU || !okD {
		return nil, fmt.Errorf("nlp: GGUF %sffn_gate_inp.weight present but ffn_{gate,up,down}_exps.weight missing (a MoE layer needs the fused expert tensors)", p)
	}
	if gate.Ndim() != 3 || up.Ndim() != 3 || down.Ndim() != 3 {
		return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_*_exps must be 3-D, got %v / %v / %v", p, gate.Shape(), up.Shape(), down.Shape())
	}
	e := gate.Shape()[0]
	if ec, err := metaInt(meta, "jamba."+ggufExpertCnt); err == nil && ec != e {
		return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_gate_exps has %d experts, jamba.expert_count says %d", p, e, ec)
	}
	if up.Shape()[0] != e || down.Shape()[0] != e || router.Shape()[0] != e {
		return nil, fmt.Errorf("nlp: Jamba GGUF %s expert tensors disagree on the expert count (gate %v, up %v, down %v, router %v)", p, gate.Shape(), up.Shape(), down.Shape(), router.Shape())
	}
	if gate.Shape()[1] != ffn || up.Shape()[1] != ffn || down.Shape()[2] != ffn || gate.Shape()[2] != dim {
		return nil, fmt.Errorf("nlp: Jamba GGUF %sffn_*_exps shapes %v/%v/%v disagree with feed_forward_length %d / dim %d", p, gate.Shape(), up.Shape(), down.Shape(), ffn, dim)
	}
	moe := &JambaMoE{Router: transpose2D(router), TopK: topK} // [E,dim] → [dim,E]
	for i := range e {
		moe.Experts = append(moe.Experts, &nn.SwiGLU{
			Wgate: transpose2D(sub3D(gate, i)), // [ffn,dim] → [dim,ffn]
			Wup:   transpose2D(sub3D(up, i)),
			Wdown: transpose2D(sub3D(down, i)), // [dim,ffn] → [ffn,dim]
		})
	}
	return moe, nil
}

// JambaToGGUF is the inverse of [JambaFromGGUF]: it serializes a Jamba (e.g.
// from [JambaFromHF]) into GGUF metadata + tensor maps under
// general.architecture "jamba", exactly the way llama.cpp's converter lays the
// file out — the layer interleave as the per-layer head_count_kv vector (0 on
// Mamba layers, KVHeads on attention layers), NoPE attention as bias-free
// unpermuted attn_q/k/v/attn_output, the Mamba mixers with ssm_a = −exp(A_log)
// / squeezed conv / packed ssm_in+ssm_x / the dedicated ssm_{dt,b,c}_norm
// weights, MoE layers as ffn_gate_inp + fused 3-D ffn_{gate,up,down}_exps
// (torch.stack layout) and dense layers as ffn_gate/up/down. The SSM dims and
// feed_forward_length are recovered from the first Mamba/FFN layer;
// expert_count/expert_used_count are written only when a MoE layer exists
// (JambaModel always writes them, but a MoE-free model has no expert source
// and the loader's router probe never reads them). context_length is written
// as the 2^20 placeholder ([Jamba] carries no context field — attention is
// NoPE and the SSM state is unbounded — mirroring the mamba converter's
// "arbitrary value" default). Mirroring the on-disk reality of a tied
// checkpoint (no lm_head tensor to convert), output.weight is omitted when Out
// equals the embedding transpose. Pass the result to gguf.Write via a
// gguf.File.
func JambaToGGUF(m *Jamba) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "jamba"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	topk := c.TopK
	if topk <= 0 {
		topk = 2
	}
	kvVec := make([]any, len(m.Layers))
	experts := 0
	ffn := 0
	var ssmBlock *nn.MambaBlock
	for l, layer := range m.Layers {
		kv := 0
		if layer.Attn != nil {
			kv = c.kvHeads()
		} else if ssmBlock == nil {
			ssmBlock = layer.Mamba.Block
		}
		kvVec[l] = uint32(kv)
		if layer.MoE != nil {
			experts = len(layer.MoE.Experts)
			ffn = layer.MoE.Experts[0].Wgate.Shape()[1]
		} else if layer.Dense != nil {
			ffn = layer.Dense.Wgate.Shape()[1]
		}
	}
	meta := map[string]any{
		ggufArch:          arch,
		key(ggufCtxLen):   uint32(1 << 20), // placeholder; NoPE attention + unbounded SSM state
		key(ggufEmbLen):   uint32(c.Dim),
		key(ggufBlockCnt): uint32(c.Layers),
		key(ggufFFLen):    uint32(ffn),
		key(ggufHeadCnt):  uint32(c.Heads),
		key(ggufHeadKV):   kvVec,
		key(ggufRMSEps):   float32(c.Eps),
	}
	if ssmBlock != nil {
		meta[key(ggufSSMConvKernel)] = uint32(ssmBlock.DConv)
		meta[key(ggufSSMInnerSize)] = uint32(ssmBlock.DInner)
		meta[key(ggufSSMStateSize)] = uint32(ssmBlock.N)
		meta[key(ggufSSMDtRank)] = uint32(ssmBlock.DtRank)
	}
	if experts > 0 {
		meta[key(ggufExpertCnt)] = uint32(experts)
		meta[key(ggufExpertUsed)] = uint32(topk)
	}

	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
	}
	if !equalsTransposed(m.Out, m.TokEmb) {
		ts["output.weight"] = transpose2D(m.Out) // untied head, [dim,vocab] → [vocab,dim]
	}
	for l, layer := range m.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(layer.InputNorm.Gamma)
		ts[p+"ffn_norm.weight"] = cloneF64(layer.PreFFNorm.Gamma)
		if layer.Attn != nil {
			ts[p+"attn_q.weight"] = transpose2D(layer.Attn.Wq) // NoPE: no permute, GoAI [in,out] → torch [out,in]
			ts[p+"attn_k.weight"] = transpose2D(layer.Attn.Wk)
			ts[p+"attn_v.weight"] = transpose2D(layer.Attn.Wv)
			ts[p+"attn_output.weight"] = transpose2D(layer.Attn.Wo)
		} else {
			ssmMixerToGGUF(ts, p, layer.Mamba.Block)
			ts[p+"ssm_dt_norm.weight"] = cloneF64(layer.Mamba.DtNorm.Gamma)
			ts[p+"ssm_b_norm.weight"] = cloneF64(layer.Mamba.BNorm.Gamma)
			ts[p+"ssm_c_norm.weight"] = cloneF64(layer.Mamba.CNorm.Gamma)
		}
		if layer.MoE != nil {
			ts[p+"ffn_gate_inp.weight"] = transpose2D(layer.MoE.Router) // [dim,E] → [E,dim]
			gates := make([]*tensor.Tensor, len(layer.MoE.Experts))
			ups := make([]*tensor.Tensor, len(layer.MoE.Experts))
			downs := make([]*tensor.Tensor, len(layer.MoE.Experts))
			for e, ex := range layer.MoE.Experts {
				gates[e] = transpose2D(ex.Wgate) // [dim,ffn] → [ffn,dim]
				ups[e] = transpose2D(ex.Wup)
				downs[e] = transpose2D(ex.Wdown) // [ffn,dim] → [dim,ffn]
			}
			ts[p+"ffn_gate_exps.weight"] = stack3D(gates)
			ts[p+"ffn_up_exps.weight"] = stack3D(ups)
			ts[p+"ffn_down_exps.weight"] = stack3D(downs)
		} else {
			ts[p+"ffn_gate.weight"] = transpose2D(layer.Dense.Wgate)
			ts[p+"ffn_up.weight"] = transpose2D(layer.Dense.Wup)
			ts[p+"ffn_down.weight"] = transpose2D(layer.Dense.Wdown)
		}
	}
	return meta, ts
}

// metaIntVec reads a GGUF metadata value as a per-layer int vector of length
// layers. GGUF arrays arrive as []any from the reader; a scalar broadcasts to
// every layer (llama.cpp's get_key_or_arr tolerance — Jamba's converter always
// writes the array form for head_count_kv).
func metaIntVec(meta map[string]any, key string, layers int) ([]int, error) {
	arr, ok := meta[key].([]any)
	if !ok {
		// Scalar broadcast.
		v, err := metaInt(meta, key)
		if err != nil {
			return nil, err
		}
		out := make([]int, layers)
		for i := range out {
			out[i] = v
		}
		return out, nil
	}
	if len(arr) != layers {
		return nil, fmt.Errorf("nlp: GGUF %s has %d entries, want one per layer (%d)", key, len(arr), layers)
	}
	out := make([]int, layers)
	for i, e := range arr {
		switch n := e.(type) {
		case uint32:
			out[i] = int(n)
		case int32:
			out[i] = int(n)
		case uint64:
			out[i] = int(n)
		case int64:
			out[i] = int(n)
		default:
			return nil, fmt.Errorf("nlp: GGUF %s[%d] is %T, want an integer", key, i, e)
		}
	}
	return out, nil
}
