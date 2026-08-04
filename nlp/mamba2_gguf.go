package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// ggufSSMGroupCount is the Mamba-2 grouping key (gguf-py constants.py
// Keys.SSM.GROUP_COUNT = "{arch}.ssm.group_count", mirrored by
// src/llama-arch.cpp LLM_KV_SSM_GROUP_COUNT). Arch-prefixed like the rest:
// "mamba2.ssm.group_count".
const ggufSSMGroupCount = "ssm.group_count"

// Mamba2FromGGUF builds a [Mamba2] from the metadata and (dequantized) tensor
// maps of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose
// general.architecture is "mamba2" — llama.cpp's arch string for the
// state-space-duality family (Mamba2ForCausalLM). The config comes from the
// mamba2.* metadata keys and the weights from the token_embd / blk.N.ssm_* /
// output_norm tensors.
//
// The Mamba-2 conventions this loader implements were verified against
// llama.cpp master (conversion/mamba.py Mamba2Model +
// src/models/mamba2.cpp load_arch_tensors + src/models/mamba-base.cpp
// build_mamba2_layer + gguf-py constants.py / tensor_mapping.py +
// src/llama-arch.cpp):
//
//   - ssm_in stays PACKED: Mamba2Model.modify_tensors has NO branch for
//     SSM_IN, so blk.N.ssm_in.weight is HF's fused in_proj
//     [d_in_proj, d_model] unchanged (pure rename),
//     d_in_proj = 2·d_inner + 2·n_group·d_state + n_head — and
//     build_mamba2_layer view-splits the PRODUCT zxBCdt at row offsets
//     d_inner (z | xBC) and 2·d_inner + 2·n_group·d_state (xBC | dt).
//     That packed [z | xBC | dt] split order is exactly what GoAI's
//     [Mamba2Mixer] applies to its packed InProj, so the tensor is copied
//     with NO split and NO transpose (the mixer dots over the input axis).
//   - ssm.time_step_rank stores NUM_HEADS, not a Δ low-rank: mamba-2 has no
//     dt_proj weight at all, and Mamba2Model writes
//     add_ssm_time_step_rank(d_inner // head_dim). build_mamba2_layer reads
//     it back as `n_head = hparams.ssm_dt_rank` and derives
//     head_dim = d_inner / n_head; ssm.group_count carries n_groups. This
//     loader does the same (HeadDim = Intermediate / NumHeads).
//   - ssm_dt is BIAS-ONLY: the converter renames HF's mixer.dt_bias to
//     dt_proj.bias (filter_tensors' ".dt_bias" → ".dt_proj.bias"), mapping to
//     blk.N.ssm_dt.bias [n_head]; there is NO blk.N.ssm_dt.weight
//     (load_arch_tensors creates only ssm_dt_b). The graph computes
//     Δ = softplus(dt + dt_bias) from the in_proj's dt band directly.
//   - ssm_a stores A = −exp(A_log) UNSQUEEZED to [n_head, 1]: modify_tensors
//     first reshapes A_log (and D) with `(*shape, 1)` ("unsqueeze A to use
//     similar shape semantics as Mamba-1"), then applies
//     `-torch.exp(data_torch)` ("A_log --> A"); load_arch_tensors creates
//     ggml {1, dt_rank} = row-major [n_head, 1]. Like [MambaFromGGUF], this
//     loader INVERTS the value transform (ALog = ln(−ssm_a), flattened to
//     [n_head]) and rejects any non-negative element — a file from a
//     different convention, not NaN-loadable. ssm_d is [n_head, 1] too,
//     flattened here; both carry NO ".weight" suffix (llama-arch.cpp: 'no
//     "weight" suffix for these').
//   - ssm_conv1d.weight is stored SQUEEZED, [conv_dim, d_conv] with
//     conv_dim = d_inner + 2·n_group·d_state (the conv covers x AND B AND C):
//     the converter squeezes HF's size-1 middle axis, and load_arch_tensors
//     reads ggml {d_conv, conv_dim} = row-major [conv_dim, d_conv] — exactly
//     Mamba2Mixer.ConvW, copied with NO transpose. ssm_conv1d.bias
//     [conv_dim] is required.
//   - ssm_norm.weight is stored RESHAPED PER GROUP: modify_tensors reshapes
//     the [d_inner] gated-norm gain to [n_group, d_inner/n_group]
//     (load_arch_tensors: ggml {d_inner/n_group, n_group}). The reshape is a
//     contiguous row-major regrouping, so this loader flattens it back to
//     the [d_inner] vector transformers holds. NOTE the one RUNTIME-semantics
//     divergence between the references: build_mamba2_layer applies the gated
//     RMSNorm PER GROUP (normalizing over d_inner/n_group elements), while
//     transformers' MambaRMSNormGated — which GoAI's [Mamba2Mixer] mirrors
//     bit-for-bit against its HF golden — normalizes over the FULL d_inner
//     width. The stored weights are identical either way and the two agree
//     exactly when n_groups = 1 (every official HF Mamba2ForCausalLM
//     release); GoAI keeps the transformers semantics its golden anchors.
//   - ssm_out.weight is torch [d_model, d_inner] unchanged — Mamba2Mixer's
//     OutProj orientation, copied with no transpose.
//   - Per-block norm is blk.N.attn_norm.weight (llama.cpp reuses the
//     LLM_TENSOR_ATTN_NORM slot for backbone.layers.N.norm despite there
//     being no attention), a standard RMSNorm with the epsilon from
//     mamba2.attention.layer_norm_rms_epsilon (default 1e-5).
//   - TIED head with an optional override: output.weight is
//     TENSOR_NOT_REQUIRED in load_arch_tensors with a token_embd duplicate
//     fallback — an absent output.weight ties Head to Embedᵀ, a present one
//     overrides it. (A tied HF checkpoint never materializes lm_head in its
//     state dict, so the converter writes no output.weight for it.)
//
// Dimensions come from metadata and are cross-checked against every tensor
// (mirroring llama.cpp's fixed create_tensor shapes); Vocab comes from
// token_embd. Metadata the graph never consumes (the converter's placeholder
// context_length 2^20, feed_forward_length 0, attention.head_count 0) is
// ignored. Unlike "mamba" there is no ssm.dt_b_c_rms key for this arch.
func Mamba2FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Mamba2, error) {
	cfg, err := mamba2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	convDim := cfg.Intermediate + 2*cfg.NGroups*cfg.N
	dInProj := cfg.Intermediate + convDim + cfg.NumHeads

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if tok.Ndim() != 2 || tok.Shape()[1] != cfg.DModel {
		return nil, fmt.Errorf("nlp: Mamba2 GGUF token_embd.weight %v does not match embedding_length %d", tok.Shape(), cfg.DModel)
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Mamba2{Config: cfg, Embed: cloneF64(tok)}
	//perfscan:ignore PS5001 false-positive on range-loop; cold loader path
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		norm, err := g("attn_norm.weight")
		if err != nil {
			return nil, err
		}
		inProj, err := g("ssm_in.weight")
		if err != nil {
			return nil, err
		}
		convW, err := g("ssm_conv1d.weight")
		if err != nil {
			return nil, err
		}
		convB, err := g("ssm_conv1d.bias")
		if err != nil {
			return nil, err
		}
		dtB, err := g("ssm_dt.bias") // bias only — mamba-2 has no dt_proj weight
		if err != nil {
			return nil, err
		}
		a, err := g("ssm_a") // no ".weight" suffix (llama-arch.cpp)
		if err != nil {
			return nil, err
		}
		d, err := g("ssm_d") // no ".weight" suffix
		if err != nil {
			return nil, err
		}
		normW, err := g("ssm_norm.weight")
		if err != nil {
			return nil, err
		}
		outProj, err := g("ssm_out.weight")
		if err != nil {
			return nil, err
		}

		// Mirror llama.cpp's fixed create_tensor shapes — a mismatched tensor
		// would otherwise view-split at wrong row offsets and misload silently.
		if inProj.Ndim() != 2 || inProj.Shape()[0] != dInProj || inProj.Shape()[1] != cfg.DModel {
			return nil, fmt.Errorf("nlp: GGUF %sssm_in.weight %v, want packed [2·d_inner+2·n_group·d_state+n_head, d_model] = [%d, %d]", p, inProj.Shape(), dInProj, cfg.DModel)
		}
		if convW.Ndim() != 2 || convW.Shape()[0] != convDim || convW.Shape()[1] != cfg.DConv {
			return nil, fmt.Errorf("nlp: GGUF %sssm_conv1d.weight %v, want squeezed [d_inner+2·n_group·d_state, d_conv] = [%d, %d] (the converter drops HF's size-1 middle axis)", p, convW.Shape(), convDim, cfg.DConv)
		}
		if convB.Ndim() != 1 || convB.Shape()[0] != convDim {
			return nil, fmt.Errorf("nlp: GGUF %sssm_conv1d.bias %v, want [%d]", p, convB.Shape(), convDim)
		}
		if dtB.Ndim() != 1 || dtB.Shape()[0] != cfg.NumHeads {
			return nil, fmt.Errorf("nlp: GGUF %sssm_dt.bias %v, want [n_head] = [%d]", p, dtB.Shape(), cfg.NumHeads)
		}
		if a.Ndim() != 2 || a.Shape()[0] != cfg.NumHeads || a.Shape()[1] != 1 {
			return nil, fmt.Errorf("nlp: GGUF %sssm_a %v, want unsqueezed [n_head, 1] = [%d, 1] (the mamba2 converter's Mamba-1-like shape semantics)", p, a.Shape(), cfg.NumHeads)
		}
		if d.Ndim() != 2 || d.Shape()[0] != cfg.NumHeads || d.Shape()[1] != 1 {
			return nil, fmt.Errorf("nlp: GGUF %sssm_d %v, want unsqueezed [n_head, 1] = [%d, 1]", p, d.Shape(), cfg.NumHeads)
		}
		if normW.Ndim() != 2 || normW.Shape()[0] != cfg.NGroups || normW.Shape()[1] != cfg.Intermediate/cfg.NGroups {
			return nil, fmt.Errorf("nlp: GGUF %sssm_norm.weight %v, want group-reshaped [n_group, d_inner/n_group] = [%d, %d]", p, normW.Shape(), cfg.NGroups, cfg.Intermediate/cfg.NGroups)
		}
		if outProj.Ndim() != 2 || outProj.Shape()[0] != cfg.DModel || outProj.Shape()[1] != cfg.Intermediate {
			return nil, fmt.Errorf("nlp: GGUF %sssm_out.weight %v, want [d_model, d_inner] = [%d, %d]", p, outProj.Shape(), cfg.DModel, cfg.Intermediate)
		}

		aLog, err := alogFromGGUFA(a, p) // ln(−ssm_a), strictly-negative gate
		if err != nil {
			return nil, err
		}
		mixer := &Mamba2Mixer{
			InProj: cloneF64(inProj), // packed [z|xBC|dt], torch orientation
			ConvW:  cloneF64(convW),  // stored squeezed — no transpose
			ConvB:  cloneF64(convB),  // [conv_dim]
			//perfscan:ignore PS6016 reshape literal in loader, one-time load resource-only
			ALog: reshapeF64(aLog, tensor.Shape{cfg.NumHeads}), // [n_head] (unsqueeze undone)
			//perfscan:ignore PS6016 reshape literal in loader, one-time load resource-only
			D:      reshapeF64(d, tensor.Shape{cfg.NumHeads}), // [n_head]
			DtBias: cloneF64(dtB),                             // [n_head]
			//perfscan:ignore PS6016 reshape literal in loader, one-time load resource-only
			NormW:   reshapeF64(normW, tensor.Shape{cfg.Intermediate}), // group reshape flattened
			OutProj: cloneF64(outProj),                                 // [d_model, d_inner], torch orientation
			DModel:  cfg.DModel, NumHeads: cfg.NumHeads, HeadDim: cfg.HeadDim,
			NGroups: cfg.NGroups, N: cfg.N, DConv: cfg.DConv, Intermediate: cfg.Intermediate,
			ConvDim: convDim, Eps: cfg.Eps,
		}
		m.Layers = append(m.Layers, Mamba2Layer{Norm: rmsFromGGUF(norm, cfg.Eps), Mixer: mixer})
	}

	on, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.Norm = rmsFromGGUF(on, cfg.Eps)
	// Tied head with optional override: absent output.weight → Embedᵀ
	// (llama.cpp's TENSOR_DUPLICATED fallback).
	if o, ok := tensors["output.weight"]; ok {
		if o.Ndim() != 2 || o.Shape()[0] != cfg.Vocab || o.Shape()[1] != cfg.DModel {
			return nil, fmt.Errorf("nlp: Mamba2 GGUF output.weight %v, want [vocab, d_model] = [%d, %d]", o.Shape(), cfg.Vocab, cfg.DModel)
		}
		m.Head = transpose2D(o)
	} else {
		m.Head = transpose2D(m.Embed)
	}
	m.Config = cfg
	return m, nil
}

// mamba2CfgFromGGUFMeta parses the mamba2.* metadata keys of a GGUF file whose
// general.architecture is "mamba2" into a Mamba2Config. The key mapping
// follows conversion/mamba.py Mamba2Model.set_gguf_parameters and
// src/models/mamba2.cpp load_arch_hparams: ssm.inner_size is the intermediate
// (expand·d_model) width, ssm.time_step_rank REUSES the Mamba-1 slot to carry
// NUM_HEADS (= d_inner // head_dim; mamba-2 has no Δ low-rank), and
// ssm.group_count carries n_groups; HeadDim is derived as
// Intermediate/NumHeads exactly like build_mamba2_layer's
// `head_dim = d_inner / n_head`. Vocab is left for the tensor loaders
// (token_embd-derived). Shared by [Mamba2FromGGUF] and [QuantMamba2FromGGUF].
func mamba2CfgFromGGUFMeta(meta map[string]any) (Mamba2Config, error) {
	const arch = "mamba2"
	if a, _ := meta[ggufArch].(string); a != arch {
		return Mamba2Config{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dModel, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return Mamba2Config{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return Mamba2Config{}, err
	}
	dConv, err := metaInt(meta, key(ggufSSMConvKernel))
	if err != nil {
		return Mamba2Config{}, err
	}
	inner, err := metaInt(meta, key(ggufSSMInnerSize))
	if err != nil {
		return Mamba2Config{}, err
	}
	n, err := metaInt(meta, key(ggufSSMStateSize))
	if err != nil {
		return Mamba2Config{}, err
	}
	nHeads, err := metaInt(meta, key(ggufSSMDtRank)) // ssm.time_step_rank = n_head for mamba2
	if err != nil {
		return Mamba2Config{}, err
	}
	nGroups, err := metaInt(meta, key(ggufSSMGroupCount))
	if err != nil {
		return Mamba2Config{}, err
	}
	if dModel <= 0 || layers <= 0 || dConv <= 0 || n <= 0 {
		return Mamba2Config{}, fmt.Errorf("nlp: Mamba2 GGUF has non-positive geometry (d_model %d, layers %d, conv %d, state %d)", dModel, layers, dConv, n)
	}
	if nHeads <= 0 || inner <= 0 || inner%nHeads != 0 {
		return Mamba2Config{}, fmt.Errorf("nlp: Mamba2 GGUF ssm.inner_size %d not divisible by ssm.time_step_rank %d (= n_head; head_dim = inner/n_head)", inner, nHeads)
	}
	if nGroups <= 0 || nHeads%nGroups != 0 {
		return Mamba2Config{}, fmt.Errorf("nlp: Mamba2 GGUF ssm.time_step_rank %d (n_head) not divisible by ssm.group_count %d", nHeads, nGroups)
	}
	return Mamba2Config{
		DModel: dModel, NumHeads: nHeads, HeadDim: inner / nHeads,
		NGroups: nGroups, N: n, DConv: dConv, Intermediate: inner,
		Layers: layers,
		Eps:    metaFloat(meta, key(ggufRMSEps), 1e-5),
	}, nil
}

// Mamba2ToGGUF is the inverse of [Mamba2FromGGUF]: it serializes a Mamba2
// (e.g. from [Mamba2FromHF]) into GGUF metadata + tensor maps under
// general.architecture "mamba2", exactly the way llama.cpp's converter lays
// the file out — the fused in_proj kept PACKED under ssm_in.weight, the conv
// kernel squeezed to [conv_dim, d_conv], ssm_a stored as −exp(A_log) and (with
// ssm_d) unsqueezed to [n_head, 1] without a ".weight" suffix, the gated-norm
// gain reshaped to [n_group, d_inner/n_group], ssm_dt carrying ONLY its bias,
// n_head under ssm.time_step_rank and n_groups under ssm.group_count, the
// per-block norm under attn_norm and the epsilon under
// attention.layer_norm_rms_epsilon. output.weight is OMITTED when the head
// equals the embedding transpose (the tied case — llama.cpp re-derives it via
// its token_embd fallback, and a tied HF checkpoint never materializes
// lm_head for the converter to see) and written only for an untied head. The
// converter's placeholder metadata (context_length 2^20, feed_forward_length
// 0, attention.head_count 0) is reproduced so the file loads in llama.cpp.
// Pass the result to gguf.Write via a gguf.File.
func Mamba2ToGGUF(m *Mamba2) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "mamba2"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:               arch,
		key(ggufCtxLen):        uint32(1 << 20), // converter: "arbitrary value; for those who use the default"
		key(ggufEmbLen):        uint32(c.DModel),
		key(ggufBlockCnt):      uint32(c.Layers),
		key(ggufFFLen):         uint32(0), // converter: "unused, but seemingly required when loading"
		key(ggufHeadCnt):       uint32(0),
		key(ggufSSMConvKernel): uint32(c.DConv),
		key(ggufSSMInnerSize):  uint32(c.Intermediate),
		key(ggufSSMStateSize):  uint32(c.N),
		key(ggufSSMDtRank):     uint32(c.NumHeads), // d_inner // head_dim — the converter's n_head slot
		key(ggufSSMGroupCount): uint32(c.NGroups),
		key(ggufRMSEps):        float32(c.Eps),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.Embed),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
	}
	// A tied head is omitted (the converter never sees a tied lm_head;
	// llama.cpp re-derives it from token_embd) — the [MambaToGGUF] pattern.
	if !equalsTransposed(m.Head, m.Embed) {
		ts["output.weight"] = transpose2D(m.Head) // [d_model,vocab] → [vocab,d_model]
	}
	for l := range m.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		mx := m.Layers[l].Mixer
		ts[p+"attn_norm.weight"] = cloneF64(m.Layers[l].Norm.Gamma)
		ts[p+"ssm_in.weight"] = cloneF64(mx.InProj)    // kept packed [z|xBC|dt] — pure rename
		ts[p+"ssm_conv1d.weight"] = cloneF64(mx.ConvW) // stored squeezed [conv_dim, d_conv]
		ts[p+"ssm_conv1d.bias"] = cloneF64(mx.ConvB)
		ts[p+"ssm_dt.bias"] = cloneF64(mx.DtBias)                                    // bias only — no dt_proj weight exists
		ts[p+"ssm_a"] = reshapeF64(negExpF64(mx.ALog), tensor.Shape{mx.NumHeads, 1}) // −exp(A_log), unsqueezed; no ".weight"
		ts[p+"ssm_d"] = reshapeF64(mx.D, tensor.Shape{mx.NumHeads, 1})               // unsqueezed; no ".weight"
		ts[p+"ssm_norm.weight"] = reshapeF64(mx.NormW, tensor.Shape{mx.NGroups, mx.Intermediate / mx.NGroups})
		ts[p+"ssm_out.weight"] = cloneF64(mx.OutProj)
	}
	return meta, ts
}

// reshapeF64 clones t into a fresh contiguous F64 tensor with the given shape
// (same element count, row-major order preserved) — the contiguous-regrouping
// helper for the mamba2 converter's unsqueeze/[n_group, d_inner/n_group]
// reshapes and their inverses. The caller guarantees numel matches (the
// loaders shape-check every tensor first).
func reshapeF64(t *tensor.Tensor, shape tensor.Shape) *tensor.Tensor {
	c := cloneF64(t)
	out := tensor.New(tensor.F64, shape)
	copy(out.Storage().F64(), c.Storage().F64())
	return out
}
