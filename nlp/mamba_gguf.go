package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GGUF SSM metadata keys (gguf-py constants.py Keys.SSM — CONV_KERNEL /
// INNER_SIZE / STATE_SIZE / TIME_STEP_RANK / DT_B_C_RMS, mirrored by
// src/llama-arch.cpp LLM_KV_SSM_*). Arch-prefixed like the rest:
// "mamba.ssm.conv_kernel", "jamba.ssm.inner_size", …
const (
	ggufSSMConvKernel = "ssm.conv_kernel"
	ggufSSMInnerSize  = "ssm.inner_size"
	ggufSSMStateSize  = "ssm.state_size"
	ggufSSMDtRank     = "ssm.time_step_rank"
	ggufSSMDtBCRMS    = "ssm.dt_b_c_rms"
)

// MambaFromGGUF builds a [Mamba] from the metadata and (dequantized) tensor maps
// of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose
// general.architecture is "mamba" — llama.cpp's arch string for the selective
// state-space family (MambaForCausalLM / MambaLMHeadModel). The config comes
// from the mamba.* metadata keys (embedding_length, block_count,
// ssm.conv_kernel, ssm.inner_size, ssm.state_size, ssm.time_step_rank,
// attention.layer_norm_rms_epsilon) and the weights from the token_embd /
// blk.N.ssm_* / output_norm tensors. Every projection is transposed from GGUF's
// torch [out, in] layout into GoAI's [in, out].
//
// The Mamba conventions this loader implements were verified against llama.cpp
// master (conversion/mamba.py MambaModel + src/models/mamba.cpp +
// src/models/mamba-base.cpp build_mamba_layer + gguf-py constants.py /
// tensor_mapping.py + src/llama-arch.cpp):
//
//   - ssm_a stores A = −exp(A_log), NOT A_log: MambaModel.modify_tensors
//     applies `data_torch = -torch.exp(data_torch)` to every *.A_log tensor
//     ("A_log --> A"), and build_mamba_layer feeds layer.ssm_a into
//     ggml_ssm_scan directly (the kernel computes exp(Δ·A) with A already
//     negative). GoAI's nn.MambaBlock holds A_log and forms A = −exp(A_log) in
//     Forward, so this loader INVERTS the converter: ALog = ln(−ssm_a). Every
//     stored element must be strictly negative — a non-negative one has no
//     representable A_log and marks a file from a different convention, so it
//     is rejected rather than NaN-loaded. Loading ssm_a as A_log raw would
//     double-negate/exponentiate and diverge O(1).
//   - ssm_conv1d.weight is stored SQUEEZED: the converter drops HF conv1d's
//     size-1 middle axis ([d_inner, 1, d_conv] → [d_inner, d_conv], the
//     "[4 1 8192 1] -> [4 8192 1 1]" comment in modify_tensors), and
//     src/models/mamba.cpp loads it as ggml {d_conv, d_inner} = row-major
//     [d_inner, d_conv] — exactly nn.MambaBlock's ConvW, copied with NO
//     transpose and NO squeeze here (contrast [MambaFromHF]'s squeezeMid).
//     ssm_conv1d.bias [d_inner] is required (HF use_conv_bias).
//   - Packed x+z input projection: blk.N.ssm_in.weight is HF's in_proj
//     [2·d_inner, d_model] unchanged (pure rename); build_mamba_layer splits
//     the PRODUCT rows [0:d_inner]=x, [d_inner:2·d_inner]=z — so this loader
//     row-splits the weight at d_inner into InX/InZ, like the HF loader.
//   - Packed Δ+B+C projection: blk.N.ssm_x.weight is x_proj
//     [dt_rank+2·d_state, d_inner] unchanged; the graph splits the product
//     [0:dt_rank]=Δ_low, then B, then C — row offsets dt_rank, dt_rank+N.
//   - Δ bias inside softplus: blk.N.ssm_dt.weight [d_inner, dt_rank] and
//     .bias [d_inner]; the graph computes dt = ssm_dt·Δ_low + ssm_dt_b and
//     ggml_ssm_scan applies softplus INTERNALLY (ggml's dt_soft_plus) — the
//     same Δ = softplus(dt_proj·x + bias) GoAI's block computes explicitly.
//   - ssm_a/ssm_d have NO ".weight" suffix (src/llama-arch.cpp maps
//     LLM_TENSOR_SSM_A/…_D to "blk.%d.ssm_a"/"blk.%d.ssm_d"; mamba.cpp notes
//     'no "weight" suffix for these'). ssm_a is ggml {d_state, d_inner} =
//     row-major [d_inner, d_state]; ssm_d is [d_inner]; ssm_out.weight is
//     torch [d_model, d_inner], transposed here.
//   - Per-block norm is blk.N.attn_norm.weight (LLM_TENSOR_ATTN_NORM despite
//     there being no attention — llama.cpp reuses the slot for backbone
//     .layers.N.norm), a standard RMSNorm with the epsilon from
//     mamba.attention.layer_norm_rms_epsilon (the converter maps config.json's
//     layer_norm_epsilon there; default 1e-5).
//   - TIED head with an optional override: output.weight is TENSOR_NOT_REQUIRED
//     in src/models/mamba.cpp with a token_embd fallback, and the converter
//     OMITS lm_head when it torch.equal's the embedding (the MambaModel
//     _tok_embd check) — so an absent output.weight ties Head to Embedᵀ and a
//     present one overrides it.
//   - FalconMamba files are REJECTED: FalconMambaForCausalLM converts to the
//     SAME "mamba" arch string but with mamba.ssm.dt_b_c_rms = true, which
//     makes build_mamba_layer RMS-normalize Δ_low/B/C (weightless build_norm).
//     GoAI's [Mamba] has no such norms, so a true flag is rejected rather than
//     silently un-normalized. ([Jamba]'s weighted dt/b/c norms are dedicated
//     tensors instead — see [JambaFromGGUF].)
//
// Dimensions come from metadata and are cross-checked against every tensor
// (mirroring llama.cpp's fixed create_tensor shapes); Vocab comes from
// token_embd. Metadata the graph never consumes (context_length — the
// converter writes an arbitrary 2^20; the recurrence has no context limit —
// feed_forward_length=0, attention.head_count=0) is ignored.
func MambaFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Mamba, error) {
	cfg, dInner, err := mambaCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if tok.Ndim() != 2 || tok.Shape()[1] != cfg.DModel {
		return nil, fmt.Errorf("nlp: Mamba GGUF token_embd.weight %v does not match embedding_length %d", tok.Shape(), cfg.DModel)
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Mamba{Config: cfg, Embed: cloneF64(tok)}
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
		mixer, err := ssmMixerFromGGUF(tensors, p, cfg.DModel, dInner, cfg.DConv, cfg.N, cfg.DtRank)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, MambaLayer{Norm: rmsFromGGUF(norm, cfg.Eps), Mixer: mixer})
	}

	on, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.Norm = rmsFromGGUF(on, cfg.Eps)
	// Tied head with optional override: absent output.weight → Embedᵀ (llama.cpp's
	// TENSOR_DUPLICATED fallback; the converter omits a torch.equal-tied lm_head).
	if o, ok := tensors["output.weight"]; ok {
		m.Head = transpose2D(o)
	} else {
		m.Head = transpose2D(m.Embed)
	}
	return m, nil
}

// mambaCfgFromGGUFMeta parses the mamba.* metadata keys of a GGUF file whose
// general.architecture is "mamba" into a MambaConfig, returning d_inner
// alongside (Expand = d_inner/DModel must divide evenly). A true
// mamba.ssm.dt_b_c_rms (a FalconMamba file — same arch string, weightless
// Δ/B/C RMS norms in the graph) is rejected here. Vocab is left for
// [MambaFromGGUF] (token_embd-derived).
func mambaCfgFromGGUFMeta(meta map[string]any) (MambaConfig, int, error) {
	const arch = "mamba"
	if a, _ := meta[ggufArch].(string); a != arch {
		return MambaConfig{}, 0, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	if metaBool(meta, key(ggufSSMDtBCRMS), false) {
		return MambaConfig{}, 0, fmt.Errorf("nlp: Mamba GGUF has %s=true (a FalconMamba file with Δ/B/C RMS norms); GoAI's Mamba implements only the classic un-normalized form", key(ggufSSMDtBCRMS))
	}
	dModel, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return MambaConfig{}, 0, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return MambaConfig{}, 0, err
	}
	dConv, err := metaInt(meta, key(ggufSSMConvKernel))
	if err != nil {
		return MambaConfig{}, 0, err
	}
	dInner, err := metaInt(meta, key(ggufSSMInnerSize))
	if err != nil {
		return MambaConfig{}, 0, err
	}
	n, err := metaInt(meta, key(ggufSSMStateSize))
	if err != nil {
		return MambaConfig{}, 0, err
	}
	dtRank, err := metaInt(meta, key(ggufSSMDtRank))
	if err != nil {
		return MambaConfig{}, 0, err
	}
	if dModel <= 0 || dInner <= 0 || dInner%dModel != 0 {
		return MambaConfig{}, 0, fmt.Errorf("nlp: Mamba GGUF ssm.inner_size %d not a multiple of embedding_length %d", dInner, dModel)
	}
	cfg := MambaConfig{
		DModel: dModel, N: n, DConv: dConv, Expand: dInner / dModel,
		DtRank: dtRank, Layers: layers,
		Eps: metaFloat(meta, key(ggufRMSEps), 1e-5),
	}
	return cfg, dInner, nil
}

// ssmMixerFromGGUF loads one block's Mamba-1 selective-scan mixer from the GGUF
// blk-prefix p (tensors ssm_in / ssm_conv1d(+bias) / ssm_x / ssm_dt(+bias) /
// ssm_a / ssm_d / ssm_out) into an nn.MambaBlock, applying the verified
// conventions of [MambaFromGGUF]: packed-row splits at d_inner (x|z) and
// dt_rank/N (Δ|B|C), the pre-squeezed no-transpose conv kernel, and
// ALog = ln(−ssm_a). Shared by [MambaFromGGUF] and [JambaFromGGUF] (whose
// mixer adds the dt/b/c norms on top).
func ssmMixerFromGGUF(tensors map[string]*tensor.Tensor, p string, dModel, dInner, dConv, n, dtRank int) (*nn.MambaBlock, error) {
	g := func(name string) (*tensor.Tensor, error) {
		t, ok := tensors[p+name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
		}
		return t, nil
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
	xProj, err := g("ssm_x.weight")
	if err != nil {
		return nil, err
	}
	dtW, err := g("ssm_dt.weight")
	if err != nil {
		return nil, err
	}
	dtB, err := g("ssm_dt.bias")
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
	outProj, err := g("ssm_out.weight")
	if err != nil {
		return nil, err
	}

	// Mirror llama.cpp's fixed create_tensor shapes — a mismatched tensor would
	// otherwise split at wrong row offsets and misload silently.
	if inProj.Ndim() != 2 || inProj.Shape()[0] != 2*dInner || inProj.Shape()[1] != dModel {
		return nil, fmt.Errorf("nlp: GGUF %sssm_in.weight %v, want [2·d_inner, d_model] = [%d, %d]", p, inProj.Shape(), 2*dInner, dModel)
	}
	if convW.Ndim() != 2 || convW.Shape()[0] != dInner || convW.Shape()[1] != dConv {
		return nil, fmt.Errorf("nlp: GGUF %sssm_conv1d.weight %v, want squeezed [d_inner, d_conv] = [%d, %d] (the converter drops HF's size-1 middle axis)", p, convW.Shape(), dInner, dConv)
	}
	if xProj.Ndim() != 2 || xProj.Shape()[0] != dtRank+2*n || xProj.Shape()[1] != dInner {
		return nil, fmt.Errorf("nlp: GGUF %sssm_x.weight %v, want [dt_rank+2·d_state, d_inner] = [%d, %d]", p, xProj.Shape(), dtRank+2*n, dInner)
	}
	if dtW.Ndim() != 2 || dtW.Shape()[0] != dInner || dtW.Shape()[1] != dtRank {
		return nil, fmt.Errorf("nlp: GGUF %sssm_dt.weight %v, want [d_inner, dt_rank] = [%d, %d]", p, dtW.Shape(), dInner, dtRank)
	}
	if a.Ndim() != 2 || a.Shape()[0] != dInner || a.Shape()[1] != n {
		return nil, fmt.Errorf("nlp: GGUF %sssm_a %v, want [d_inner, d_state] = [%d, %d]", p, a.Shape(), dInner, n)
	}
	aLog, err := alogFromGGUFA(a, p)
	if err != nil {
		return nil, err
	}

	// ssm_in product rows [0:d_inner]=x, [d_inner:2·d_inner]=z (build_mamba_layer's
	// xz views); ssm_x product rows [0:dt_rank]=Δ_low, then B, then C.
	inX := sliceRows(inProj, 0, dInner)
	inZ := sliceRows(inProj, dInner, 2*dInner)
	dtRows := sliceRows(xProj, 0, dtRank)
	bRows := sliceRows(xProj, dtRank, dtRank+n)
	cRows := sliceRows(xProj, dtRank+n, dtRank+2*n)

	return &nn.MambaBlock{
		InX:     &nn.Linear{W: transpose2D(inX)},                   // [d_model, d_inner]
		InZ:     &nn.Linear{W: transpose2D(inZ)},                   // [d_model, d_inner]
		ConvW:   cloneF64(convW),                                   // [d_inner, d_conv], stored squeezed — no transpose
		ConvB:   cloneF64(convB),                                   // [d_inner]
		DtLow:   &nn.Linear{W: transpose2D(dtRows)},                // [d_inner, dt_rank]
		BProj:   &nn.Linear{W: transpose2D(bRows)},                 // [d_inner, d_state]
		CProj:   &nn.Linear{W: transpose2D(cRows)},                 // [d_inner, d_state]
		DtProj:  &nn.Linear{W: transpose2D(dtW), B: cloneF64(dtB)}, // [dt_rank, d_inner] + Δ bias
		ALog:    aLog,                                              // ln(−ssm_a) — the converter's A = −exp(A_log) inverted
		Dskip:   cloneF64(d),                                       // [d_inner]
		OutProj: &nn.Linear{W: transpose2D(outProj)},               // [d_inner, d_model]
		DModel:  dModel, DInner: dInner, DConv: dConv, N: n, DtRank: dtRank,
	}, nil
}

// alogFromGGUFA inverts llama.cpp's converter transform on the SSM state
// matrix: GGUF stores ssm_a = −exp(A_log) (conversion/mamba.py & jamba.py
// modify_tensors), so A_log = ln(−a). Every element must be strictly negative;
// a non-negative element has no real A_log and marks a file that does not
// follow the mamba/jamba convention, so the load fails instead of storing NaN.
func alogFromGGUFA(a *tensor.Tensor, p string) (*tensor.Tensor, error) {
	af := cloneF64(a)
	xs := af.Storage().F64()
	for i, v := range xs {
		if v >= 0 {
			return nil, fmt.Errorf("nlp: GGUF %sssm_a element %d is %g ≥ 0; the mamba/jamba convention stores A = −exp(A_log), which is strictly negative", p, i, v)
		}
		xs[i] = math.Log(-v)
	}
	return af, nil
}

// negExpF64 is the converter-side transform ln(−a) inverts: −exp(aLog),
// elementwise, into a fresh F64 tensor — what [MambaToGGUF] / [JambaToGGUF]
// store as ssm_a (conversion/mamba.py "A_log --> A").
func negExpF64(aLog *tensor.Tensor) *tensor.Tensor {
	out := cloneF64(aLog)
	xs := out.Storage().F64()
	for i, v := range xs {
		xs[i] = -math.Exp(v)
	}
	return out
}

// ssmMixerToGGUF is the inverse of [ssmMixerFromGGUF]: it serializes one
// nn.MambaBlock into the blk-prefix p using llama.cpp's on-disk conventions —
// re-packed ssm_in (x rows above z) and ssm_x (Δ_low|B|C), the squeezed
// no-transpose conv kernel, ssm_a = −exp(A_log), and ssm_a/ssm_d without the
// ".weight" suffix. Shared by [MambaToGGUF] and [JambaToGGUF].
func ssmMixerToGGUF(ts map[string]*tensor.Tensor, p string, b *nn.MambaBlock) {
	ts[p+"ssm_in.weight"] = concatRows(transpose2D(b.InX.W), transpose2D(b.InZ.W)) // [2·d_inner, d_model]
	ts[p+"ssm_conv1d.weight"] = cloneF64(b.ConvW)                                  // stored squeezed [d_inner, d_conv]
	ts[p+"ssm_conv1d.bias"] = cloneF64(b.ConvB)
	ts[p+"ssm_x.weight"] = concatRows(concatRows(transpose2D(b.DtLow.W), transpose2D(b.BProj.W)), transpose2D(b.CProj.W))
	ts[p+"ssm_dt.weight"] = transpose2D(b.DtProj.W)
	ts[p+"ssm_dt.bias"] = cloneF64(b.DtProj.B)
	ts[p+"ssm_a"] = negExpF64(b.ALog) // the converter's A = −exp(A_log); no ".weight" suffix
	ts[p+"ssm_d"] = cloneF64(b.Dskip)
	ts[p+"ssm_out.weight"] = transpose2D(b.OutProj.W)
}

// MambaToGGUF is the inverse of [MambaFromGGUF]: it serializes a Mamba (e.g.
// from [MambaFromHF]) into GGUF metadata + tensor maps under
// general.architecture "mamba", exactly the way llama.cpp's converter lays the
// file out — ssm_a stored as −exp(A_log), the conv kernel squeezed to
// [d_inner, d_conv], the packed ssm_in/ssm_x projections in torch [out, in]
// layout, ssm_a/ssm_d without ".weight", the per-block norm under attn_norm,
// the epsilon under attention.layer_norm_rms_epsilon and ssm.dt_b_c_rms=false
// (the classic, non-FalconMamba form). Mirroring MambaModel's _tok_embd check,
// output.weight is OMITTED when the head equals the embedding transpose (the
// tied case — llama.cpp re-derives it) and written only for an untied head.
// The converter's placeholder metadata (context_length 2^20,
// feed_forward_length 0, attention.head_count 0) is reproduced so the file
// loads in llama.cpp. Pass the result to gguf.Write via a gguf.File.
func MambaToGGUF(m *Mamba) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "mamba"
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
		key(ggufSSMInnerSize):  uint32(c.Expand * c.DModel),
		key(ggufSSMStateSize):  uint32(c.N),
		key(ggufSSMDtRank):     uint32(c.DtRank),
		key(ggufRMSEps):        float32(c.Eps),
		key(ggufSSMDtBCRMS):    false,
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.Embed),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
	}
	// MambaModel omits lm_head when torch.equal to the embedding; replicate so a
	// tied model writes the converter's exact tensor set.
	if !equalsTransposed(m.Head, m.Embed) {
		ts["output.weight"] = transpose2D(m.Head) // [d_model,vocab] → [vocab,d_model]
	}
	for l := range m.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(m.Layers[l].Norm.Gamma)
		ssmMixerToGGUF(ts, p, m.Layers[l].Mixer)
	}
	return meta, ts
}

// equalsTransposed reports whether head [d, v] is exactly the transpose of
// emb [v, d] — the tied-LM-head check MambaToGGUF uses to mirror the
// converter's torch.equal omission.
func equalsTransposed(head, emb *tensor.Tensor) bool {
	if head.Ndim() != 2 || emb.Ndim() != 2 ||
		head.Shape()[0] != emb.Shape()[1] || head.Shape()[1] != emb.Shape()[0] {
		return false
	}
	v, d := emb.Shape()[0], emb.Shape()[1]
	for i := range v {
		for j := range d {
			if emb.AtF64(i, j) != head.AtF64(j, i) {
				return false
			}
		}
	}
	return true
}
