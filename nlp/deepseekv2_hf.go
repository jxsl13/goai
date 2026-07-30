package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// DeepSeekV2FromHF builds a [DeepSeekV2] from a Hugging Face DeepseekV2ForCausalLM tensor
// map, in either FFN configuration: the first cfg.FirstKDense layers load a dense SwiGLU
// (mlp.gate_proj/up_proj/down_proj) and every later layer loads a DeepSeekMoE (see
// [loadDeepSeekMoE]). With FirstKDense ≥ num_layers (or NRoutedExperts = 0) every layer is
// dense — the MLA-only variant. The MLA geometry that is not derivable from the tensors
// (Heads, QLoraRank, KVLoraRank, QKNope, QKRope, VHead, Eps, RopeBase, Ctx, SoftmaxScale)
// and the MoE geometry (FirstKDense, NRoutedExperts, NSharedExperts, TopK, MoEHidden,
// RoutedScale) come from config.json and are passed in cfg; Vocab, Dim, Layers and the
// dense FFN width are inferred from the tensors. HF stores each projection as torch
// [out, in]; GoAI wants [in, out], so every weight is transposed.
//
// DeepSeek-V2 rotates its decoupled q_pe/k_pe with the interleaved (view_as_complex)
// convention. To reproduce that with GoAI's split-half OpRoPE, the pe rows of q_b_proj (per
// head) and the k_pe rows of kv_a_proj_with_mqa are de-interleaved at load (see
// [deinterleaveRoPE]); every other weight is transposed as-is.
func DeepSeekV2FromHF(ts map[string]*tensor.Tensor, cfg DeepSeekV2Config) (*DeepSeekV2, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF DeepSeekV2 missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: DeepSeekV2FromHF needs cfg.Heads")
	}
	if cfg.QLoraRank <= 0 || cfg.KVLoraRank <= 0 || cfg.QKNope <= 0 || cfg.QKRope <= 0 || cfg.VHead <= 0 {
		return nil, fmt.Errorf("nlp: DeepSeekV2FromHF needs positive QLoraRank/KVLoraRank/QKNope/QKRope/VHead")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_a_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF DeepSeekV2 has no model.layers.* (expected MLA q_a_proj)")
	}
	cfg.Layers = layers
	if gate, ok := ts["model.layers.0.mlp.gate_proj.weight"]; ok {
		cfg.FFN = gate.Shape()[0]
	}

	m := &DeepSeekV2{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF DeepSeekV2 missing %s%s", p, name)
			}
			return t, nil
		}
		wqA, err := g("self_attn.q_a_proj.weight")
		if err != nil {
			return nil, err
		}
		qaNorm, err := g("self_attn.q_a_layernorm.weight")
		if err != nil {
			return nil, err
		}
		wqB, err := g("self_attn.q_b_proj.weight")
		if err != nil {
			return nil, err
		}
		wkvA, err := g("self_attn.kv_a_proj_with_mqa.weight")
		if err != nil {
			return nil, err
		}
		kvaNorm, err := g("self_attn.kv_a_layernorm.weight")
		if err != nil {
			return nil, err
		}
		wkvB, err := g("self_attn.kv_b_proj.weight")
		if err != nil {
			return nil, err
		}
		wo, err := g("self_attn.o_proj.weight")
		if err != nil {
			return nil, err
		}
		inNorm, err := g("input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postAttn, err := g("post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		// FFN sublayer: dense SwiGLU for the first FirstKDense layers, else a DeepSeekMoE.
		var dense *nn.SwiGLU
		var moe *nn.DeepSeekMoE
		if l < cfg.FirstKDense || cfg.NRoutedExperts <= 0 {
			gate, err := g("mlp.gate_proj.weight")
			if err != nil {
				return nil, err
			}
			up, err := g("mlp.up_proj.weight")
			if err != nil {
				return nil, err
			}
			down, err := g("mlp.down_proj.weight")
			if err != nil {
				return nil, err
			}
			dense = swiGLUFromGGUF(gate, up, down)
		} else {
			if moe, err = loadDeepSeekMoE(ts, p+"mlp.", cfg); err != nil {
				return nil, err
			}
		}
		// De-interleave the pe channels so split-half OpRoPE reproduces DeepSeek's
		// interleaved rotary. q_b_proj: heads blocks of [QKNope | QKRope]; permute the
		// QKRope rows of each head. kv_a_proj_with_mqa: one block of [KVLoraRank | QKRope];
		// permute the trailing QKRope (k_pe) rows.
		qbPerm := deinterleaveRoPE(wqB, cfg.Heads, cfg.QKNope+cfg.QKRope, cfg.QKNope, cfg.QKRope)
		kvaPerm := deinterleaveRoPE(wkvA, 1, cfg.KVLoraRank+cfg.QKRope, cfg.KVLoraRank, cfg.QKRope)

		m.Blocks = append(m.Blocks, &DeepSeekV2Block{
			InputNorm:    rmsFromGGUF(inNorm, cfg.Eps),
			WqA:          transpose2D(wqA),
			QANorm:       rmsFromGGUF(qaNorm, cfg.Eps),
			WqB:          transpose2D(qbPerm),
			WkvA:         transpose2D(kvaPerm),
			KvANorm:      rmsFromGGUF(kvaNorm, cfg.Eps),
			WkvB:         transpose2D(wkvB),
			Wo:           transpose2D(wo),
			PostAttnNorm: rmsFromGGUF(postAttn, cfg.Eps),
			Dense:        dense,
			MoE:          moe,
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF DeepSeekV2 missing model.norm.weight")
	}
	m.FinalNorm = rmsFromGGUF(norm, cfg.Eps)
	head, ok := ts["lm_head.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF DeepSeekV2 missing lm_head.weight (untied)")
	}
	m.LmHead = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// loadDeepSeekMoE builds an [nn.DeepSeekMoE] from a HF DeepseekV2Moe layer's tensors under
// prefix (e.g. "model.layers.1.mlp."). It captures transformers 5.x's FUSED expert layout:
//
//   - Routed experts: mlp.experts.gate_up_proj [E, 2·MoEHidden, dim] and
//     mlp.experts.down_proj [E, dim, MoEHidden] — a single 3-D tensor for the whole bank,
//     NOT per-expert mlp.experts.J.{gate,up,down}_proj. Expert j's gate_up_proj[j] rows split
//     [0:MoEHidden]=gate_proj, [MoEHidden:2·MoEHidden]=up_proj (matching torch's
//     linear(x, gate_up_proj[j]).chunk(2)); each expert becomes one nn.SwiGLU.
//   - Shared experts: mlp.shared_experts.{gate,up,down}_proj.weight — a SINGLE SwiGLU of
//     width NSharedExperts·MoEHidden (transformers fuses the n_shared experts into one MLP).
//   - Router: mlp.gate.weight [E, dim], no bias.
//
// The weights are held in an nn.DeepSeekMoE; the mixture itself is combined by
// [DeepSeekV2.moeFFN] (raw-softmax top-k, no renormalization) rather than the type's own
// renormalizing Forward — see moeFFN for why.
func loadDeepSeekMoE(ts map[string]*tensor.Tensor, prefix string, cfg DeepSeekV2Config) (*nn.DeepSeekMoE, error) {
	if cfg.TopK <= 0 || cfg.TopK > cfg.NRoutedExperts || cfg.MoEHidden <= 0 {
		return nil, fmt.Errorf("nlp: DeepSeekV2 MoE needs 0 < TopK ≤ NRoutedExperts and MoEHidden > 0 (got %d, %d, %d)", cfg.TopK, cfg.NRoutedExperts, cfg.MoEHidden)
	}
	get := func(name string) (*tensor.Tensor, error) {
		t, ok := ts[prefix+name]
		if !ok {
			return nil, fmt.Errorf("nlp: HF DeepSeekV2 MoE missing %s%s", prefix, name)
		}
		return t, nil
	}
	// Router: gate.weight [E, dim] → GoAI Linear.W [dim, E], no bias.
	gate, err := get("gate.weight")
	if err != nil {
		return nil, err
	}

	// Routed experts: unfuse the 3-D gate_up_proj / down_proj into E per-expert SwiGLUs.
	gateUp, err := get("experts.gate_up_proj") // [E, 2·MoEHidden, dim]
	if err != nil {
		return nil, err
	}
	down, err := get("experts.down_proj") // [E, dim, MoEHidden]
	if err != nil {
		return nil, err
	}
	e, inter := cfg.NRoutedExperts, cfg.MoEHidden
	if gateUp.Ndim() != 3 || gateUp.Shape()[0] != e || gateUp.Shape()[1] != 2*inter {
		return nil, fmt.Errorf("nlp: experts.gate_up_proj shape %v != [%d,%d,dim]", gateUp.Shape(), e, 2*inter)
	}
	if down.Ndim() != 3 || down.Shape()[0] != e || down.Shape()[2] != inter {
		return nil, fmt.Errorf("nlp: experts.down_proj shape %v != [%d,dim,%d]", down.Shape(), e, inter)
	}
	experts := make([]*nn.SwiGLU, e)
	for j := range e {
		gu, err := slice3D(gateUp, j) // [2·MoEHidden, dim] (torch [out,in])
		if err != nil {
			return nil, err
		}
		gj, err := gu.Slice(0, 0, inter) // gate_proj rows
		if err != nil {
			return nil, err
		}
		uj, err := gu.Slice(0, inter, 2*inter) // up_proj rows
		if err != nil {
			return nil, err
		}
		dj, err := slice3D(down, j) // [dim, MoEHidden] (torch [out,in])
		if err != nil {
			return nil, err
		}
		experts[j] = swiGLUFromGGUF(gj, uj, dj) // transposes each into GoAI [in,out]
	}

	// Shared experts: one fused SwiGLU of width NSharedExperts·MoEHidden.
	sg, err := get("shared_experts.gate_proj.weight")
	if err != nil {
		return nil, err
	}
	su, err := get("shared_experts.up_proj.weight")
	if err != nil {
		return nil, err
	}
	sd, err := get("shared_experts.down_proj.weight")
	if err != nil {
		return nil, err
	}

	return &nn.DeepSeekMoE{
		Shared: []*nn.SwiGLU{swiGLUFromGGUF(sg, su, sd)},
		Routed: &nn.SparseMoE{
			Router:  &nn.Linear{W: transpose2D(gate)}, // no bias (DeepseekV2TopkRouter)
			Experts: experts,
			TopK:    cfg.TopK,
		},
	}, nil
}

// slice3D extracts index j along axis 0 of a 3-D tensor as a contiguous 2-D [d1,d2] tensor.
func slice3D(t *tensor.Tensor, j int) (*tensor.Tensor, error) {
	v, err := t.Slice(0, j, j+1) // [1, d1, d2]
	if err != nil {
		return nil, err
	}
	return v.Contiguous().Reshape(tensor.Shape{t.Shape()[1], t.Shape()[2]})
}

// deinterleaveRoPE reorders the ROWS of a torch [out, in] projection weight so that GoAI's
// split-half OpRoPE reproduces DeepSeek-V2's interleaved (view_as_complex) rotary on the pe
// sub-block. The output rows are grouped into `heads` blocks of `block` rows each; within
// each block the `ropeDim` rows starting at `peOffset` are regrouped from interleaved
// (0,1,2,3,…) into split-half order (evens first, then odds): the row at offset peOffset+i
// takes the original peOffset+2i, and peOffset+(i+ropeDim/2) takes peOffset+2i+1, for i in
// [0,ropeDim/2). All other rows are copied unchanged. A fresh [out, in] tensor is returned;
// the input is left untouched. ropeDim must be even.
func deinterleaveRoPE(w *tensor.Tensor, heads, block, peOffset, ropeDim int) *tensor.Tensor {
	out, in := w.Shape()[0], w.Shape()[1]
	res := tensor.New(tensor.F64, tensor.Shape{out, in})
	// res is constructed F64 right here, so its storage can be written directly and no dtype
	// fallback is needed or wanted (PERF-NO-FALLBACK-FOR-A-FIXED-DTYPE-001). That removes one of the
	// two interface dispatches per element — out*in of them in this copy alone, 4.2M at [4096,1024].
	// w keeps its accessor: its dtype comes from the file being loaded, so a typed read there would
	// need a dual path and a test reaching both arms.
	rf := res.Storage().F64()
	// Copy every row as-is first; the pe rows are then overwritten in permuted order.
	for r := range out {
		row := rf[r*in : (r+1)*in]
		for c := range in {
			row[c] = w.AtF64(r, c)
		}
	}
	half := ropeDim / 2
	for h := range heads {
		base := h*block + peOffset
		for i := range half {
			src0 := base + 2*i      // interleaved even channel
			src1 := base + 2*i + 1  // interleaved odd channel
			dst0 := base + i        // split-half: first half
			dst1 := base + i + half // split-half: second half
			r0 := rf[dst0*in : (dst0+1)*in]
			r1 := rf[dst1*in : (dst1+1)*in]
			for c := range in {
				r0[c] = w.AtF64(src0, c)
				r1[c] = w.AtF64(src1, c)
			}
		}
	}
	return res
}
