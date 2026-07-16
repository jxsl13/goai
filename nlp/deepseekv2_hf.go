package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// DeepSeekV2FromHF builds a [DeepSeekV2] from a Hugging Face DeepseekV2ForCausalLM tensor
// map (the dense-FFN variant — first_k_dense_replace = num_layers, so every layer's MLP is
// a plain SwiGLU). The MLA geometry that is not derivable from the tensors (Heads,
// QLoraRank, KVLoraRank, QKNope, QKRope, VHead, Eps, RopeBase, Ctx, SoftmaxScale) comes
// from config.json and is passed in cfg; Vocab, Dim, Layers and FFN are inferred from the
// tensors. HF stores each projection as torch [out, in]; GoAI wants [in, out], so every
// weight is transposed.
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
			FFN:          swiGLUFromGGUF(gate, up, down),
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
	// Copy every row as-is first; the pe rows are then overwritten in permuted order.
	for r := range out {
		for c := range in {
			res.SetF64(w.AtF64(r, c), r, c)
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
			for c := range in {
				res.SetF64(w.AtF64(src0, c), dst0, c)
				res.SetF64(w.AtF64(src1, c), dst1, c)
			}
		}
	}
	return res
}
