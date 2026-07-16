package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Qwen2MoeFromHF builds a [Qwen2MoE] from a Hugging Face checkpoint (transformers'
// Qwen2MoeForCausalLM). Layout, all confirmed against the golden state_dict:
//
//   - Attention (Qwen2): self_attn.{q,k,v,o}_proj.weight [out,in] transposed to [in,out];
//     q/k/v carry a bias (self_attn.{q,k,v}_proj.bias), o_proj does not. No QK-norm.
//
//   - Sparse MoE: mlp.gate.weight [E,dim] router and the FUSED expert tensors
//     mlp.experts.gate_up_proj [E, 2·moe_inter, dim] + mlp.experts.down_proj
//     [E, dim, moe_inter] — byte-for-byte the Mixtral fused layout — loaded via the shared
//     mixtralMoE helper into an nn.SparseMoE.
//
//   - Shared expert (the Qwen2-MoE addition): mlp.shared_expert.{gate,up,down}_proj.weight
//     (a SwiGLU of shared_expert_intermediate_size width) and the bias-free gate
//     mlp.shared_expert_gate.weight [1,dim], transposed to [dim,1].
//
// cfg supplies Heads, KVHeads, TopK, Eps, RopeBase, Ctx; Vocab, Dim, Layers, Hidden
// (moe_intermediate_size), SharedHidden (shared_expert_intermediate_size) and Experts are
// inferred from the checkpoint. This loader assumes every layer is a MoE layer
// (decoder_sparse_step=1, mlp_only_layers=[], the Qwen2-MoE default); a checkpoint with dense
// layers (a plain Qwen2MLP under model.layers.N.mlp) is not supported.
//
// Limitation: the sparse combine always renormalizes the top-k router weights (nn.SparseMoE,
// matching transformers with norm_topk_prob=True). A norm_topk_prob=False checkpoint would
// route identically but weight the experts without renormalization and so would NOT match.
func Qwen2MoeFromHF(ts map[string]*tensor.Tensor, cfg Qwen2MoeConfig) (*Qwen2MoE, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Qwen2MoE missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: Qwen2MoeFromHF needs cfg.Heads")
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 2
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Qwen2MoE has no model.layers.*")
	}
	cfg.Layers = layers

	m := &Qwen2MoE{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Qwen2MoE missing %s%s", p, name)
			}
			return t, nil
		}
		wq, err := g("self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		wk, err := g("self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		wv, err := g("self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		wo, err := g("self_attn.o_proj.weight")
		if err != nil {
			return nil, err
		}
		bq, err := g("self_attn.q_proj.bias")
		if err != nil {
			return nil, err
		}
		bk, err := g("self_attn.k_proj.bias")
		if err != nil {
			return nil, err
		}
		bv, err := g("self_attn.v_proj.bias")
		if err != nil {
			return nil, err
		}
		an, err := g("input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		fn, err := g("post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		// Sparse routed experts: reuse the Mixtral fused loader (identical layout).
		moe, err := mixtralMoE(ts, p, MixtralConfig{Dim: cfg.Dim, TopK: cfg.TopK})
		if err != nil {
			return nil, err
		}
		// Shared expert (SwiGLU) + its bias-free sigmoid gate.
		shared, gate, err := qwen2SharedExpert(ts, p)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &Qwen2MoeBlock{
			AttnNorm:   rmsFromGGUF(an, cfg.Eps),
			Wq:         transpose2D(wq),
			Wk:         transpose2D(wk),
			Wv:         transpose2D(wv),
			Wo:         transpose2D(wo),
			Bq:         cloneF64(bq),
			Bk:         cloneF64(bk),
			Bv:         cloneF64(bv),
			FFNNorm:    rmsFromGGUF(fn, cfg.Eps),
			MoE:        moe,
			Shared:     shared,
			SharedGate: gate,
		})
	}
	if l0 := m.Blocks[0]; len(l0.MoE.Experts) > 0 {
		cfg.Experts = len(l0.MoE.Experts)
		cfg.Hidden = l0.MoE.Experts[0].Wgate.Shape()[1]
		cfg.SharedHidden = l0.Shared.Wgate.Shape()[1]
		m.Config = cfg
	}

	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Qwen2MoE missing model.norm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head := tok
	if o, ok := ts["lm_head.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head)
	return m, nil
}

// qwen2SharedExpert loads one block's shared expert: the SwiGLU
// mlp.shared_expert.{gate,up,down}_proj and the bias-free gate mlp.shared_expert_gate.weight
// [1,dim] transposed to [dim,1]. All projections are torch [out,in] → GoAI [in,out].
func qwen2SharedExpert(ts map[string]*tensor.Tensor, p string) (*nn.SwiGLU, *tensor.Tensor, error) {
	get := func(name string) (*tensor.Tensor, error) {
		t, ok := ts[p+name]
		if !ok {
			return nil, fmt.Errorf("nlp: HF Qwen2MoE missing %s%s", p, name)
		}
		return t, nil
	}
	gp, err := get("mlp.shared_expert.gate_proj.weight")
	if err != nil {
		return nil, nil, err
	}
	up, err := get("mlp.shared_expert.up_proj.weight")
	if err != nil {
		return nil, nil, err
	}
	dn, err := get("mlp.shared_expert.down_proj.weight")
	if err != nil {
		return nil, nil, err
	}
	gate, err := get("mlp.shared_expert_gate.weight")
	if err != nil {
		return nil, nil, err
	}
	if gate.Ndim() != 2 || gate.Shape()[0] != 1 {
		return nil, nil, fmt.Errorf("nlp: Qwen2MoE shared_expert_gate must be [1,dim], got %v", gate.Shape())
	}
	shared := &nn.SwiGLU{
		Wgate: transpose2D(gp), // [dim, shared_inter]
		Wup:   transpose2D(up), // [dim, shared_inter]
		Wdown: transpose2D(dn), // [shared_inter, dim]
	}
	return shared, transpose2D(gate), nil // gate [1,dim] → [dim,1]
}
