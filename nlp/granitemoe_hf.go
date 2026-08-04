package nlp

import (
	"encoding/json"
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GraniteMoeConfigFromHF parses a Hugging Face GraniteMoE config.json into a
// [GraniteMoeConfig] for [GraniteMoeFromHF] — the non-tensor-derivable Heads, KVHeads,
// TopK, Eps, RopeBase, Ctx PLUS Granite's four scalar multipliers (embedding_multiplier,
// attention_multiplier, residual_multiplier, logits_scaling). Dim, Vocab, Layers, Hidden
// and Experts are inferred from the weights. Missing rms_norm_eps / rope_theta /
// num_experts_per_tok fall back to (1e-5, 10000, 2); an absent multiplier stays 0, which
// [GraniteMoeConfig] treats as the identity (so it degrades to plain Mixtral-style
// behaviour rather than erroring).
func GraniteMoeConfigFromHF(configJSON []byte) (GraniteMoeConfig, error) {
	var j struct {
		NumAttentionHeads int     `json:"num_attention_heads"`
		NumKeyValueHeads  *int    `json:"num_key_value_heads"`
		NumExpertsPerTok  int     `json:"num_experts_per_tok"`
		RMSNormEps        float64 `json:"rms_norm_eps"`
		RopeTheta         float64 `json:"rope_theta"`
		MaxPos            int     `json:"max_position_embeddings"`
		EmbeddingMult     float64 `json:"embedding_multiplier"`
		AttentionMult     float64 `json:"attention_multiplier"`
		ResidualMult      float64 `json:"residual_multiplier"`
		LogitsScaling     float64 `json:"logits_scaling"`
	}
	if err := json.Unmarshal(configJSON, &j); err != nil {
		return GraniteMoeConfig{}, fmt.Errorf("nlp: parse GraniteMoE config.json: %w", err)
	}
	if j.NumAttentionHeads <= 0 {
		return GraniteMoeConfig{}, fmt.Errorf("nlp: config.json missing num_attention_heads")
	}
	kv := j.NumAttentionHeads
	if j.NumKeyValueHeads != nil {
		kv = *j.NumKeyValueHeads
	}
	eps := j.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	topK := j.NumExpertsPerTok
	if topK <= 0 {
		topK = 2
	}
	return GraniteMoeConfig{
		Heads:         j.NumAttentionHeads,
		KVHeads:       kv,
		TopK:          topK,
		Eps:           eps,
		RopeBase:      j.RopeTheta, // 0 → GraniteMoeFromHF defaults to 10000
		Ctx:           j.MaxPos,
		EmbeddingMult: j.EmbeddingMult, // 0 → identity (no embedding scaling)
		AttentionMult: j.AttentionMult, // 0 → default 1/√headDim attention scale
		ResidualMult:  j.ResidualMult,  // 0 → identity (unscaled residual adds)
		LogitsScale:   j.LogitsScaling, // 0 → identity (unscaled logits)
	}, nil
}

// GraniteMoeFromHF builds a [GraniteMoE] from a Hugging Face checkpoint (transformers'
// GraniteMoeForCausalLM). The attention weights load exactly as [LlamaFromHF] /
// [MixtralFromHF] do (bias-free, no QK-norm); the per-block MoE loads into nn.SparseMoE
// from the SAME row-packed fused layout Mixtral uses, only under a `block_sparse_moe.*`
// prefix: the router is `block_sparse_moe.router.weight` [E, dim], and the experts are
// FUSED as `block_sparse_moe.experts.gate_up_proj` [E, 2·ffn, dim] and
// `block_sparse_moe.experts.down_proj` [E, dim, ffn]. Each expert e unpacks as
// gate = gate_up[e] rows [0,ffn), up = rows [ffn,2·ffn), down = down[e]; all transposed
// from torch [out,in] to GoAI [in,out]. There is NO shared expert (that is the separate
// granitemoeshared variant). The router does softmax over only the top-k logits, which is
// exactly nn.SparseMoE's renormalized top-k, so routing is bit-identical.
//
// cfg supplies Heads, KVHeads, TopK, Eps, RopeBase, Ctx and the four Granite scalar
// multipliers; Dim, Vocab, Layers, Hidden and Experts are inferred from the weights.
func GraniteMoeFromHF(ts map[string]*tensor.Tensor, cfg GraniteMoeConfig) (*GraniteMoE, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF GraniteMoE missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: GraniteMoeFromHF needs cfg.Heads")
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
		return nil, fmt.Errorf("nlp: HF GraniteMoE has no model.layers.*")
	}
	cfg.Layers = layers

	m := &GraniteMoE{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 per-layer loader loop; cold
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF GraniteMoE missing %s%s", p, name)
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
		an, err := g("input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		fn, err := g("post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		moe, err := graniteMoeExperts(ts, p, cfg)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &GraniteMoeBlock{
			AttnNorm: rmsFromGGUF(an, cfg.Eps),
			Wq:       transpose2D(wq),
			Wk:       transpose2D(wk),
			Wv:       transpose2D(wv),
			Wo:       transpose2D(wo),
			FFNNorm:  rmsFromGGUF(fn, cfg.Eps),
			MoE:      moe,
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF GraniteMoE missing model.norm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head := tok
	if o, ok := ts["lm_head.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head)
	return m, nil
}

// graniteMoeExperts loads one block's sparse-MoE (router + fused experts) into an
// nn.SparseMoE. The layout matches Mixtral's fused experts (see [MixtralFromHF]) but the
// keys live under `block_sparse_moe.*`: router `block_sparse_moe.router.weight` [E, dim],
// experts `block_sparse_moe.experts.gate_up_proj` [E, 2·ffn, dim] and `.down_proj`
// [E, dim, ffn].
func graniteMoeExperts(ts map[string]*tensor.Tensor, p string, cfg GraniteMoeConfig) (*nn.SparseMoE, error) {
	router, ok := ts[p+"block_sparse_moe.router.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF GraniteMoE missing %sblock_sparse_moe.router.weight", p)
	}
	experts := router.Shape()[0] // [E, dim]

	gu, okGU := ts[p+"block_sparse_moe.experts.gate_up_proj"]
	dn, okDN := ts[p+"block_sparse_moe.experts.down_proj"]
	if !okGU || !okDN {
		return nil, fmt.Errorf("nlp: HF GraniteMoE %sblock_sparse_moe.experts.{gate_up_proj,down_proj} not found", p)
	}
	if gu.Ndim() != 3 || dn.Ndim() != 3 {
		return nil, fmt.Errorf("nlp: GraniteMoE fused experts must be 3-D, got %v / %v", gu.Shape(), dn.Shape())
	}
	ffn := gu.Shape()[1] / 2 // [E, 2·ffn, dim]

	moe := nn.NewSparseMoE(tensor.F64, cfg.Dim, ffn, experts, cfg.TopK, 0)
	moe.Router.W = transpose2D(router) // [E,dim] → [dim,E]
	//perfscan:ignore PS3060 per-expert MoE loader loop; cold
	for e := range experts {
		guE := sub3D(gu, e)                                          // [2·ffn, dim]
		moe.Experts[e].Wgate = transpose2D(sliceRows(guE, 0, ffn))   // [dim, ffn]
		moe.Experts[e].Wup = transpose2D(sliceRows(guE, ffn, 2*ffn)) // [dim, ffn]
		moe.Experts[e].Wdown = transpose2D(sub3D(dn, e))             // [dim, ffn] → [ffn, dim]
	}
	return moe, nil
}
