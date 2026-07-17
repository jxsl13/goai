package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MixtralFromHF builds a [Mixtral] from a Hugging Face checkpoint (transformers'
// MixtralForCausalLM). The attention weights load exactly as [LlamaFromHF] does;
// the per-block MoE loads into nn.SparseMoE. Recent transformers store the experts
// FUSED — `mlp.experts.gate_up_proj` [E, 2·ffn, dim] and `mlp.experts.down_proj`
// [E, dim, ffn] — with per-expert gate/up packing exactly like Phi-3, so each
// expert e is unpacked as: gate = gate_up[e] rows [0,ffn), up = rows [ffn,2·ffn),
// down = down[e]; all transposed from torch [out,in] to GoAI [in,out]. The router
// is `mlp.gate.weight` [E, dim].
//
// cfg supplies Heads, KVHeads, TopK, Eps, RopeBase, Ctx; Dim, Vocab, Layers,
// Hidden and Experts are inferred. If the checkpoint carries per-head QK-norm gains
// (`self_attn.{q,k}_norm.weight`, present in Qwen3-MoE, absent in plain Mixtral) they
// load into each block's QNorm/KNorm; see [Qwen3MoeFromHF].
func MixtralFromHF(ts map[string]*tensor.Tensor, cfg MixtralConfig) (*Mixtral, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Mixtral missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: MixtralFromHF needs cfg.Heads")
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
		return nil, fmt.Errorf("nlp: HF Mixtral has no model.layers.*")
	}
	cfg.Layers = layers

	m := &Mixtral{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Mixtral missing %s%s", p, name)
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
		moe, err := mixtralMoE(ts, p, cfg)
		if err != nil {
			return nil, err
		}
		// Optional Qwen3-MoE per-head QK-norm gains (absent in plain Mixtral → nil, no-op).
		qkNorm := func(name string) *nn.RMSNorm {
			if t, ok := ts[p+name]; ok {
				return rmsFromGGUF(t, cfg.Eps)
			}
			return nil
		}
		m.Blocks = append(m.Blocks, &MixtralBlock{
			AttnNorm: rmsFromGGUF(an, cfg.Eps),
			Wq:       transpose2D(wq),
			Wk:       transpose2D(wk),
			Wv:       transpose2D(wv),
			Wo:       transpose2D(wo),
			QNorm:    qkNorm("self_attn.q_norm.weight"),
			KNorm:    qkNorm("self_attn.k_norm.weight"),
			FFNNorm:  rmsFromGGUF(fn, cfg.Eps),
			MoE:      moe,
		})
	}
	// Infer the MoE geometry into the config (as documented on MixtralConfig):
	// Experts from the router width, Hidden from an expert's [dim, ffn] gate.
	m.Config.Experts = len(m.Blocks[0].MoE.Experts)
	m.Config.Hidden = m.Blocks[0].MoE.Experts[0].Wgate.Shape()[1]
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Mixtral missing model.norm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head := tok
	if o, ok := ts["lm_head.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head)
	return m, nil
}

// mixtralMoE loads one block's sparse-MoE (router + experts) into an nn.SparseMoE.
func mixtralMoE(ts map[string]*tensor.Tensor, p string, cfg MixtralConfig) (*nn.SparseMoE, error) {
	gate, ok := ts[p+"block_sparse_moe.gate.weight"]
	if !ok {
		gate, ok = ts[p+"mlp.gate.weight"] // newer transformers key
	}
	if !ok {
		return nil, fmt.Errorf("nlp: HF Mixtral missing %s(block_sparse_moe|mlp).gate.weight", p)
	}
	experts := gate.Shape()[0] // [E, dim]

	// Fused expert tensors: gate_up_proj [E, 2·ffn, dim], down_proj [E, dim, ffn].
	gu, okGU := ts[p+"mlp.experts.gate_up_proj"]
	dn, okDN := ts[p+"mlp.experts.down_proj"]
	if !okGU || !okDN {
		return nil, fmt.Errorf("nlp: HF Mixtral %smlp.experts.{gate_up_proj,down_proj} not found (only the fused layout is supported)", p)
	}
	if gu.Ndim() != 3 || dn.Ndim() != 3 {
		return nil, fmt.Errorf("nlp: Mixtral fused experts must be 3-D, got %v / %v", gu.Shape(), dn.Shape())
	}
	ffn := gu.Shape()[1] / 2 // [E, 2·ffn, dim]

	moe := nn.NewSparseMoE(tensor.F64, cfg.Dim, ffn, experts, cfg.TopK, 0)
	moe.Router.W = transpose2D(gate) // [E,dim] → [dim,E]
	for e := range experts {
		guE := sub3D(gu, e)                                          // [2·ffn, dim]
		moe.Experts[e].Wgate = transpose2D(sliceRows(guE, 0, ffn))   // [dim, ffn]
		moe.Experts[e].Wup = transpose2D(sliceRows(guE, ffn, 2*ffn)) // [dim, ffn]
		moe.Experts[e].Wdown = transpose2D(sub3D(dn, e))             // [dim, ffn] → [ffn, dim]
	}
	return moe, nil
}

// sub3D returns slice e of a [E, R, C] tensor as a fresh F64 [R, C] tensor.
func sub3D(t *tensor.Tensor, e int) *tensor.Tensor {
	r, c := t.Shape()[1], t.Shape()[2]
	src := cloneF64(t).Storage().F64()
	out := tensor.New(tensor.F64, tensor.Shape{r, c})
	copy(out.Storage().F64(), src[e*r*c:(e+1)*r*c])
	return out
}
