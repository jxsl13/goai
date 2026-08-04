package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// JambaFromHF builds a [Jamba] from a Hugging Face JambaForCausalLM checkpoint's
// tensor map (as loaded by safetensors.Load / .LoadFile). Each layer's TYPE is
// decided from the keys it carries, not from config values, so any
// attn/expert-period pattern loads correctly:
//
//   - model.layers.N.self_attn.q_proj.weight present → ATTENTION mixer;
//     otherwise model.layers.N.mamba.in_proj.weight must exist → MAMBA mixer.
//   - model.layers.N.feed_forward.router.weight present → sparse-MoE FFN;
//     otherwise feed_forward.{gate,up,down}_proj.weight → dense SwiGLU.
//
// Weight mapping (verified against transformers.models.jamba.modeling_jamba):
//
//   - Attention: q/k/v/o_proj torch [out,in] → transposed, bias-free GQA. NO
//     rotary keys exist and JambaAttention.forward applies no RoPE (NoPE).
//   - Mamba mixer: identical to [MambaFromHF]'s mapping (in_proj split into
//     x/gate rows, conv1d squeeze, x_proj split into Δ_low/B/C rows, dt_proj
//     with its Δ bias, A_log, D, out_proj — Jamba's mamba_proj_bias=false means
//     in/out_proj carry no bias) PLUS Jamba's three RMSNorms
//     mamba.{dt,b,c}_layernorm.weight on the x_proj outputs.
//   - MoE FFN: feed_forward.router.weight [E, dim] → Router (transposed);
//     fused experts feed_forward.experts.gate_up_proj [E, 2·ffn, dim] (rows
//     [0,ffn) = gate, [ffn,2·ffn) = up per expert) and
//     feed_forward.experts.down_proj [E, dim, ffn], each expert slice
//     transposed — the same fused layout mixtralMoE unpacks.
//   - Dense FFN: feed_forward.{gate,up,down}_proj.weight → nn.SwiGLU.
//   - input_layernorm / pre_ff_layernorm / model.final_layernorm → RMSNorm γ.
//   - lm_head.weight → Out (untied; falls back to the embedding transpose when
//     absent, i.e. tie_word_embeddings).
//
// cfg supplies Heads, TopK and Eps (defaults 2 and 1e-6); Vocab, Dim, KVHeads,
// Layers and all mixer/FFN dims are inferred from the tensors.
func JambaFromHF(ts map[string]*tensor.Tensor, cfg JambaConfig) (*Jamba, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Jamba missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: JambaFromHF needs cfg.Heads")
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 2
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.input_layernorm.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Jamba has no model.layers.*")
	}
	cfg.Layers = layers

	m := &Jamba{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 per-layer loader loop; cold
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Jamba missing %s%s", p, name)
			}
			return t, nil
		}
		in, err := g("input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		ff, err := g("pre_ff_layernorm.weight")
		if err != nil {
			return nil, err
		}
		layer := &JambaLayer{InputNorm: rmsFromGGUF(in, cfg.Eps), PreFFNorm: rmsFromGGUF(ff, cfg.Eps)}

		// Mixer: attention if the layer carries self_attn keys, else Mamba.
		if wq, ok := ts[p+"self_attn.q_proj.weight"]; ok {
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
			// Infer KVHeads from the k_proj height: head_dim = q_out/Heads.
			if hd := wq.Shape()[0] / cfg.Heads; hd > 0 && cfg.KVHeads <= 0 {
				cfg.KVHeads = wk.Shape()[0] / hd
			}
			layer.Attn = &JambaAttention{
				Wq: transpose2D(wq), Wk: transpose2D(wk), Wv: transpose2D(wv), Wo: transpose2D(wo),
			}
		} else {
			mixer, err := jambaMixer(ts, p, cfg)
			if err != nil {
				return nil, err
			}
			layer.Mamba = mixer
		}

		// FFN: sparse MoE if the layer carries a router, else dense SwiGLU.
		if router, ok := ts[p+"feed_forward.router.weight"]; ok {
			moe, err := jambaMoE(ts, p, router, cfg.TopK)
			if err != nil {
				return nil, err
			}
			layer.MoE = moe
		} else {
			gate, err := g("feed_forward.gate_proj.weight")
			if err != nil {
				return nil, err
			}
			up, err := g("feed_forward.up_proj.weight")
			if err != nil {
				return nil, err
			}
			down, err := g("feed_forward.down_proj.weight")
			if err != nil {
				return nil, err
			}
			layer.Dense = swiGLUFromGGUF(gate, up, down)
		}
		m.Layers = append(m.Layers, layer)
	}

	norm, ok := ts["model.final_layernorm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Jamba missing model.final_layernorm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head := tok
	if o, ok := ts["lm_head.weight"]; ok {
		head = o // untied LM head
	}
	m.Out = transpose2D(head)
	m.Config = cfg
	return m, nil
}

// jambaMixer loads one layer's Mamba mixer (prefix p+"mamba.") into a
// [JambaMixer]: the [MambaFromHF] weight mapping plus the dt/b/c RMSNorms.
func jambaMixer(ts map[string]*tensor.Tensor, p string, cfg JambaConfig) (*JambaMixer, error) {
	g := func(name string) (*tensor.Tensor, error) {
		t, ok := ts[p+"mamba."+name]
		if !ok {
			return nil, fmt.Errorf("nlp: HF Jamba missing %smamba.%s", p, name)
		}
		return t, nil
	}
	inProj, err := g("in_proj.weight")
	if err != nil {
		return nil, err
	}
	convW, err := g("conv1d.weight")
	if err != nil {
		return nil, err
	}
	convB, err := g("conv1d.bias")
	if err != nil {
		return nil, err
	}
	xProj, err := g("x_proj.weight")
	if err != nil {
		return nil, err
	}
	dtProjW, err := g("dt_proj.weight")
	if err != nil {
		return nil, err
	}
	dtProjB, err := g("dt_proj.bias")
	if err != nil {
		return nil, err
	}
	aLog, err := g("A_log")
	if err != nil {
		return nil, err
	}
	dSkip, err := g("D")
	if err != nil {
		return nil, err
	}
	outProj, err := g("out_proj.weight")
	if err != nil {
		return nil, err
	}
	dtNorm, err := g("dt_layernorm.weight")
	if err != nil {
		return nil, err
	}
	bNorm, err := g("b_layernorm.weight")
	if err != nil {
		return nil, err
	}
	cNorm, err := g("c_layernorm.weight")
	if err != nil {
		return nil, err
	}

	dInner := inProj.Shape()[0] / 2 // in_proj is [2·d_inner, dim]
	n := aLog.Shape()[1]            // A_log is [d_inner, d_state]
	dConv := convW.Shape()[2]       // conv1d.weight is [d_inner, 1, d_conv]
	dtRank := dtProjW.Shape()[1]    // dt_proj.weight is [d_inner, dt_rank]

	// in_proj split: rows [0:d_inner] = x branch, [d_inner:2·d_inner] = gate.
	inX := sliceRows(inProj, 0, dInner)
	inZ := sliceRows(inProj, dInner, 2*dInner)
	// x_proj output split: [dt_rank | d_state | d_state] = (Δ_low, B, C).
	dtRows := sliceRows(xProj, 0, dtRank)
	bRows := sliceRows(xProj, dtRank, dtRank+n)
	cRows := sliceRows(xProj, dtRank+n, dtRank+2*n)

	block := &nn.MambaBlock{
		InX:     &nn.Linear{W: transpose2D(inX)},
		InZ:     &nn.Linear{W: transpose2D(inZ)},
		ConvW:   squeezeMid(convW),
		ConvB:   cloneF64(convB),
		DtLow:   &nn.Linear{W: transpose2D(dtRows)},
		BProj:   &nn.Linear{W: transpose2D(bRows)},
		CProj:   &nn.Linear{W: transpose2D(cRows)},
		DtProj:  &nn.Linear{W: transpose2D(dtProjW), B: cloneF64(dtProjB)},
		ALog:    cloneF64(aLog),
		Dskip:   cloneF64(dSkip),
		OutProj: &nn.Linear{W: transpose2D(outProj)},
		DModel:  cfg.Dim, DInner: dInner, DConv: dConv, N: n, DtRank: dtRank,
	}
	return &JambaMixer{
		Block:  block,
		DtNorm: rmsFromGGUF(dtNorm, cfg.Eps),
		BNorm:  rmsFromGGUF(bNorm, cfg.Eps),
		CNorm:  rmsFromGGUF(cNorm, cfg.Eps),
	}, nil
}

// jambaMoE loads one layer's fused sparse-MoE (prefix p+"feed_forward.") into a
// [JambaMoE]: gate_up_proj [E, 2·ffn, dim] and down_proj [E, dim, ffn], each
// expert slice unpacked and transposed exactly like mixtralMoE.
func jambaMoE(ts map[string]*tensor.Tensor, p string, router *tensor.Tensor, topK int) (*JambaMoE, error) {
	gu, okGU := ts[p+"feed_forward.experts.gate_up_proj"]
	dn, okDN := ts[p+"feed_forward.experts.down_proj"]
	if !okGU || !okDN {
		return nil, fmt.Errorf("nlp: HF Jamba %sfeed_forward.experts.{gate_up_proj,down_proj} not found (only the fused layout is supported)", p)
	}
	if gu.Ndim() != 3 || dn.Ndim() != 3 {
		return nil, fmt.Errorf("nlp: Jamba fused experts must be 3-D, got %v / %v", gu.Shape(), dn.Shape())
	}
	experts := gu.Shape()[0]
	ffn := gu.Shape()[1] / 2 // [E, 2·ffn, dim]

	moe := &JambaMoE{Router: transpose2D(router), TopK: topK} // [E,dim] → [dim,E]
	//perfscan:ignore PS3060 per-expert MoE loader loop; cold
	for e := range experts {
		guE := sub3D(gu, e) // [2·ffn, dim]
		moe.Experts = append(moe.Experts, &nn.SwiGLU{
			Wgate: transpose2D(sliceRows(guE, 0, ffn)),     // [dim, ffn]
			Wup:   transpose2D(sliceRows(guE, ffn, 2*ffn)), // [dim, ffn]
			Wdown: transpose2D(sub3D(dn, e)),               // [dim, ffn] → [ffn, dim]
		})
	}
	return moe, nil
}
