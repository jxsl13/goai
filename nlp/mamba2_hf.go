package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// Mamba2FromHF builds a [Mamba2] from a Hugging Face Mamba2ForCausalLM
// checkpoint's tensor map (as loaded by safetensors.Load / .LoadFile). It is the
// state-space-duality counterpart of [MambaFromHF].
//
// Most dimensions are inferred from the tensors; the conv width only pins the
// PRODUCT n_groups·state_size, so the caller must set exactly one of cfg.N
// (state_size) or cfg.NGroups — the other is derived. cfg.Eps defaults to 1e-5.
//
// Weight mapping (verified against transformers.models.mamba2.modeling_mamba2's
// Mamba2Mixer.torch_forward, reproduced in host f64 by [Mamba2Mixer.forward] —
// so the SSD math needs no backend op):
//
//   - backbone.embeddings.weight [vocab, d_model] → Embed (no transpose); the tied
//     LM head is its transpose [d_model, vocab].
//   - mixer.in_proj.weight [projection_size, d_model], projection_size =
//     intermediate + conv_dim + num_heads, conv_dim = intermediate + 2·n_groups·N.
//     in_proj OUTPUT splits as [gate(=z) | xBC | dt] (the two zero-width d_mlp
//     splits HF emits are absent). use_bias=false → no bias. Kept in torch [out,in]
//     orientation; the mixer dots over the input axis.
//   - mixer.conv1d.weight [conv_dim, 1, d_conv] → ConvW [conv_dim, d_conv] (squeeze
//     the size-1 middle axis; no flip — Conv1d is cross-correlation = the causal
//     conv the mixer applies). conv1d.bias → ConvB. Depthwise (groups=conv_dim).
//   - mixer.A_log, mixer.D, mixer.dt_bias each [num_heads] (per-head SCALARS —
//     Mamba-2's defining restriction). A = −exp(A_log); Δ = softplus(dt+dt_bias);
//     the D skip adds D[h]·x_h.
//   - mixer.norm.weight [intermediate] → NormW (the gated RMSNorm: norm(y·SiLU(z))).
//   - mixer.out_proj.weight [d_model, intermediate] → OutProj (torch orientation),
//     no bias.
//   - backbone.norm_f.weight and each backbone.layers.N.norm.weight → RMSNorm γ.
//     Residual is pre-norm: h = h + Mixer(RMSNorm(h)).
func Mamba2FromHF(ts map[string]*tensor.Tensor, cfg Mamba2Config) (*Mamba2, error) {
	emb, ok := ts["backbone.embeddings.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Mamba2 missing backbone.embeddings.weight")
	}
	if emb.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: backbone.embeddings.weight must be 2-D, got %v", emb.Shape())
	}
	cfg.Vocab, cfg.DModel = emb.Shape()[0], emb.Shape()[1]
	if cfg.Eps == 0 {
		cfg.Eps = 1e-5
	}
	if cfg.N <= 0 && cfg.NGroups <= 0 {
		return nil, fmt.Errorf("nlp: Mamba2FromHF needs cfg.N (state_size) or cfg.NGroups")
	}

	// Count layers by probing backbone.layers.N.mixer.
	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("backbone.layers.%d.mixer.A_log", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Mamba2 has no backbone.layers.*")
	}
	cfg.Layers = layers

	m := &Mamba2{Config: cfg, Embed: cloneF64(emb)}
	for l := range layers {
		p := fmt.Sprintf("backbone.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Mamba2 missing %s%s", p, name)
			}
			return t, nil
		}
		norm, err := g("norm.weight")
		if err != nil {
			return nil, err
		}
		inProj, err := g("mixer.in_proj.weight")
		if err != nil {
			return nil, err
		}
		convW, err := g("mixer.conv1d.weight")
		if err != nil {
			return nil, err
		}
		convB, err := g("mixer.conv1d.bias")
		if err != nil {
			return nil, err
		}
		aLog, err := g("mixer.A_log")
		if err != nil {
			return nil, err
		}
		dSkip, err := g("mixer.D")
		if err != nil {
			return nil, err
		}
		dtBias, err := g("mixer.dt_bias")
		if err != nil {
			return nil, err
		}
		normW, err := g("mixer.norm.weight")
		if err != nil {
			return nil, err
		}
		outProj, err := g("mixer.out_proj.weight")
		if err != nil {
			return nil, err
		}

		// Infer the block dims from the tensors.
		numHeads := aLog.Shape()[0]        // A_log [num_heads]
		convDim := convW.Shape()[0]        // conv1d.weight [conv_dim, 1, d_conv]
		dConv := convW.Shape()[2]          //
		intermediate := outProj.Shape()[1] // out_proj [d_model, intermediate]
		if numHeads == 0 || intermediate%numHeads != 0 {
			return nil, fmt.Errorf("nlp: Mamba2 intermediate %d not divisible by num_heads %d", intermediate, numHeads)
		}
		headDim := intermediate / numHeads
		// conv_dim = intermediate + 2·n_groups·N ⇒ n_groups·N = (conv_dim−intermediate)/2.
		gN := (convDim - intermediate) / 2
		if gN <= 0 || (convDim-intermediate)%2 != 0 {
			return nil, fmt.Errorf("nlp: Mamba2 conv_dim %d inconsistent with intermediate %d", convDim, intermediate)
		}
		nGroups, stateN := cfg.NGroups, cfg.N
		switch {
		case stateN > 0 && nGroups > 0:
			if nGroups*stateN != gN {
				return nil, fmt.Errorf("nlp: Mamba2 n_groups·N = %d, but conv implies %d", nGroups*stateN, gN)
			}
		case stateN > 0:
			if gN%stateN != 0 {
				return nil, fmt.Errorf("nlp: Mamba2 n_groups·N %d not divisible by N %d", gN, stateN)
			}
			nGroups = gN / stateN
		default: // nGroups > 0
			if gN%nGroups != 0 {
				return nil, fmt.Errorf("nlp: Mamba2 n_groups·N %d not divisible by n_groups %d", gN, nGroups)
			}
			stateN = gN / nGroups
		}
		if numHeads%nGroups != 0 {
			return nil, fmt.Errorf("nlp: Mamba2 num_heads %d not divisible by n_groups %d", numHeads, nGroups)
		}
		cfg.NumHeads, cfg.HeadDim, cfg.NGroups, cfg.N = numHeads, headDim, nGroups, stateN
		cfg.DConv, cfg.Intermediate = dConv, intermediate

		mixer := &Mamba2Mixer{
			InProj:  cloneF64(inProj),  // [projection_size, d_model], torch orientation
			ConvW:   squeezeMid(convW), // [conv_dim, d_conv]
			ConvB:   cloneF64(convB),   // [conv_dim]
			ALog:    cloneF64(aLog),    // [num_heads]
			D:       cloneF64(dSkip),   // [num_heads]
			DtBias:  cloneF64(dtBias),  // [num_heads]
			NormW:   cloneF64(normW),   // [intermediate]
			OutProj: cloneF64(outProj), // [d_model, intermediate], torch orientation
			DModel:  cfg.DModel, NumHeads: numHeads, HeadDim: headDim,
			NGroups: nGroups, N: stateN, DConv: dConv, Intermediate: intermediate,
			ConvDim: convDim, Eps: cfg.Eps,
		}
		m.Layers = append(m.Layers, Mamba2Layer{Norm: rmsFromGGUF(norm, cfg.Eps), Mixer: mixer})
	}

	normF, ok := ts["backbone.norm_f.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Mamba2 missing backbone.norm_f.weight")
	}
	m.Norm = rmsFromGGUF(normF, cfg.Eps)
	m.Config = cfg
	m.Head = transpose2D(m.Embed) // tied head: logits = hidden · Embedᵀ, [d_model, vocab]
	return m, nil
}
