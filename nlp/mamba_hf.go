package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MambaFromHF builds a [Mamba] from a Hugging Face MambaForCausalLM checkpoint's
// tensor map (as loaded by safetensors.Load / .LoadFile). Every dimension is
// inferred from the tensors; only cfg.Eps is honoured (default 1e-5). It is the
// state-space counterpart of [LlamaFromHF].
//
// Weight mapping (verified against transformers.models.mamba.modeling_mamba's
// MambaMixer.slow_forward, which the ref OpConv1D/OpSSM kernels reproduce
// exactly — no loader-level adaptation of the SSM math is needed):
//
//   - backbone.embeddings.weight [vocab, d_model] → Embed (no transpose); the
//     tied LM head is its transpose [d_model, vocab].
//   - mixer.in_proj.weight [2·d_inner, d_model]: HF computes in_proj(u) then
//     chunks into (hidden, gate). So rows [0:d_inner] are the x branch → InX and
//     rows [d_inner:2·d_inner] are the z/gate branch → InZ. Each is torch
//     [out,in] → transpose to nn.Linear's [in,out]. use_bias=false → no bias.
//   - mixer.conv1d.weight [d_inner, 1, d_conv] → ConvW [d_inner, d_conv] (squeeze
//     the size-1 middle axis; no flip — PyTorch Conv1d is cross-correlation, which
//     is exactly the ref conv1d tap convention). conv1d.bias → ConvB.
//   - mixer.x_proj.weight [dt_rank+2·d_state, d_inner]: HF splits the OUTPUT into
//     [dt_rank, d_state, d_state] = (Δ_low, B, C). So rows [0:dt_rank]→DtLow,
//     [dt_rank:dt_rank+N]→BProj, [dt_rank+N:dt_rank+2N]→CProj, each transposed.
//   - mixer.dt_proj.weight [d_inner, dt_rank] → DtProj.W (transposed); dt_proj.bias
//     → DtProj.B. The Δ bias is applied INSIDE softplus (Δ = softplus(dt_proj·x +
//     dt_bias)) — it must be loaded or parity breaks.
//   - mixer.A_log [d_inner, d_state] → ALog (no transpose); the block forms
//     A = −exp(A_log). mixer.D [d_inner] → Dskip (the u skip). out_proj.weight
//     [d_model, d_inner] → OutProj (transposed).
//   - backbone.norm_f.weight and each backbone.layers.N.norm.weight → RMSNorm γ.
//     Residual is pre-norm: h = h + Mixer(RMSNorm(h)).
func MambaFromHF(ts map[string]*tensor.Tensor, cfg MambaConfig) (*Mamba, error) {
	emb, ok := ts["backbone.embeddings.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Mamba missing backbone.embeddings.weight")
	}
	if emb.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: backbone.embeddings.weight must be 2-D, got %v", emb.Shape())
	}
	cfg.Vocab, cfg.DModel = emb.Shape()[0], emb.Shape()[1]
	if cfg.Eps == 0 {
		cfg.Eps = 1e-5
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
		return nil, fmt.Errorf("nlp: HF Mamba has no backbone.layers.*")
	}
	cfg.Layers = layers

	m := &Mamba{Config: cfg, Embed: cloneF64(emb)} // [vocab,d_model], no transpose
	//perfscan:ignore PS3060 model-load weight transpose, one-time
	for l := range layers {
		p := fmt.Sprintf("backbone.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Mamba missing %s%s", p, name)
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
		xProj, err := g("mixer.x_proj.weight")
		if err != nil {
			return nil, err
		}
		dtProjW, err := g("mixer.dt_proj.weight")
		if err != nil {
			return nil, err
		}
		dtProjB, err := g("mixer.dt_proj.bias")
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
		outProj, err := g("mixer.out_proj.weight")
		if err != nil {
			return nil, err
		}

		// Infer the block dims from the tensors.
		dInner := inProj.Shape()[0] / 2 // in_proj is [2·d_inner, d_model]
		n := aLog.Shape()[1]            // A_log is [d_inner, d_state]
		dConv := convW.Shape()[2]       // conv1d.weight is [d_inner, 1, d_conv]
		dtRank := dtProjW.Shape()[1]    // dt_proj.weight is [d_inner, dt_rank]
		cfg.N, cfg.DConv, cfg.DtRank, cfg.Expand = n, dConv, dtRank, dInner/cfg.DModel

		// in_proj split: rows [0:d_inner] = x branch, [d_inner:2·d_inner] = gate.
		inX := sliceRows(inProj, 0, dInner)
		inZ := sliceRows(inProj, dInner, 2*dInner)
		// x_proj output split: [dt_rank | d_state | d_state] = (Δ_low, B, C).
		dtRows := sliceRows(xProj, 0, dtRank)
		bRows := sliceRows(xProj, dtRank, dtRank+n)
		cRows := sliceRows(xProj, dtRank+n, dtRank+2*n)

		mixer := &nn.MambaBlock{
			InX:     &nn.Linear{W: transpose2D(inX)},                           // [d_model, d_inner], no bias
			InZ:     &nn.Linear{W: transpose2D(inZ)},                           // [d_model, d_inner], no bias
			ConvW:   squeezeMid(convW),                                         // [d_inner, d_conv]
			ConvB:   cloneF64(convB),                                           // [d_inner]
			DtLow:   &nn.Linear{W: transpose2D(dtRows)},                        // [d_inner, dt_rank], no bias
			BProj:   &nn.Linear{W: transpose2D(bRows)},                         // [d_inner, d_state], no bias
			CProj:   &nn.Linear{W: transpose2D(cRows)},                         // [d_inner, d_state], no bias
			DtProj:  &nn.Linear{W: transpose2D(dtProjW), B: cloneF64(dtProjB)}, // [dt_rank, d_inner] + Δ bias
			ALog:    cloneF64(aLog),                                            // [d_inner, d_state]
			Dskip:   cloneF64(dSkip),                                           // [d_inner]
			OutProj: &nn.Linear{W: transpose2D(outProj)},                       // [d_inner, d_model], no bias
			DModel:  cfg.DModel, DInner: dInner, DConv: dConv, N: n, DtRank: dtRank,
		}
		m.Layers = append(m.Layers, MambaLayer{Norm: rmsFromGGUF(norm, cfg.Eps), Mixer: mixer})
	}

	normF, ok := ts["backbone.norm_f.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Mamba missing backbone.norm_f.weight")
	}
	m.Norm = rmsFromGGUF(normF, cfg.Eps)
	m.Config = cfg
	m.Head = transpose2D(m.Embed) // tied head: logits = hidden · Embedᵀ, [d_model, vocab]
	return m, nil
}

// squeezeMid drops the size-1 middle axis of a [D,1,K] tensor, returning a fresh
// F64 [D,K] (row-major storage order is unchanged by dropping a size-1 axis). Used
// to turn HF's conv1d.weight [d_inner, 1, d_conv] into nn.MambaBlock's ConvW.
func squeezeMid(t *tensor.Tensor) *tensor.Tensor {
	d, k := t.Shape()[0], t.Shape()[2]
	out := tensor.New(tensor.F64, tensor.Shape{d, k})
	copy(out.Storage().F64(), cloneF64(t).Storage().F64())
	return out
}
