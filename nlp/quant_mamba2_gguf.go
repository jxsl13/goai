package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantMamba2FromGGUF builds a [QuantMamba2] from the metadata and
// STILL-QUANTIZED tensor map of a GGUF file whose general.architecture is
// "mamba2" (gguf.ReadRaw) — the quantized twin of [Mamba2FromGGUF]: a
// llama.cpp-quantized Mamba-2 checkpoint loads straight into QuantLinear
// projections without ever materializing full-precision weights. The config
// comes from the same mamba2.* metadata keys as the float loader (shared
// [mamba2CfgFromGGUFMeta] — ssm.time_step_rank carrying n_head,
// ssm.group_count carrying n_groups, the epsilon under
// attention.layer_norm_rms_epsilon) and the weights from the same token_embd /
// blk.N.ssm_* / output_norm tensor names.
//
// The mamba2 conventions documented at [Mamba2FromGGUF] carry over to the
// quantized path, split by what stays quantized:
//
//   - The fused ssm_in stays PACKED *and* QUANTIZED: blk.N.ssm_in.weight
//     [2·d_inner+2·n_group·d_state+n_head, d_model] wraps as ONE QuantLinear
//     — no split at all, because GoAI's [QuantMamba2Mixer] (like llama.cpp's
//     build_mamba2_layer and the float [Mamba2Mixer]) splits the PRODUCT rows
//     [z | xBC | dt] after the matmul, not the weight. ssm_out.weight
//     [d_model, d_inner] wraps directly too — no transpose, no
//     re-quantization (GGUF's [out, in] row layout is exactly what
//     QuantLinear consumes). These two are the only projections mamba-2 has,
//     and the only mamba2 tensors llama.cpp's quantizer touches besides the
//     embedding/head (src/llama-quant.cpp: only names ending in "weight"
//     qualify; "_norm.weight" and ssm_conv1d are excluded by name).
//   - ssm_a kept as stored, NO log inversion: the file stores A = −exp(A_log)
//     as F32 [n_head, 1] (suffix-less, so it never qualifies for
//     quantization), flattened here to the [n_head] scalar-per-head vector
//     the scan consumes directly — exactly what build_mamba2_layer feeds
//     ggml_ssm_scan. The float loader inverts to A_log only because
//     [Mamba2Mixer] holds the trainable parametrization. The
//     strictly-negative gate is the same: a non-negative element marks a file
//     from a different convention and is rejected, not loaded.
//   - Small tensors decode to f32: the squeezed no-transpose conv kernel
//     [conv_dim, d_conv] and its bias, the Δ bias (ssm_dt.bias [n_head] —
//     mamba-2's dt is bias-only), the D skip (ssm_d [n_head, 1], flattened),
//     the gated-norm gain (ssm_norm.weight [n_group, d_inner/n_group],
//     flattened back to [d_inner]) and the attn_norm/output_norm RMSNorm
//     gains. All stay unquantized in real files.
//   - TIED head with an optional override: a present output.weight (the
//     untied case) wraps as the head; an absent one ties the head to
//     token_embd (llama.cpp's TENSOR_DUPLICATED fallback), which must then
//     itself be stored in a quantized-matmul format. token_embd additionally
//     always feeds the f32 lookup table via dequantization, whatever its
//     storage type.
//
// Dimensions come from metadata and are cross-checked against every tensor
// (mirroring the float loader's fixed create_tensor shapes — a mismatched
// packed tensor would otherwise split its product at wrong offsets and
// misload silently); Vocab comes from token_embd.
func QuantMamba2FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantMamba2, error) {
	cfg, err := mamba2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	convDim := cfg.Intermediate + 2*cfg.NGroups*cfg.N
	dInProj := cfg.Intermediate + convDim + cfg.NumHeads

	// wrap a GGUF projection (quantized [out, in]) as a QuantLinear — no transpose, no requant.
	mkQ := func(name string, qt gguf.QuantTensor) (*nn.QuantLinear, error) {
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s ggml type %d is not a quantized-matmul format", name, qt.GGType)
		}
		return &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}, nil
	}
	// decode a REQUIRED small tensor (conv kernel/bias, Δ bias, A, D, norm gains) to f32.
	mkF32 := func(name string) (*tensor.Tensor, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		v, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return v, nil
	}
	mkRMS := func(name string) (*nn.RMSNorm, error) {
		g, err := mkF32(name)
		if err != nil {
			return nil, err
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if len(tok.Shape) != 2 || tok.Shape[1] != cfg.DModel {
		return nil, fmt.Errorf("nlp: Mamba2 GGUF token_embd.weight %v does not match embedding_length %d", tok.Shape, cfg.DModel)
	}
	cfg.Vocab = tok.Shape[0]
	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantMamba2{Config: cfg, Embed: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		get := func(name string) (gguf.QuantTensor, error) {
			qt, ok := tensors[p+name]
			if !ok {
				return gguf.QuantTensor{}, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return qt, nil
		}
		norm, err := mkRMS(p + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		qm := &QuantMamba2Mixer{
			DModel: cfg.DModel, NumHeads: cfg.NumHeads, HeadDim: cfg.HeadDim,
			NGroups: cfg.NGroups, N: cfg.N, DConv: cfg.DConv,
			Intermediate: cfg.Intermediate, ConvDim: convDim, Eps: cfg.Eps,
		}

		// Fused input projection: kept packed [z|xBC|dt] on the quantized
		// bytes — the product splits at runtime, exactly like the float mixer.
		inProj, err := get("ssm_in.weight")
		if err != nil {
			return nil, err
		}
		if len(inProj.Shape) != 2 || inProj.Shape[0] != dInProj || inProj.Shape[1] != cfg.DModel {
			return nil, fmt.Errorf("nlp: GGUF %sssm_in.weight %v, want packed [2·d_inner+2·n_group·d_state+n_head, d_model] = [%d, %d]", p, inProj.Shape, dInProj, cfg.DModel)
		}
		if qm.InProj, err = mkQ(p+"ssm_in.weight", inProj); err != nil {
			return nil, err
		}

		outW, err := get("ssm_out.weight")
		if err != nil {
			return nil, err
		}
		if len(outW.Shape) != 2 || outW.Shape[0] != cfg.DModel || outW.Shape[1] != cfg.Intermediate {
			return nil, fmt.Errorf("nlp: GGUF %sssm_out.weight %v, want [d_model, d_inner] = [%d, %d]", p, outW.Shape, cfg.DModel, cfg.Intermediate)
		}
		if qm.OutProj, err = mkQ(p+"ssm_out.weight", outW); err != nil {
			return nil, err
		}

		// Squeezed no-transpose conv kernel (excluded from quantization by name).
		if qm.ConvW, err = mkF32(p + "ssm_conv1d.weight"); err != nil {
			return nil, err
		}
		if qm.ConvW.Ndim() != 2 || qm.ConvW.Shape()[0] != convDim || qm.ConvW.Shape()[1] != cfg.DConv {
			return nil, fmt.Errorf("nlp: GGUF %sssm_conv1d.weight %v, want squeezed [d_inner+2·n_group·d_state, d_conv] = [%d, %d] (the converter drops HF's size-1 middle axis)", p, qm.ConvW.Shape(), convDim, cfg.DConv)
		}
		if qm.ConvB, err = mkF32(p + "ssm_conv1d.bias"); err != nil {
			return nil, err
		}
		if qm.ConvB.Ndim() != 1 || qm.ConvB.Shape()[0] != convDim {
			return nil, fmt.Errorf("nlp: GGUF %sssm_conv1d.bias %v, want [%d]", p, qm.ConvB.Shape(), convDim)
		}

		// Δ bias — mamba-2's ssm_dt is bias-only (no dt_proj weight exists).
		if qm.DtBias, err = mkF32(p + "ssm_dt.bias"); err != nil {
			return nil, err
		}
		if qm.DtBias.Ndim() != 1 || qm.DtBias.Shape()[0] != cfg.NumHeads {
			return nil, fmt.Errorf("nlp: GGUF %sssm_dt.bias %v, want [n_head] = [%d]", p, qm.DtBias.Shape(), cfg.NumHeads)
		}

		// ssm_a / ssm_d: suffix-less, unsqueezed [n_head, 1], F32. A stays as
		// stored — −exp(A_log), consumed directly by the scan — gated strictly
		// negative; both flatten to the [n_head] per-head vectors.
		a, err := mkF32(p + "ssm_a")
		if err != nil {
			return nil, err
		}
		if a.Ndim() != 2 || a.Shape()[0] != cfg.NumHeads || a.Shape()[1] != 1 {
			return nil, fmt.Errorf("nlp: GGUF %sssm_a %v, want unsqueezed [n_head, 1] = [%d, 1]", p, a.Shape(), cfg.NumHeads)
		}
		for h := range cfg.NumHeads {
			if v := a.AtF64(h, 0); v >= 0 {
				return nil, fmt.Errorf("nlp: GGUF %sssm_a element %d is %g ≥ 0; the mamba2 convention stores A = −exp(A_log), which is strictly negative", p, h, v)
			}
		}
		qm.A = flattenF32(a)
		d, err := mkF32(p + "ssm_d")
		if err != nil {
			return nil, err
		}
		if d.Ndim() != 2 || d.Shape()[0] != cfg.NumHeads || d.Shape()[1] != 1 {
			return nil, fmt.Errorf("nlp: GGUF %sssm_d %v, want unsqueezed [n_head, 1] = [%d, 1]", p, d.Shape(), cfg.NumHeads)
		}
		qm.D = flattenF32(d)

		// Group-reshaped gated-norm gain, flattened back to [d_inner]
		// (a "_norm.weight" name — llama.cpp never quantizes it).
		nw, err := mkF32(p + "ssm_norm.weight")
		if err != nil {
			return nil, err
		}
		if nw.Ndim() != 2 || nw.Shape()[0] != cfg.NGroups || nw.Shape()[1] != cfg.Intermediate/cfg.NGroups {
			return nil, fmt.Errorf("nlp: GGUF %sssm_norm.weight %v, want group-reshaped [n_group, d_inner/n_group] = [%d, %d]", p, nw.Shape(), cfg.NGroups, cfg.Intermediate/cfg.NGroups)
		}
		qm.NormW = flattenF32(nw)

		q.Layers = append(q.Layers, QuantMamba2Layer{Norm: norm, Mixer: qm})
	}

	if q.Norm, err = mkRMS("output_norm.weight"); err != nil {
		return nil, err
	}
	// Tied head with optional override: absent output.weight → the quantized
	// token_embd bytes (llama.cpp's TENSOR_DUPLICATED fallback); present → untied.
	if o, ok := tensors["output.weight"]; ok {
		if len(o.Shape) != 2 || o.Shape[0] != cfg.Vocab || o.Shape[1] != cfg.DModel {
			return nil, fmt.Errorf("nlp: Mamba2 GGUF output.weight %v, want [vocab, d_model] = [%d, %d]", o.Shape, cfg.Vocab, cfg.DModel)
		}
		if q.Head, err = mkQ("output.weight", o); err != nil {
			return nil, err
		}
	} else if q.Head, err = mkQ("token_embd.weight", tok); err != nil {
		return nil, err
	}
	return q, nil
}

// flattenF32 copies a contiguous F32 tensor into a fresh 1-D F32 vector of the
// same element count, row-major order preserved — the loader-side inverse of
// the mamba2 converter's contiguous reshapes ([n_head, 1] unsqueeze and the
// [n_group, d_inner/n_group] norm regroup). The caller shape-checks first.
func flattenF32(t *tensor.Tensor) *tensor.Tensor {
	c := t.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{t.Numel()})
	copy(out.Storage().F32(), c.Storage().F32())
	return out
}
