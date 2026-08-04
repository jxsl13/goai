package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantMambaFromGGUF builds a [QuantMamba] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "mamba" (gguf.ReadRaw)
// — the quantized twin of [MambaFromGGUF], and the first quantized loader for a
// recurrent (non-attention) architecture: a llama.cpp-quantized Mamba checkpoint
// loads straight into QuantLinear projections without ever materializing
// full-precision weights. The config comes from the same mamba.* metadata keys
// as the float loader (shared [mambaCfgFromGGUFMeta] — the SSM geometry under
// mamba.ssm.*, epsilon under attention.layer_norm_rms_epsilon, FalconMamba's
// dt_b_c_rms=true rejected) and the weights from the same token_embd /
// blk.N.ssm_* / output_norm tensor names.
//
// The mamba conventions documented at [MambaFromGGUF] carry over to the
// quantized path, split by what stays quantized:
//
//   - Packed ssm_in UNPACKED WITHOUT dequantizing: blk.N.ssm_in.weight
//     [2·d_inner, d_model] is row-split at d_inner into the x and z branches via
//     [quantSliceRows] — the byte-exact quantized form of the float loader's
//     sliceRows split (ggml blocks are row-granular, so each product-row band is
//     bit-identical to quantizing the split projection directly). Likewise
//     packed ssm_x [dt_rank+2·d_state, d_inner] splits at dt_rank and
//     dt_rank+d_state into the Δ_low | B | C bands, and ssm_dt.weight /
//     ssm_out.weight wrap directly — no transpose, no re-quantization
//     (GGUF's [out, in] row layout is exactly what QuantLinear consumes).
//   - ssm_a kept as stored, NO log inversion: the file stores A = −exp(A_log)
//     (the converter's "A_log --> A") as F32 — the suffix-less ssm_a never
//     qualifies for quantization (llama.cpp only quantizes tensors ending in
//     "weight"), and ssm_conv1d is excluded by name ("do not quantize Mamba's
//     small yet 2D weights") — and [QuantMamba] consumes A directly, exactly like
//     build_mamba_layer feeds ssm_a into ggml_ssm_scan. The float loader inverts
//     to A_log only because [nn.MambaBlock] holds the trainable parametrization.
//     The strictly-negative gate is the same: a non-negative element marks a
//     file from a different convention and is rejected, not loaded.
//   - Small tensors decode to f32: the squeezed no-transpose conv kernel
//     [d_inner, d_conv] and its bias, the Δ bias (ssm_dt.bias), the D skip
//     (ssm_d — suffix-less, like ssm_a), and the attn_norm/output_norm RMSNorm
//     gains. All are stored unquantized in real files (1-D tensors never
//     block-quantize; ssm_conv1d/ssm_a are excluded by name).
//   - TIED head with an optional override: a present output.weight (the untied
//     case) wraps as the head; an absent one ties the head to token_embd
//     (llama.cpp's TENSOR_DUPLICATED fallback), which must then itself be stored
//     in a quantized-matmul format. token_embd additionally always feeds the f32
//     lookup table via dequantization, whatever its storage type.
//
// Dimensions come from metadata and are cross-checked against every tensor
// (mirroring the float loader's fixed create_tensor shapes — a mismatched
// tensor would otherwise split at wrong row offsets and misload silently);
// Vocab comes from token_embd.
func QuantMambaFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantMamba, error) {
	cfg, dInner, err := mambaCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}

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
	// decode a REQUIRED small tensor (conv kernel/bias, Δ bias, A, D, norm gain) to f32.
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
		return nil, fmt.Errorf("nlp: Mamba GGUF token_embd.weight %v does not match embedding_length %d", tok.Shape, cfg.DModel)
	}
	cfg.Vocab = tok.Shape[0]
	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantMamba{Config: cfg, Embed: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		norm, err := mkRMS(p + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		qm, err := quantSSMMixerFromGGUF(tensors, p, cfg.DModel, dInner, cfg.DConv, cfg.N, cfg.DtRank)
		if err != nil {
			return nil, err
		}
		q.Layers = append(q.Layers, QuantMambaLayer{Norm: norm, Mixer: qm})
	}

	if q.Norm, err = mkRMS("output_norm.weight"); err != nil {
		return nil, err
	}
	// Tied head with optional override: absent output.weight → the quantized
	// token_embd bytes (llama.cpp's TENSOR_DUPLICATED fallback); present → untied.
	if o, ok := tensors["output.weight"]; ok {
		if len(o.Shape) != 2 || o.Shape[0] != cfg.Vocab || o.Shape[1] != cfg.DModel {
			return nil, fmt.Errorf("nlp: Mamba GGUF output.weight %v, want [vocab, d_model] = [%d, %d]", o.Shape, cfg.Vocab, cfg.DModel)
		}
		if q.Head, err = mkQ("output.weight", o); err != nil {
			return nil, err
		}
	} else if q.Head, err = mkQ("token_embd.weight", tok); err != nil {
		return nil, err
	}
	return q, nil
}

// quantSSMMixerFromGGUF loads one block's quantized Mamba-1 selective-scan mixer from
// the GGUF blk-prefix p (tensors ssm_in / ssm_conv1d(+bias) / ssm_x / ssm_dt(+bias) /
// ssm_a / ssm_d / ssm_out) into a [QuantMambaMixer], applying the byte-preserving
// conventions documented at [QuantMambaFromGGUF]: the packed ssm_in / ssm_x row bands
// split on the quantized bytes via [quantSliceRows] at the float loader's exact
// offsets, ssm_dt/ssm_out wrapped directly, the small tensors (conv kernel + bias,
// Δ bias, the strictly-negative ssm_a = −exp(A_log), the D skip) decoded to f32, and
// every shape cross-checked against the metadata dims — a mismatched packed tensor
// would otherwise split at wrong row offsets and misload silently. Shared by
// [QuantMambaFromGGUF] and [QuantJambaFromGGUF] (whose mixer adds the dedicated
// dt/b/c norm gains on top), exactly as the float loaders share [ssmMixerFromGGUF].
func quantSSMMixerFromGGUF(tensors map[string]gguf.QuantTensor, p string, dModel, dInner, dConv, n, dtRank int) (*QuantMambaMixer, error) {
	get := func(name string) (gguf.QuantTensor, error) {
		qt, ok := tensors[p+name]
		if !ok {
			return gguf.QuantTensor{}, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
		}
		return qt, nil
	}
	mkQ := func(name string, qt gguf.QuantTensor) (*nn.QuantLinear, error) {
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s ggml type %d is not a quantized-matmul format", name, qt.GGType)
		}
		return &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}, nil
	}
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

	qm := &QuantMambaMixer{
		DModel: dModel, DInner: dInner, DConv: dConv, N: n, DtRank: dtRank,
	}

	// Packed x+z input projection: rows [0:d_inner]=x, [d_inner:2·d_inner]=z
	// (build_mamba_layer's xz views), split on the quantized bytes.
	inProj, err := get("ssm_in.weight")
	if err != nil {
		return nil, err
	}
	if len(inProj.Shape) != 2 || inProj.Shape[0] != 2*dInner || inProj.Shape[1] != dModel {
		return nil, fmt.Errorf("nlp: GGUF %sssm_in.weight %v, want [2·d_inner, d_model] = [%d, %d]", p, inProj.Shape, 2*dInner, dModel)
	}
	for _, part := range []struct {
		w      **nn.QuantLinear
		r0, r1 int
	}{
		{&qm.InX, 0, dInner},
		{&qm.InZ, dInner, 2 * dInner},
	} {
		s, err := quantSliceRows(inProj, part.r0, part.r1)
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %sssm_in.weight: %w", p, err)
		}
		if *part.w, err = mkQ(p+"ssm_in.weight", s); err != nil {
			return nil, err
		}
	}

	// Packed Δ+B+C projection: rows [0:dt_rank]=Δ_low, then B, then C.
	xProj, err := get("ssm_x.weight")
	if err != nil {
		return nil, err
	}
	if len(xProj.Shape) != 2 || xProj.Shape[0] != dtRank+2*n || xProj.Shape[1] != dInner {
		return nil, fmt.Errorf("nlp: GGUF %sssm_x.weight %v, want [dt_rank+2·d_state, d_inner] = [%d, %d]", p, xProj.Shape, dtRank+2*n, dInner)
	}
	for _, part := range []struct {
		w      **nn.QuantLinear
		r0, r1 int
	}{
		{&qm.DtLow, 0, dtRank},
		{&qm.BProj, dtRank, dtRank + n},
		{&qm.CProj, dtRank + n, dtRank + 2*n},
	} {
		s, err := quantSliceRows(xProj, part.r0, part.r1)
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %sssm_x.weight: %w", p, err)
		}
		if *part.w, err = mkQ(p+"ssm_x.weight", s); err != nil {
			return nil, err
		}
	}

	dtW, err := get("ssm_dt.weight")
	if err != nil {
		return nil, err
	}
	if len(dtW.Shape) != 2 || dtW.Shape[0] != dInner || dtW.Shape[1] != dtRank {
		return nil, fmt.Errorf("nlp: GGUF %sssm_dt.weight %v, want [d_inner, dt_rank] = [%d, %d]", p, dtW.Shape, dInner, dtRank)
	}
	if qm.DtProj, err = mkQ(p+"ssm_dt.weight", dtW); err != nil {
		return nil, err
	}
	if qm.DtBias, err = mkF32(p + "ssm_dt.bias"); err != nil {
		return nil, err
	}

	outW, err := get("ssm_out.weight")
	if err != nil {
		return nil, err
	}
	if len(outW.Shape) != 2 || outW.Shape[0] != dModel || outW.Shape[1] != dInner {
		return nil, fmt.Errorf("nlp: GGUF %sssm_out.weight %v, want [d_model, d_inner] = [%d, %d]", p, outW.Shape, dModel, dInner)
	}
	if qm.OutProj, err = mkQ(p+"ssm_out.weight", outW); err != nil {
		return nil, err
	}

	// Squeezed no-transpose conv kernel (the converter drops HF's size-1
	// middle axis; llama.cpp excludes it from quantization by name).
	if qm.ConvW, err = mkF32(p + "ssm_conv1d.weight"); err != nil {
		return nil, err
	}
	if qm.ConvW.Ndim() != 2 || qm.ConvW.Shape()[0] != dInner || qm.ConvW.Shape()[1] != dConv {
		return nil, fmt.Errorf("nlp: GGUF %sssm_conv1d.weight %v, want squeezed [d_inner, d_conv] = [%d, %d] (the converter drops HF's size-1 middle axis)", p, qm.ConvW.Shape(), dInner, dConv)
	}
	if qm.ConvB, err = mkF32(p + "ssm_conv1d.bias"); err != nil {
		return nil, err
	}

	// ssm_a / ssm_d: no ".weight" suffix (llama-arch.cpp). A stays as stored —
	// −exp(A_log), consumed directly by the scan — gated strictly negative.
	if qm.A, err = mkF32(p + "ssm_a"); err != nil {
		return nil, err
	}
	if qm.A.Ndim() != 2 || qm.A.Shape()[0] != dInner || qm.A.Shape()[1] != n {
		return nil, fmt.Errorf("nlp: GGUF %sssm_a %v, want [d_inner, d_state] = [%d, %d]", p, qm.A.Shape(), dInner, n)
	}
	//perfscan:ignore PS1001 A-sign validation at GGUF load, one-time
	for i := range qm.A.Numel() {
		idx := tensor.Unravel(i, qm.A.Shape())
		if v := qm.A.AtF64(idx...); v >= 0 {
			return nil, fmt.Errorf("nlp: GGUF %sssm_a element %d is %g ≥ 0; the mamba convention stores A = −exp(A_log), which is strictly negative", p, i, v)
		}
	}
	if qm.Dskip, err = mkF32(p + "ssm_d"); err != nil {
		return nil, err
	}
	return qm, nil
}
