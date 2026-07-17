package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantMixtralFromGGUF builds a [QuantMixtral] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file (gguf.ReadRaw) — the quantized twin of [MixtralFromGGUF]: a
// llama.cpp-quantized Mixtral checkpoint loads straight into QuantLinear projections
// without ever materializing full-precision weights. Mixtral GGUFs carry
// general.architecture "llama" (llama.cpp has no separate mixtral arch) and are
// distinguished by the MoE metadata; this loader requires llama.expert_count — a dense
// llama file belongs to [QuantLlamaFromGGUF] instead.
//
// The two Mixtral on-disk conventions (verified against llama.cpp master for
// [MixtralFromGGUF]) are undone WITHOUT dequantizing, which is what makes this loader
// interesting:
//
//   - Q/K rows ARE permuted on disk (LlamaModel.permute, undo_permute=True): within
//     each head the converter moves the row at split-half position j·hd/2+i to
//     interleaved position 2i+j. That is a pure ROW permutation — whole rows move,
//     nothing mixes within a row (see ropeUnpermuteRows/gatherRows, which the float
//     loader applies to the same tensors) — and ggml block quantization is
//     row-granular (blocks never span rows; each row owns a disjoint byte range, the
//     quantSliceRows invariant). So the inverse permutation is applied directly to the
//     Q-block bytes by [quantPermuteRows]: the result is bit-identical to quantizing
//     the un-permuted float rows, with no dequantize→permute→requantize error.
//   - Experts are FUSED 3-D: blk.N.ffn_gate_exps/ffn_up_exps [E, ffn, dim] and
//     ffn_down_exps [E, dim, ffn] stack the per-expert matrices on dim 0. Expert e of
//     a row-major quantized [E, R, C] tensor is the contiguous byte range
//     [e·R·rowBytes, (e+1)·R·rowBytes) — the 3-D analog of the same row-granularity
//     argument — so [quantSliceExpert] un-fuses each expert as a lossless byte copy.
//
// The router blk.N.ffn_gate_inp.weight [E, dim] is loaded to f32 and transposed to
// [dim, E] — llama.cpp never quantizes it in real files (it is decoded here whatever
// its storage type), and [QuantMoE] documents why it must stay full-precision. Norm
// gains and the token embedding are dequantized to f32 exactly as in
// [QuantLlamaFromGGUF]; optional blk.N.attn_{q,k}_norm.weight gains (the GoAI
// round-trip extension) load into QNorm/KNorm. An absent output.weight ties the LM
// head to token_embd (which must then be quantized).
func QuantMixtralFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantMixtral, error) {
	const arch = "llama"
	lcfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	experts, err := metaInt(meta, arch+"."+ggufExpertCnt)
	if err != nil {
		return nil, fmt.Errorf("nlp: not a Mixtral (MoE) GGUF — dense llama files load via QuantLlamaFromGGUF: %w", err)
	}
	cfg := MixtralConfig{
		Ctx: lcfg.Ctx, Dim: lcfg.Dim, Heads: lcfg.Heads, KVHeads: lcfg.KVHeads,
		Layers: lcfg.Layers, Hidden: lcfg.Hidden, Experts: experts, TopK: 2,
		Eps: lcfg.Eps, RopeBase: lcfg.RopeBase,
	}
	if k, e := metaInt(meta, arch+"."+ggufExpertUsed); e == nil {
		cfg.TopK = k
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Mixtral GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}

	get := func(name string) (gguf.QuantTensor, error) {
		qt, ok := tensors[name]
		if !ok {
			return gguf.QuantTensor{}, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		return qt, nil
	}
	// wrap a GGUF projection (quantized [out, in]) as a QuantLinear — no transpose, no requant.
	wrapQ := func(name string, qt gguf.QuantTensor) (*nn.QuantLinear, error) {
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s ggml type %d is not a quantized-matmul format", name, qt.GGType)
		}
		return &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}, nil
	}
	mkQ := func(name string) (*nn.QuantLinear, error) {
		qt, err := get(name)
		if err != nil {
			return nil, err
		}
		return wrapQ(name, qt)
	}
	// quantized q/k: un-permute the Q-block ROWS back to the HF split-half layout
	// (the byte-exact inverse of the converter's permute), then wrap.
	mkQUnpermuted := func(name string, heads int) (*nn.QuantLinear, error) {
		qt, err := get(name)
		if err != nil {
			return nil, err
		}
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		perm, err := ropeUnpermutePerm(qt.Shape[0], heads)
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		un, err := quantPermuteRows(qt, perm)
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return wrapQ(name, un)
	}
	// dequantize a 1-D RMSNorm gain to f32.
	mkNorm := func(name string) (*nn.RMSNorm, error) {
		qt, err := get(name)
		if err != nil {
			return nil, err
		}
		g, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}, nil
	}
	// optional small f32 gain (the QK-norm round-trip extension); absent → nil.
	optNorm := func(name string) (*nn.RMSNorm, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, nil
		}
		g, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape[0]
	emb, err := tok.Dequantize() // embedding lookup needs a float table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantMixtral{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		qb := &QuantMixtralBlock{}
		if qb.AttnNorm, err = mkNorm(p + "attn_norm.weight"); err != nil {
			return nil, err
		}
		if qb.Wq, err = mkQUnpermuted(p+"attn_q.weight", cfg.Heads); err != nil {
			return nil, err
		}
		if qb.Wk, err = mkQUnpermuted(p+"attn_k.weight", cfg.kvHeads()); err != nil {
			return nil, err
		}
		if qb.Wv, err = mkQ(p + "attn_v.weight"); err != nil {
			return nil, err
		}
		if qb.Wo, err = mkQ(p + "attn_output.weight"); err != nil {
			return nil, err
		}
		if qb.QNorm, err = optNorm(p + "attn_q_norm.weight"); err != nil {
			return nil, err
		}
		if qb.KNorm, err = optNorm(p + "attn_k_norm.weight"); err != nil {
			return nil, err
		}
		if qb.FFNNorm, err = mkNorm(p + "ffn_norm.weight"); err != nil {
			return nil, err
		}
		if qb.MoE, err = quantMoEFromGGUF(cfg, tensors, p); err != nil {
			return nil, err
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Norm, err = mkNorm("output_norm.weight"); err != nil {
		return nil, err
	}
	head := "output.weight"
	if _, ok := tensors[head]; !ok {
		head = "token_embd.weight" // tied LM head
	}
	if q.Out, err = mkQ(head); err != nil {
		return nil, err
	}
	return q, nil
}

// quantMoEFromGGUF builds one block's [QuantMoE] from the GGUF router ([E, dim], loaded
// f32 — never quantized in real files, see [QuantMoE]) and the fused quantized expert
// tensors (gate/up [E, ffn, dim], down [E, dim, ffn]), un-fusing each expert with
// [quantSliceExpert] — a lossless byte-range copy, no dequantization.
func quantMoEFromGGUF(cfg MixtralConfig, tensors map[string]gguf.QuantTensor, p string) (*QuantMoE, error) {
	router, ok := tensors[p+"ffn_gate_inp.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing %sffn_gate_inp.weight", p)
	}
	if len(router.Shape) != 2 || router.Shape[0] != cfg.Experts || router.Shape[1] != cfg.Dim {
		return nil, fmt.Errorf("nlp: Mixtral GGUF %sffn_gate_inp.weight shape %v, want [%d, %d]",
			p, router.Shape, cfg.Experts, cfg.Dim)
	}
	rt, err := router.Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF %sffn_gate_inp.weight: %w", p, err)
	}
	moe := &QuantMoE{Router: f32Transpose(rt), TopK: cfg.TopK} // [E,dim] → [dim,E]

	exps := make([]gguf.QuantTensor, 3)
	for i, name := range []string{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} {
		qt, ok := tensors[p+name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
		}
		if len(qt.Shape) != 3 {
			return nil, fmt.Errorf("nlp: Mixtral GGUF %s%s must be fused 3-D, got %v", p, name, qt.Shape)
		}
		if qt.Shape[0] != cfg.Experts {
			return nil, fmt.Errorf("nlp: Mixtral GGUF %s%s has %d experts, llama.expert_count says %d",
				p, name, qt.Shape[0], cfg.Experts)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s%s ggml type %d is not a quantized-matmul format", p, name, qt.GGType)
		}
		exps[i] = qt
	}
	gate, up, down := exps[0], exps[1], exps[2]
	if ffn := gate.Shape[1]; ffn != cfg.Hidden || up.Shape[1] != cfg.Hidden || down.Shape[2] != cfg.Hidden {
		return nil, fmt.Errorf("nlp: Mixtral GGUF %s expert width (gate %v / up %v / down %v) disagrees with llama.feed_forward_length %d",
			p, gate.Shape, up.Shape, down.Shape, cfg.Hidden)
	}
	for e := range cfg.Experts {
		mk := func(fused gguf.QuantTensor, name string) (*nn.QuantLinear, error) {
			s, err := quantSliceExpert(fused, e)
			if err != nil {
				return nil, fmt.Errorf("nlp: GGUF %s%s expert %d: %w", p, name, e, err)
			}
			return &nn.QuantLinear{Weight: s.Data, QT: gguf.QuantType(s.GGType), In: s.Shape[1], Out: s.Shape[0]}, nil
		}
		g, err := mk(gate, "ffn_gate_exps.weight")
		if err != nil {
			return nil, err
		}
		u, err := mk(up, "ffn_up_exps.weight")
		if err != nil {
			return nil, err
		}
		d, err := mk(down, "ffn_down_exps.weight")
		if err != nil {
			return nil, err
		}
		moe.Experts = append(moe.Experts, &nn.QuantSwiGLU{Gate: g, Up: u, Down: d})
	}
	return moe, nil
}

// ropeUnpermutePerm builds the row permutation ropeUnpermuteRows applies as an explicit
// index map: perm[dst] = src over rows = heads·hd, with dst split-half position
// q = j·hd/2 + i sourced from interleaved position 2i + j within each head — the exact
// inverse of llama.cpp's LlamaModel.permute. Returns an error (not a panic — this runs
// on untrusted file geometry) when rows is not divisible into heads of even width.
func ropeUnpermutePerm(rows, heads int) ([]int, error) {
	if heads <= 0 || rows%heads != 0 || (rows/heads)%2 != 0 {
		return nil, fmt.Errorf("rope row permute needs rows %d divisible by heads %d with even head_dim", rows, heads)
	}
	hd := rows / heads
	perm := make([]int, rows)
	for h := range heads {
		base := h * hd
		for q := range hd {
			j, i := q/(hd/2), q%(hd/2)
			perm[base+q] = base + 2*i + j
		}
	}
	return perm, nil
}

// quantPermuteRows reorders the ROWS of a 2-D QuantTensor by perm (perm[dst] = src)
// WITHOUT dequantizing — the row-permutation sibling of [quantSliceRows], and exact by
// the same argument: ggml block quantization is row-granular (blocks run along the
// input dimension and never span rows; a row width that is not a block multiple is
// rejected, the invariant gguf.QMatMul enforces), so each row owns a disjoint,
// self-contained byte range of scale+quant blocks and moving rows is moving byte
// ranges. The result is bit-identical to quantizing the row-permuted float matrix —
// which is what lets [QuantMixtralFromGGUF] undo llama.cpp's q/k rope permute on the
// quantized bytes with zero added error. perm must be a permutation of [0, rows).
func quantPermuteRows(qt gguf.QuantTensor, perm []int) (gguf.QuantTensor, error) {
	if len(qt.Shape) != 2 {
		return gguf.QuantTensor{}, fmt.Errorf("row-permute needs a 2-D tensor, got shape %v", qt.Shape)
	}
	rows, cols := qt.Shape[0], qt.Shape[1]
	if len(perm) != rows {
		return gguf.QuantTensor{}, fmt.Errorf("row-permute has %d indices for %d rows", len(perm), rows)
	}
	rowBytes, err := quantRowBytes(qt.GGType, len(qt.Data), rows, cols)
	if err != nil {
		return gguf.QuantTensor{}, err
	}
	data := make([]byte, len(qt.Data))
	seen := make([]bool, rows)
	for dst, src := range perm {
		if src < 0 || src >= rows || seen[src] {
			return gguf.QuantTensor{}, fmt.Errorf("row-permute index %d is not a permutation of [0,%d)", src, rows)
		}
		seen[src] = true
		copy(data[dst*rowBytes:(dst+1)*rowBytes], qt.Data[src*rowBytes:(src+1)*rowBytes])
	}
	return gguf.QuantTensor{Data: data, GGType: qt.GGType, Shape: tensor.Shape{rows, cols}}, nil
}

// quantSliceExpert slices expert e out of a fused 3-D [E, R, C] QuantTensor as a 2-D
// [R, C] QuantTensor WITHOUT dequantizing — the 3-D analog of [quantSliceRows]. It is
// exact for the same reason: blocks never span rows (C must be a block multiple), the
// fused tensor is row-major, so expert e's R rows occupy the contiguous byte range
// [e·R·rowBytes, (e+1)·R·rowBytes) and slicing is one byte copy, bit-identical to
// quantizing that expert's matrix on its own. This is how [QuantMixtralFromGGUF]
// un-fuses blk.N.ffn_{gate,up,down}_exps into per-expert QuantLinear weights.
func quantSliceExpert(qt gguf.QuantTensor, e int) (gguf.QuantTensor, error) {
	if len(qt.Shape) != 3 {
		return gguf.QuantTensor{}, fmt.Errorf("expert-slice needs a 3-D tensor, got shape %v", qt.Shape)
	}
	experts, r, c := qt.Shape[0], qt.Shape[1], qt.Shape[2]
	if e < 0 || e >= experts {
		return gguf.QuantTensor{}, fmt.Errorf("expert %d outside [0,%d)", e, experts)
	}
	rowBytes, err := quantRowBytes(qt.GGType, len(qt.Data), experts*r, c)
	if err != nil {
		return gguf.QuantTensor{}, err
	}
	data := make([]byte, r*rowBytes)
	copy(data, qt.Data[e*r*rowBytes:(e+1)*r*rowBytes])
	return gguf.QuantTensor{Data: data, GGType: qt.GGType, Shape: tensor.Shape{r, c}}, nil
}

// quantRowBytes returns the byte size of one cols-wide row of a block-quantized tensor
// (rows × cols logical elements stored in dataLen bytes), validating the row-granularity
// invariant the quantized slice/permute primitives rely on: cols must be a whole number
// of ggml blocks, so no block spans two rows.
func quantRowBytes(ggType uint32, dataLen, rows, cols int) (int, error) {
	be, err := ggBlockElems(ggType)
	if err != nil {
		return 0, err
	}
	if cols <= 0 || cols%be != 0 {
		return 0, fmt.Errorf("row width %d not a multiple of the %d-element block — blocks would span rows", cols, be)
	}
	totalBlocks := rows * (cols / be)
	if totalBlocks <= 0 || dataLen%totalBlocks != 0 {
		return 0, fmt.Errorf("%d data bytes not divisible into %d blocks", dataLen, totalBlocks)
	}
	return cols / be * (dataLen / totalBlocks), nil
}

// f32Transpose returns the [b, a] F32 transpose of an [a, b] tensor, preserving f32
// values exactly (transpose2D widens to F64; the quantized pipeline's router must stay
// at the f32 the file stores so the GGUF path is bit-equal to [QuantizeMixtral]'s
// f32-cloned router).
func f32Transpose(t *tensor.Tensor) *tensor.Tensor {
	a, b := t.Shape()[0], t.Shape()[1]
	out := tensor.New(tensor.F32, tensor.Shape{b, a})
	dst := out.Storage().F32()
	tc := t.Contiguous()
	switch tc.Dtype() {
	case tensor.F32:
		src := tc.Storage().F32()
		for i := 0; i < a; i++ {
			row := i * b
			for j := 0; j < b; j++ {
				dst[j*a+i] = src[row+j]
			}
		}
	default:
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				dst[j*a+i] = float32(tc.AtF64(i, j))
			}
		}
	}
	return out
}
