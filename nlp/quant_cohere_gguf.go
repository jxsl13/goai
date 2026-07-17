package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantCohereFromGGUF builds a [QuantCohere] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "command-r" (gguf.ReadRaw) —
// the quantized twin of [CohereFromGGUF]: a llama.cpp-quantized Command-R checkpoint
// loads straight into QuantLinear projections without ever materializing full-precision
// weights. The config comes from the same command-r.* metadata keys as the float loader
// (shared [cohereCfgFromGGUFMeta] — epsilon under attention.layer_norm_epsilon, the
// full-LayerNorm key; logit_scale optional with llama.cpp's 0 = unscaled default;
// scaled-RoPE, decoupled-key-length and part-rotation files rejected) and the weights
// from the same token_embd / blk.N.* / output_norm tensor names.
//
// The command-r conventions documented at [CohereFromGGUF] carry over to the quantized
// path, one of them in a form new to the quant loaders:
//
//   - INTERLEAVED RoPE undone ON THE QUANTIZED BYTES: the file stores q/k in the HF
//     interleaved row layout (the converter is a pure rename and LLM_ARCH_COMMAND_R is
//     a LLAMA_ROPE_TYPE_NORM arch — GPT-J interleaved rotary on the stored rows). The
//     float loader permutes those rows to GoAI's split-half layout via
//     [permuteInterleaveToSplit]; here the SAME row map — perm[h·hd+i] = h·hd+2i,
//     perm[h·hd+i+hd/2] = h·hd+2i+1, which is index-for-index [ropeUnpermutePerm]
//     (llama.cpp's NORM-rope storage layout IS the GPT-J interleave, so the llama-arch
//     un-permute and the command-r interleave→split map coincide) — is applied by
//     [quantPermuteRows] directly to the Q-block rows. That is lossless: ggml block
//     quantization is row-granular, so quantize∘permute = permute∘quantize and the
//     result is bit-identical to [QuantizeCohere] on the float-permuted model. This is
//     the first NORM-rope quantized loader outside the llama lineage; v/o are
//     untouched.
//   - ONE weight-only LayerNorm per block (the parallel residual): blk.N.attn_norm
//     feeds BOTH sublayers; it and output_norm decode to f32 γ with a ZERO f32 β
//     (build_norm's NULL bias — mean-centered LayerNorm, not RMSNorm). All 1-D tensors
//     are F32 in real quantized files (llama.cpp never block-quantizes them).
//   - TIED head, no output.weight: the arch has no OUTPUT tensor slot
//     (TENSOR_DUPLICATED from token_embd), so the head wraps the quantized token_embd
//     bytes — which must therefore be stored in a quantized-matmul format — while the
//     same tensor dequantizes into the f32 lookup table. A present output.weight is
//     REJECTED rather than silently ignored, exactly like the float loader.
//   - NO biases and NO QK-norm: any attn_{q,k,v,qkv,output}.bias or per-head
//     attn_{q,k}_norm (the Command R+ n_layer ≥ 64 form) is REJECTED — GoAI's Cohere
//     implements only the bias-free, use_qk_norm=false Command-R form.
//   - Packed attn_qkv accepted: like create_tensor_qkv (and the float loader), a fused
//     blk.N.attn_qkv rows [q; k; v] replaces the split tensors. The quantized weight is
//     unpacked WITHOUT dequantizing via [quantSliceRows] at the float loader's row
//     offsets (0, dim, dim + kv·headDim), then the q/k slices are row-permuted — the
//     float loader's slice-then-permute order.
func QuantCohereFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantCohere, error) {
	cfg, err := cohereCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	dim, kvSize := cfg.Dim, cfg.kvHeads()*cfg.HeadDim

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
	// permute the interleaved q/k Q-block ROWS into GoAI's split-half RoPE layout —
	// the byte-exact quantized form of [permuteInterleaveToSplit] — then wrap.
	mkQPermuted := func(name string, qt gguf.QuantTensor, heads int) (*nn.QuantLinear, error) {
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		perm, err := ropeUnpermutePerm(qt.Shape[0], heads) // == the interleave→split row map, see doc above
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		p, err := quantPermuteRows(qt, perm)
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return mkQ(name, p)
	}
	// weight-only mean-centered LayerNorm from a REQUIRED F32 gain: f32 γ plus a ZERO
	// f32 β — the quantized twin of [layerNormFromHF], composing with f32 activations.
	mkLN := func(name string) (*nn.LayerNorm, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		g, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return &nn.LayerNorm{Gamma: g, Beta: tensor.New(tensor.F32, tensor.Shape{g.Shape()[0]}), Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if len(tok.Shape) != 2 {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight must be 2-D, got %v", tok.Shape)
	}
	cfg.Vocab = tok.Shape[0]

	// The command-r head is tied (TENSOR_DUPLICATED from token_embd; the arch's tensor
	// table has no OUTPUT entry). A present output.weight means an untied checkpoint
	// this type cannot represent — reject rather than silently serve the tied head.
	if _, ok := tensors["output.weight"]; ok {
		return nil, fmt.Errorf("nlp: Cohere GGUF carries output.weight; the command-r architecture ties the head to token_embd")
	}

	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}
	q := &QuantCohere{Config: cfg, TokEmb: emb}
	// Tied head: the SAME quantized bytes serve as the LM head, so token_embd must be
	// stored in a quantized-matmul format (as in real quantized command-r files).
	if q.Out, err = mkQ("token_embd.weight", tok); err != nil {
		return nil, err
	}

	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		// Reject the variants GoAI's Cohere cannot represent — the float loader's
		// exact reject set (qkv/output biases and the Command R+ per-head QK-norms).
		for _, unsupported := range []string{
			"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", "attn_output.bias",
			"attn_q_norm.weight", "attn_k_norm.weight",
		} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: Cohere GGUF carries %s%s; GoAI's Cohere implements only the bias-free, use_qk_norm=false Command-R form", p, unsupported)
			}
		}
		qb := &QuantCohereBlock{}
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Fused form (create_tensor_qkv's alternative layout): rows [q; k; v],
			// unpacked losslessly by quantized row-slice at the float loader's offsets,
			// then the q/k slices row-permuted (slice-then-permute, the float order).
			if len(qkv.Shape) != 2 || qkv.Shape[0] != dim+2*kvSize {
				return nil, fmt.Errorf("nlp: Cohere GGUF %sattn_qkv.weight rows %v, want [%d, in] = [dim+2·kv·hd, in]", p, qkv.Shape, dim+2*kvSize)
			}
			for _, part := range []struct {
				w      **nn.QuantLinear
				r0, r1 int
				heads  int // 0 → no RoPE row permutation (the v slice)
			}{
				{&qb.Wq, 0, dim, cfg.Heads},
				{&qb.Wk, dim, dim + kvSize, cfg.kvHeads()},
				{&qb.Wv, dim + kvSize, dim + 2*kvSize, 0},
			} {
				s, err := quantSliceRows(qkv, part.r0, part.r1)
				if err != nil {
					return nil, fmt.Errorf("nlp: Cohere GGUF %sattn_qkv.weight: %w", p, err)
				}
				if part.heads > 0 {
					*part.w, err = mkQPermuted(p+"attn_qkv.weight", s, part.heads)
				} else {
					*part.w, err = mkQ(p+"attn_qkv.weight", s)
				}
				if err != nil {
					return nil, err
				}
			}
		} else {
			get := func(name string) (gguf.QuantTensor, error) {
				qt, ok := tensors[p+name]
				if !ok {
					return gguf.QuantTensor{}, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
				}
				return qt, nil
			}
			wq, err := get("attn_q.weight")
			if err != nil {
				return nil, err
			}
			if qb.Wq, err = mkQPermuted(p+"attn_q.weight", wq, cfg.Heads); err != nil {
				return nil, err
			}
			wk, err := get("attn_k.weight")
			if err != nil {
				return nil, err
			}
			if qb.Wk, err = mkQPermuted(p+"attn_k.weight", wk, cfg.kvHeads()); err != nil {
				return nil, err
			}
			wv, err := get("attn_v.weight")
			if err != nil {
				return nil, err
			}
			if qb.Wv, err = mkQ(p+"attn_v.weight", wv); err != nil {
				return nil, err
			}
		}
		proj := func(name string) (*nn.QuantLinear, error) {
			qt, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return mkQ(p+name, qt)
		}
		if qb.Wo, err = proj("attn_output.weight"); err != nil {
			return nil, err
		}
		if qb.InputNorm, err = mkLN(p + "attn_norm.weight"); err != nil {
			return nil, err
		}
		gate, err := proj("ffn_gate.weight")
		if err != nil {
			return nil, err
		}
		up, err := proj("ffn_up.weight")
		if err != nil {
			return nil, err
		}
		down, err := proj("ffn_down.weight")
		if err != nil {
			return nil, err
		}
		qb.FFN = &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Norm, err = mkLN("output_norm.weight"); err != nil {
		return nil, err
	}
	return q, nil
}
