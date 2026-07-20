package nlp

import (
	"fmt"
	"strings"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GGUF LLM / attention / rope metadata keys (the ggml/llama.cpp convention, §R93).
// Every LLM key is prefixed with the general.architecture string — the same suffix set
// serves "llama.embedding_length", "qwen2.embedding_length", "qwen3.embedding_length", …
// so the llama-family loaders share these suffixes via llamaArchFromGGUF.
const (
	ggufArch     = "general.architecture"
	ggufEmbLen   = "embedding_length"
	ggufBlockCnt = "block_count"
	ggufFFLen    = "feed_forward_length"
	ggufHeadCnt  = "attention.head_count"
	ggufHeadKV   = "attention.head_count_kv"
	ggufRMSEps   = "attention.layer_norm_rms_epsilon"
	ggufCtxLen   = "context_length"
	ggufRopeFreq = "rope.freq_base"
)

// LlamaFromGGUF builds a Llama from the metadata and (dequantized) tensor maps of a
// parsed GGUF model file (gguf.File.Metadata / .Tensors), reading the ggml/llama.cpp
// convention (§R93): the config from the llama.* metadata keys and the weights from the
// token_embd / blk.N.* / output tensors. GGUF stores linear weights in torch
// [out, in] layout, so every projection is TRANSPOSED into GoAI's [in, out]; the
// embedding and RMSNorm gains are copied as-is. An absent output.weight ties the LM
// head to token_embd.
//
// NOTE (§B67, supersedes the original §R93 reading): llama.cpp's converter PERMUTES the
// attn_q/attn_k rows of the llama arch (conversion/llama.py LlamaModel, undo_permute=True:
// per head, split-half row i↔interleaved row 2i pairing) because ggml's NORM rope rotates
// CONSECUTIVE value pairs — so an on-disk llama/mistral GGUF carries the INTERLEAVED
// layout. GoAI's OpRoPE is the split-half convention (matching raw HF weights), so this
// loader UN-permutes q/k at load via [ropeUnpermuteRows] — the same verified transform the
// Mixtral (arch "llama" with experts) and Granite loaders apply — and likewise the optional
// q/k BIASES via [ropeUnpermuteVec], since the converter permutes q_proj.bias/k_proj.bias
// identically (bias-carrying llama-arch files: the LLaMAfied-Qwen lineage). [LlamaToGGUF]
// writes the permuted on-disk form, so GoAI round-trips stay exact AND the files match
// llama.cpp's convention. The original §R93 note claimed the stored layout already matched GoAI's rope
// and was masked by GoAI-only round-trip fixtures (§B67); the decisive fixture now applies
// llama.cpp's permute independently in the test and demands HF-golden logit parity.
// Qwen2/Qwen3 ride [llamaArchFromGGUF] directly: those archs are NEOX/no-permute on disk.
func LlamaFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	cfg, err := llamaCfgFromGGUFMeta("llama", meta)
	if err != nil {
		return nil, err
	}
	ts := make(map[string]*tensor.Tensor, len(tensors))
	for k, v := range tensors {
		ts[k] = v
	}
	// The rope permutes below slice each head's rows out of the tensor, so the
	// tensor's OWN row count has to agree with the head count and leave an even
	// per-head width (RoPE rotates channel pairs). That row count is separately
	// attacker-controlled — a validated config does not imply it — so it is checked
	// here: without this the mismatch reached gatherRows and PANICKED, where §V29
	// requires a clean error. [ropeRowGeom] / [ropeVecGeom] route that check through
	// ropeUnpermutePerm, the same predicate the quantized twins gate on.
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		// The rank test is INSIDE the body, not part of the presence condition.
		// Written as `ok && wq.Ndim() == 2` it read like a guard but behaved as the
		// opposite: a rank-1 tensor made the condition false, skipped the very
		// validation meant to reject it, and fell through to transpose2D, which
		// panicked with `index out of range [1] with length 1`. A guard expressed
		// as a condition is not a guard — it only runs on the inputs that were
		// already fine (§B77).
		if wq, ok := ts[p+"attn_q.weight"]; ok {
			if wq.Ndim() != 2 {
				return nil, fmt.Errorf("nlp: %sattn_q.weight has rank %d, want a 2-D [rows, dim] tensor",
					p, wq.Ndim())
			}
			if err := ropeRowGeom(p+"attn_q.weight", wq, cfg.Heads); err != nil {
				return nil, err
			}
			ts[p+"attn_q.weight"] = ropeUnpermuteRows(wq, cfg.Heads)
		}
		if wk, ok := ts[p+"attn_k.weight"]; ok {
			if wk.Ndim() != 2 {
				return nil, fmt.Errorf("nlp: %sattn_k.weight has rank %d, want a 2-D [rows, dim] tensor",
					p, wk.Ndim())
			}
			if err := ropeRowGeom(p+"attn_k.weight", wk, cfg.kvHeads()); err != nil {
				return nil, err
			}
			ts[p+"attn_k.weight"] = ropeUnpermuteRows(wk, cfg.kvHeads())
		}
		// llama.cpp's converter permutes q_proj.bias/k_proj.bias exactly like the
		// weights (modify_tensors matches both name endings), so bias-carrying
		// llama-arch files — the LLaMAfied-Qwen lineage behind ggml's optional
		// bq/bk/bv tensors — need the same un-permute; v stays raw (§B67 follow-up).
		if bq, ok := ts[p+"attn_q.bias"]; ok && bq.Ndim() == 1 {
			if err := ropeVecGeom(p+"attn_q.bias", bq, cfg.Heads); err != nil {
				return nil, err
			}
			ts[p+"attn_q.bias"] = ropeUnpermuteVec(bq, cfg.Heads)
		}
		if bk, ok := ts[p+"attn_k.bias"]; ok && bk.Ndim() == 1 {
			if err := ropeVecGeom(p+"attn_k.bias", bk, cfg.kvHeads()); err != nil {
				return nil, err
			}
			ts[p+"attn_k.bias"] = ropeUnpermuteVec(bk, cfg.kvHeads())
		}
	}
	return llamaArchFromGGUF("llama", meta, ts)
}

// llamaArchFromGGUF is the shared llama-family GGUF loader behind [LlamaFromGGUF],
// [Qwen2FromGGUF] and [Qwen3FromGGUF]: identical block structure and tensor names, only
// the general.architecture string (= the metadata key prefix) differs, plus optional
// per-block tensors — attn_{q,k,v}.bias (Qwen2 family) and attn_{q,k}_norm.weight
// (Qwen3 per-head QK-norm) — loaded when present, nil (a Forward no-op) otherwise.
func llamaArchFromGGUF(arch string, meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	cfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	layers := cfg.Layers

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Llama{Config: cfg, TokEmb: cloneF64(tok)} // embedding: [vocab,dim], no transpose
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		names := []string{"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
			"attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
		w := make([]*tensor.Tensor, len(names))
		for i, n := range names {
			if w[i], err = g(n); err != nil {
				return nil, err
			}
		}
		// Rank-guard every projection this block will transpose (all names but the two
		// 1-D RMSNorm gains, which load through rmsFromGGUF untransposed). Mirrors
		// quantLlamaArchFromGGUF's mkQ `len(qt.Shape) != 2`; without it a rank-1
		// projection falls through to transpose2D's bare Shape()[1] and PANICS (§B77).
		for i, n := range names {
			if isNormWeight(n) {
				continue
			}
			if err := require2D(p+n, w[i]); err != nil {
				return nil, err
			}
		}
		// Optional q/k/v biases (1-D, copied as-is — no transpose). Qwen2/Qwen3 store
		// them unpermuted on disk (§R93); for the llama arch [LlamaFromGGUF] has
		// already un-permuted q/k before delegating here (§B67 follow-up).
		bias := func(name string) *tensor.Tensor {
			if t, ok := tensors[p+name]; ok {
				return cloneF64(t)
			}
			return nil
		}
		// Optional Qwen3 per-head QK-norm gains ([headDim] each).
		qkNorm := func(name string) *nn.RMSNorm {
			if t, ok := tensors[p+name]; ok {
				return rmsFromGGUF(t, cfg.Eps)
			}
			return nil
		}
		m.Blocks = append(m.Blocks, &LlamaBlock{
			AttnNorm: rmsFromGGUF(w[0], cfg.Eps),
			Wq:       transpose2D(w[1]), // GGUF [out,in] → GoAI [in,out]
			Wk:       transpose2D(w[2]),
			Wv:       transpose2D(w[3]),
			Wo:       transpose2D(w[4]),
			Bq:       bias("attn_q.bias"),
			Bk:       bias("attn_k.bias"),
			Bv:       bias("attn_v.bias"),
			QNorm:    qkNorm("attn_q_norm.weight"),
			KNorm:    qkNorm("attn_k_norm.weight"),
			FFNNorm:  rmsFromGGUF(w[5], cfg.Eps),
			FFN:      swiGLUFromGGUF(w[6], w[7], w[8]),
		})
	}
	on, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.Norm = rmsFromGGUF(on, cfg.Eps)
	head, headName := tok, "token_embd.weight (tied LM head)" // tied when output.weight is absent
	if o, ok := tensors["output.weight"]; ok {
		head, headName = o, "output.weight"
	}
	if err := require2D(headName, head); err != nil { // guards transpose2D below (§B77)
		return nil, err
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// LlamaToGGUF is the inverse of LlamaFromGGUF: it serializes a Llama into the GGUF
// metadata + tensor maps (ggml/llama.cpp names and layout), transposing every
// projection back into torch [out, in] and PERMUTING the attn_q/attn_k rows — weights
// AND optional biases — into the llama arch's on-disk interleaved rotary layout (§B67 —
// llama.cpp's converter permute, applied via [permuteSplitToInterleave] /
// [ropePermuteVec]), so the produced file follows the same convention as a real
// llama.cpp conversion. Pass the result to gguf.Write.
//
// The emitted metadata carries every MODEL-side key llama.cpp's loader requires;
// to produce a file real llama.cpp will load, the caller must additionally add
// the tokenizer block — tokenizer.ggml.model, tokenizer.ggml.tokens (whose
// length MUST equal the model's Vocab) and, for BPE, tokenizer.ggml.merges —
// since the vocab is the caller's data. The full recipe, the required-key table
// and the live verification harness (a GoAI-written file loads, generates, and
// greedy-matches GoAI token-for-token in a real llama.cpp build) are documented
// in docs/gguf.md "Exporting to llama.cpp" / scripts/interop_llamacpp.sh.
func LlamaToGGUF(m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	meta, ts := llamaArchToGGUF("llama", m)
	c := m.Config
	for l := range c.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		if wq, ok := ts[p+"attn_q.weight"]; ok {
			ts[p+"attn_q.weight"] = permuteSplitToInterleave(wq, c.Heads, wq.Shape()[0]/c.Heads)
		}
		if wk, ok := ts[p+"attn_k.weight"]; ok {
			ts[p+"attn_k.weight"] = permuteSplitToInterleave(wk, c.kvHeads(), wk.Shape()[0]/c.kvHeads())
		}
		// q/k biases ride the same converter permute as the weights (v stays raw) —
		// llama.cpp reads a llama-arch file's bq/bk in the interleaved rotary order,
		// so an unpermuted bias would pair wrong channels there (§B67 follow-up).
		if bq, ok := ts[p+"attn_q.bias"]; ok {
			ts[p+"attn_q.bias"] = ropePermuteVec(bq, c.Heads)
		}
		if bk, ok := ts[p+"attn_k.bias"]; ok {
			ts[p+"attn_k.bias"] = ropePermuteVec(bk, c.kvHeads())
		}
	}
	return meta, ts
}

// llamaArchToGGUF is the shared llama-family GGUF serializer behind [LlamaToGGUF],
// [Qwen2ToGGUF] and [Qwen3ToGGUF]: the arch string becomes general.architecture and the
// metadata key prefix; optional block tensors (Qwen2 q/k/v biases, Qwen3 QK-norm gains)
// are emitted only when non-nil, so each architecture writes exactly its tensor set.
func llamaArchToGGUF(arch string, m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:          arch,
		key(ggufEmbLen):   uint32(c.Dim),
		key(ggufBlockCnt): uint32(c.Layers),
		key(ggufFFLen):    uint32(c.Hidden),
		key(ggufHeadCnt):  uint32(c.Heads),
		key(ggufHeadKV):   uint32(c.kvHeads()),
		key(ggufCtxLen):   uint32(c.Ctx),
		key(ggufRMSEps):   float32(c.Eps),
		key(ggufRopeFreq): float32(ropeBaseOr(c.RopeBase)),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim]
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.AttnNorm.Gamma)
		ts[p+"attn_q.weight"] = transpose2D(b.Wq) // GoAI [in,out] → GGUF [out,in]
		ts[p+"attn_k.weight"] = transpose2D(b.Wk)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_norm.weight"] = cloneF64(b.FFNNorm.Gamma)
		ts[p+"ffn_gate.weight"] = transpose2D(b.FFN.Wgate)
		ts[p+"ffn_up.weight"] = transpose2D(b.FFN.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.FFN.Wdown)
		if b.Bq != nil {
			ts[p+"attn_q.bias"] = cloneF64(b.Bq) // 1-D; [LlamaToGGUF] re-permutes q/k for the llama arch
		}
		if b.Bk != nil {
			ts[p+"attn_k.bias"] = cloneF64(b.Bk)
		}
		if b.Bv != nil {
			ts[p+"attn_v.bias"] = cloneF64(b.Bv)
		}
		if b.QNorm != nil {
			ts[p+"attn_q_norm.weight"] = cloneF64(b.QNorm.Gamma)
		}
		if b.KNorm != nil {
			ts[p+"attn_k_norm.weight"] = cloneF64(b.KNorm.Gamma)
		}
	}
	return meta, ts
}

// llamaCfgFromGGUFMeta checks general.architecture against arch and reads the
// llama-family LlamaConfig from the arch-prefixed GGUF metadata keys
// ("<arch>.embedding_length", …). Vocab is left 0 — callers take it from the
// token_embd.weight shape. Shared by llamaArchFromGGUF and [QuantLlamaFromGGUF].
//
// Every structural count is validated to be POSITIVE before it is returned (§V29:
// a malformed file yields a clean error, never a panic and never a wrong-but-
// nil-error model). This is the single choke point every llama-family loader —
// [LlamaFromGGUF], [MixtralFromGGUF], [GraniteFromGGUF], Qwen2/Qwen3 and the
// quantized twins — reads its geometry through, so validating here closes the
// class rather than one caller.
//
// Failure modes this rejects, all reachable from an UNTRUSTED file because the
// counts are attacker-controlled metadata (§B70's lesson from format/pytorch):
//   - head_count = 0 divided straight through to `rows / heads` in gatherRows,
//     an integer-divide-by-zero PANIC before that function's own heads<=0 guard
//     on the following line could fire;
//   - a negative or zero embedding_length / block_count / feed_forward_length
//     produced a model that loaded with err == nil and only misbehaved later,
//     which is the harder bug to trace back to its cause.
//
// A metadata key that is ABSENT keeps its documented default (head_count_kv falls
// back to head_count, context_length to embedding_length); only a key that is
// PRESENT and non-positive is an error. Valid files are unaffected — llamaArchToGGUF
// writes the resolved kvHeads(), never a 0 placeholder.
func llamaCfgFromGGUFMeta(arch string, meta map[string]any) (LlamaConfig, error) {
	if a, _ := meta[ggufArch].(string); a != arch {
		return LlamaConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	// positive reads the count at suffix and rejects a non-positive value: every
	// llama-family structural count is a dimension, and no dimension is ever ≤ 0.
	positive := func(suffix string) (int, error) { return metaPositiveInt(meta, key(suffix)) }
	dim, err := positive(ggufEmbLen)
	if err != nil {
		return LlamaConfig{}, err
	}
	layers, err := positive(ggufBlockCnt)
	if err != nil {
		return LlamaConfig{}, err
	}
	hidden, err := positive(ggufFFLen)
	if err != nil {
		return LlamaConfig{}, err
	}
	heads, err := positive(ggufHeadCnt)
	if err != nil {
		return LlamaConfig{}, err
	}
	// head_count_kv is optional (absent ⇒ multi-head, kv = heads), but a PRESENT
	// value of 0 or less is a malformed geometry, not a request for the fallback.
	kv, err := metaOptionalPositiveInt(meta, key(ggufHeadKV), heads)
	if err != nil {
		return LlamaConfig{}, err
	}
	cfg := LlamaConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: kv,
		Eps: metaFloat(meta, key(ggufRMSEps), 1e-5), RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000),
		Ctx: dim, // provisional; overwritten from context_length below
	}
	if cfg.Ctx, err = metaOptionalPositiveInt(meta, key(ggufCtxLen), dim); err != nil {
		return LlamaConfig{}, err
	}
	return cfg, nil
}

func ropeBaseOr(b float64) float64 {
	if b <= 0 {
		return 10000
	}
	return b
}

// metaInt reads a GGUF metadata value as an int, tolerating the integer types a
// reader may yield (uint32 per spec).
func metaInt(meta map[string]any, key string) (int, error) {
	switch n := meta[key].(type) {
	case uint32:
		return int(n), nil
	case int32:
		return int(n), nil
	case uint64:
		return int(n), nil
	case int64:
		return int(n), nil
	case nil:
		return 0, fmt.Errorf("nlp: GGUF missing %s", key)
	default:
		return 0, fmt.Errorf("nlp: GGUF %s is %T, want an integer", key, n)
	}
}

// metaPositiveInt reads the structural count at key and rejects a non-positive value.
// Every GGUF count this package reads is a DIMENSION — an embedding width, a block or
// head count, a context length — and no dimension is ever ≤ 0, so a zero or negative
// one is a malformed file, not a configuration (§V29: a clean error, never a panic and
// never a wrong-but-nil-error model).
//
// Extracted from [llamaCfgFromGGUFMeta]'s own validation (T861/§B76) so the
// architectures with their OWN metadata parsers — gemmaCfgFromGGUFMeta and
// gemma2CfgFromGGUFMeta, which the llama-family choke point does NOT reach — share one
// predicate and one wording instead of drifting apart. A head_count of 0 reaching a
// bare `rows / heads` is an integer-divide-by-zero PANIC; a negative one silently
// yields a negative head width and loads with err == nil.
func metaPositiveInt(meta map[string]any, key string) (int, error) {
	n, err := metaInt(meta, key)
	if err != nil {
		return 0, err
	}
	if n <= 0 {
		return 0, fmt.Errorf("nlp: GGUF %s = %d, want a positive count", key, n)
	}
	return n, nil
}

// metaOptionalPositiveInt is [metaPositiveInt] for a key that carries a documented
// FALLBACK: an ABSENT key yields def with no error, while a key that is PRESENT and
// non-positive (or not an integer at all) is rejected. That asymmetry is the property
// round-trips depend on — head_count_kv absent means "multi-head, kv = heads" and
// context_length absent means "embedding_length", but neither ever means 0.
func metaOptionalPositiveInt(meta map[string]any, key string, def int) (int, error) {
	if _, present := meta[key]; !present {
		return def, nil
	}
	return metaPositiveInt(meta, key)
}

// metaFloat reads a GGUF metadata value as a float64, returning def when absent.
func metaFloat(meta map[string]any, key string, def float64) float64 {
	switch f := meta[key].(type) {
	case float32:
		return float64(f)
	case float64:
		return f
	default:
		return def
	}
}

// rmsFromGGUF wraps a GGUF RMSNorm gain tensor as an nn.RMSNorm.
func rmsFromGGUF(gamma *tensor.Tensor, eps float64) *nn.RMSNorm {
	return &nn.RMSNorm{Gamma: cloneF64(gamma), Eps: eps}
}

// swiGLUFromGGUF builds an nn.SwiGLU from the GGUF ffn gate/up/down weights,
// transposing each from torch [out,in] into GoAI's [in,out].
func swiGLUFromGGUF(gate, up, down *tensor.Tensor) *nn.SwiGLU {
	return &nn.SwiGLU{Wgate: transpose2D(gate), Wup: transpose2D(up), Wdown: transpose2D(down)}
}

// transpose2D returns the [b,a] transpose of a [a,b] tensor (F64). Runs once per
// weight matrix at model load — a direct typed index transpose (§base-perf) instead
// of per-element AtF64/SetF64 dispatch keeps loading large models fast.
func transpose2D(t *tensor.Tensor) *tensor.Tensor {
	a, b := t.Shape()[0], t.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{b, a})
	tc := t.Contiguous()
	dst := out.Storage().F64() // [b,a] row-major
	switch tc.Dtype() {
	case tensor.F64:
		src := tc.Storage().F64() // [a,b] row-major
		for i := 0; i < a; i++ {
			row := i * b
			for j := 0; j < b; j++ {
				dst[j*a+i] = src[row+j]
			}
		}
	case tensor.F32:
		src := tc.Storage().F32()
		for i := 0; i < a; i++ {
			row := i * b
			for j := 0; j < b; j++ {
				dst[j*a+i] = float64(src[row+j])
			}
		}
	default:
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				dst[j*a+i] = tc.AtF64(i, j)
			}
		}
	}
	return out
}

// require2D reports a clean error when a GGUF weight a loader is about to TRANSPOSE
// ([transpose2D]) or ROW-SLICE ([sliceRows]) is not rank-2, naming the tensor and its
// actual rank. Both helpers read Shape()[1] with no bounds test of their own, so a
// rank-1 tensor from a malformed file indexes out of range and PANICS where §V29 wants
// a clean error (§B77).
//
// This is the float-side twin of the `len(qt.Shape) != 2` guard every quantized loader
// gates each projection on before wrapping it — quantLlamaArchFromGGUF's mkQ,
// QuantCohereFromGGUF, QuantFalconFromGGUF, QuantMixtralFromGGUF and their siblings. The
// predicate is shared as ONE helper so the float loaders reject exactly the rank the
// quantized twins do and NO MORE: only the tensors a loader will transpose or slice are
// checked, matching mkQ, which never rank-checks the 1-D RMSNorm/LayerNorm gains it loads
// through mkNorm. Restating the check inline at each of the ~30 call sites would let the
// float loaders drift apart from the quantized ones and from each other (§B77 lesson:
// share the predicate the twin already uses; match its strictness, no more).
func require2D(name string, t *tensor.Tensor) error {
	if t.Ndim() != 2 {
		return fmt.Errorf("nlp: GGUF %s must be 2-D, got rank %d (shape %v)", name, t.Ndim(), t.Shape())
	}
	return nil
}

// ggufWeight pairs a GGUF tensor with its name for the batched rank guard below.
type ggufWeight struct {
	name string
	t    *tensor.Tensor
}

// require2DEach rank-guards ([require2D]) each projection a block is about to transpose or
// row-slice, returning the first failure. The fused-qkv and own-build loaders call it once
// per block so the shared predicate covers every transpose2D/sliceRows input in one place,
// never restated at the individual call sites (§B77).
func require2DEach(ws ...ggufWeight) error {
	for _, w := range ws {
		if err := require2D(w.name, w.t); err != nil {
			return err
		}
	}
	return nil
}

// isNormWeight reports whether a GGUF block-tensor name is a 1-D norm gain
// (attn_norm.weight, ffn_norm.weight, post_attention_norm.weight, …) rather than a 2-D
// projection. The llama/gemma/gemma2 block builders load every listed tensor through the
// same fetch loop, then transpose only the projections; this lets them [require2D] the
// projections while leaving the norm gains — which are legitimately 1-D — unchecked, the
// same split the quantized twins make between mkQ and mkNorm.
func isNormWeight(name string) bool {
	return strings.HasSuffix(name, "norm.weight")
}

// cloneF64 copies a tensor into a fresh F64 tensor of the same shape (widening from
// any dtype). This is exactly Cast to F64, which is the §base-perf fast typed path.
func cloneF64(t *tensor.Tensor) *tensor.Tensor {
	return t.Cast(tensor.F64)
}
