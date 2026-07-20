package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GGUF MoE metadata keys (gguf-py constants.py Keys.LLM.EXPERT_COUNT /
// EXPERT_USED_COUNT). Arch-prefixed like the rest: "llama.expert_count".
const (
	ggufExpertCnt  = "expert_count"
	ggufExpertUsed = "expert_used_count"
)

// MixtralFromGGUF builds a [Mixtral] from the metadata and (dequantized) tensor
// maps of a parsed GGUF file (gguf.File.Metadata / .Tensors). Mixtral GGUFs carry
// general.architecture "llama" — llama.cpp has no separate mixtral arch;
// MixtralForCausalLM is registered on the llama converter (conversion/llama.py
// LlamaModel) — and are distinguished by the MoE metadata (llama.expert_count,
// llama.expert_used_count) and the per-block expert tensors. This loader
// therefore requires llama.expert_count; a dense llama file belongs to
// [LlamaFromGGUF] instead.
//
// The Mixtral conventions this loader implements were verified against llama.cpp
// master (conversion/llama.py + gguf-py tensor_mapping/constants + the rope-type
// switch in src/llama-model.cpp):
//
//   - Q/K rows ARE permuted on disk: LlamaModel has undo_permute = True, so the
//     converter reorders attn_q/attn_k rows per head from HF's split-half layout
//     into the interleaved pairing (LlamaModel.permute:
//     reshape(head, 2, hd/2, in).swapaxes(1, 2)), which LLM_ARCH_LLAMA consumes
//     with LLAMA_ROPE_TYPE_NORM ("pairs of consecutive head values"). GoAI's
//     OpRoPE is the split-half (HF rotate_half) convention, so this loader
//     UN-permutes attn_q/attn_k rows back into the HF layout
//     (dst[head·hd + j·hd/2 + i] = src[head·hd + 2i + j]) before transposing —
//     the exact inverse of the converter. Skipping the un-permute would show a
//     rotary-sized divergence against the [MixtralFromHF] parity anchor.
//   - Experts are FUSED 3-D: per-expert HF w1/w2/w3 are torch.stack'ed on dim 0
//     and mapped (tensor_mapping: "layers.N.feed_forward.experts.w1/w2/w3") to
//     blk.N.ffn_gate_exps.weight [E, ffn, dim], blk.N.ffn_down_exps.weight
//     [E, dim, ffn] and blk.N.ffn_up_exps.weight [E, ffn, dim]; the router
//     (HF block_sparse_moe.gate) is blk.N.ffn_gate_inp.weight [E, dim]. Each
//     expert slice loads into nn.SparseMoE exactly as [MixtralFromHF] unpacks the
//     HF fused layout, transposed from torch [out, in] into GoAI [in, out].
//
// The config comes from the llama.* metadata keys; Hidden is the per-expert
// SwiGLU width (llama.feed_forward_length = HF intermediate_size), cross-checked
// against the exps tensor shapes. llama.expert_used_count is optional (default
// top-2, Mixtral's value). An absent output.weight ties the LM head to
// token_embd. Optional blk.N.attn_{q,k}_norm.weight gains (a GoAI extension for
// round-tripping Qwen3-MoE-style QK-norm models — real llama-arch files never
// carry them) load into QNorm/KNorm, nil otherwise.
func MixtralFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Mixtral, error) {
	const arch = "llama"
	lcfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	experts, err := metaInt(meta, arch+"."+ggufExpertCnt)
	if err != nil {
		return nil, fmt.Errorf("nlp: not a Mixtral (MoE) GGUF — dense llama files load via LlamaFromGGUF: %w", err)
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

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Mixtral{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		names := []string{"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
			"attn_output.weight", "ffn_norm.weight",
			"ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}
		w := make([]*tensor.Tensor, len(names))
		for i, n := range names {
			if w[i], err = g(n); err != nil {
				return nil, err
			}
		}
		// Rank-guard the 2-D tensors this block feeds to ropeUnpermuteRows/transpose2D:
		// the attention projections attn_q/k/v/output (w[1..4]) and the MoE router
		// ffn_gate_inp (w[6]). The fused expert banks (w[7..9]) are 3-D and are shape-
		// checked inside mixtralMoEFromGGUF; the norm gains (w[0], w[5]) are 1-D. Mirrors
		// QuantMixtralFromGGUF's mkQ/mkQPermuted `len(qt.Shape) != 2` (§B77).
		for _, i := range []int{1, 2, 3, 4, 6} {
			if err := require2D(p+names[i], w[i]); err != nil {
				return nil, err
			}
		}
		moe, err := mixtralMoEFromGGUF(cfg, w[6], w[7], w[8], w[9], p)
		if err != nil {
			return nil, err
		}
		qkNorm := func(name string) *nn.RMSNorm {
			if t, ok := tensors[p+name]; ok {
				return rmsFromGGUF(t, cfg.Eps)
			}
			return nil
		}
		m.Blocks = append(m.Blocks, &MixtralBlock{
			AttnNorm: rmsFromGGUF(w[0], cfg.Eps),
			// GGUF llama-arch q/k are stored interleaved-permuted; restore the HF
			// split-half row order for GoAI's OpRoPE, then [out,in] → [in,out].
			Wq:      transpose2D(ropeUnpermuteRows(w[1], cfg.Heads)),
			Wk:      transpose2D(ropeUnpermuteRows(w[2], cfg.kvHeads())),
			Wv:      transpose2D(w[3]),
			Wo:      transpose2D(w[4]),
			QNorm:   qkNorm("attn_q_norm.weight"),
			KNorm:   qkNorm("attn_k_norm.weight"),
			FFNNorm: rmsFromGGUF(w[5], cfg.Eps),
			MoE:     moe,
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
	m.Out = transpose2D(head)
	return m, nil
}

// mixtralMoEFromGGUF builds one block's nn.SparseMoE from the GGUF router
// ([E, dim]) and fused expert tensors (gate/up [E, ffn, dim], down [E, dim, ffn]).
func mixtralMoEFromGGUF(cfg MixtralConfig, router, gate, up, down *tensor.Tensor, p string) (*nn.SparseMoE, error) {
	if gate.Ndim() != 3 || up.Ndim() != 3 || down.Ndim() != 3 {
		return nil, fmt.Errorf("nlp: Mixtral GGUF %sffn_*_exps must be 3-D, got %v / %v / %v", p, gate.Shape(), up.Shape(), down.Shape())
	}
	if e := gate.Shape()[0]; e != cfg.Experts || up.Shape()[0] != cfg.Experts || down.Shape()[0] != cfg.Experts || router.Shape()[0] != cfg.Experts {
		return nil, fmt.Errorf("nlp: Mixtral GGUF %s expert tensors disagree with llama.expert_count=%d (gate %v)", p, cfg.Experts, gate.Shape())
	}
	if ffn := gate.Shape()[1]; ffn != cfg.Hidden {
		return nil, fmt.Errorf("nlp: Mixtral GGUF %sffn_gate_exps width %d != llama.feed_forward_length %d", p, ffn, cfg.Hidden)
	}
	moe := nn.NewSparseMoE(tensor.F64, cfg.Dim, cfg.Hidden, cfg.Experts, cfg.TopK, 0)
	moe.Router.W = transpose2D(router) // [E,dim] → [dim,E]
	for e := range cfg.Experts {
		moe.Experts[e].Wgate = transpose2D(sub3D(gate, e)) // [ffn,dim] → [dim,ffn]
		moe.Experts[e].Wup = transpose2D(sub3D(up, e))
		moe.Experts[e].Wdown = transpose2D(sub3D(down, e)) // [dim,ffn] → [ffn,dim]
	}
	return moe, nil
}

// MixtralToGGUF is the inverse of [MixtralFromGGUF]: it serializes a Mixtral into
// GGUF metadata + tensor maps under general.architecture "llama" (the Mixtral
// convention — there is no mixtral arch string) with llama.expert_count /
// llama.expert_used_count, PERMUTING attn_q/attn_k rows into the interleaved
// llama-arch layout exactly as llama.cpp's converter does (LlamaModel.permute,
// undo_permute=True) and stacking the experts into the fused 3-D
// blk.N.ffn_{gate,up,down}_exps.weight tensors plus the blk.N.ffn_gate_inp.weight
// router. Every projection is transposed back into torch [out, in]. Optional
// QNorm/KNorm gains are emitted as blk.N.attn_{q,k}_norm.weight when non-nil (a
// GoAI round-trip extension; plain Mixtral writes none). Pass the result to
// gguf.Write via a gguf.File.
func MixtralToGGUF(m *Mixtral) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "llama"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	topk := c.TopK
	if topk <= 0 {
		topk = 2
	}
	meta := map[string]any{
		ggufArch:            arch,
		key(ggufEmbLen):     uint32(c.Dim),
		key(ggufBlockCnt):   uint32(c.Layers),
		key(ggufFFLen):      uint32(c.Hidden),
		key(ggufHeadCnt):    uint32(c.Heads),
		key(ggufHeadKV):     uint32(c.kvHeads()),
		key(ggufCtxLen):     uint32(c.Ctx),
		key(ggufRMSEps):     float32(c.Eps),
		key(ggufRopeFreq):   float32(ropeBaseOr(c.RopeBase)),
		key(ggufExpertCnt):  uint32(c.Experts),
		key(ggufExpertUsed): uint32(topk),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim]
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.AttnNorm.Gamma)
		// GoAI [in,out] → torch [out,in], then the converter's HF→interleaved permute.
		ts[p+"attn_q.weight"] = ropePermuteRows(transpose2D(b.Wq), c.Heads)
		ts[p+"attn_k.weight"] = ropePermuteRows(transpose2D(b.Wk), c.kvHeads())
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_norm.weight"] = cloneF64(b.FFNNorm.Gamma)
		ts[p+"ffn_gate_inp.weight"] = transpose2D(b.MoE.Router.W) // [dim,E] → [E,dim]
		gates := make([]*tensor.Tensor, len(b.MoE.Experts))
		ups := make([]*tensor.Tensor, len(b.MoE.Experts))
		downs := make([]*tensor.Tensor, len(b.MoE.Experts))
		for e, ex := range b.MoE.Experts {
			gates[e] = transpose2D(ex.Wgate) // [dim,ffn] → [ffn,dim]
			ups[e] = transpose2D(ex.Wup)
			downs[e] = transpose2D(ex.Wdown) // [ffn,dim] → [dim,ffn]
		}
		ts[p+"ffn_gate_exps.weight"] = stack3D(gates)
		ts[p+"ffn_up_exps.weight"] = stack3D(ups)
		ts[p+"ffn_down_exps.weight"] = stack3D(downs)
		if b.QNorm != nil {
			ts[p+"attn_q_norm.weight"] = cloneF64(b.QNorm.Gamma)
		}
		if b.KNorm != nil {
			ts[p+"attn_k_norm.weight"] = cloneF64(b.KNorm.Gamma)
		}
	}
	return meta, ts
}

// ropePermuteRows applies llama.cpp's HF→GGUF q/k row permutation for the llama
// architecture (conversion/llama.py LlamaModel.permute:
// reshape(heads, 2, hd/2, in).swapaxes(1, 2).reshape(...)): within each head,
// the row at split-half position j·hd/2 + i moves to interleaved position 2i + j.
// Rows = heads·hd; hd must be even (RoPE rotates pairs).
func ropePermuteRows(t *tensor.Tensor, heads int) *tensor.Tensor {
	return gatherRows(t, heads, func(hd, q int) int {
		// dst interleaved position q = 2i + j  ←  src split-half position j·hd/2 + i
		i, j := q/2, q%2
		return j*(hd/2) + i
	})
}

// ropeUnpermuteRows is the exact inverse of [ropePermuteRows]: it restores the HF
// split-half row order from the interleaved on-disk llama-arch layout, so GoAI's
// split-half OpRoPE sees the layout it expects.
func ropeUnpermuteRows(t *tensor.Tensor, heads int) *tensor.Tensor {
	return gatherRows(t, heads, func(hd, q int) int {
		// dst split-half position q = j·hd/2 + i  ←  src interleaved position 2i + j
		j, i := q/(hd/2), q%(hd/2)
		return 2*i + j
	})
}

// ropeRowGeom pins a projection weight's geometry BEFORE any of the float loaders'
// RoPE row permutes touch it, so a malformed file is a clean error instead of a panic
// or a silently mis-permuted model (§V29).
//
// The permutes it guards — [ropeUnpermuteRows] (llama/granite) and
// [permuteInterleaveToSplit] (command-r) — never inspect the tensor they are handed.
// Three things go wrong when the row count disagrees with the head count, all reachable
// from an untrusted file because the tensor's rows are attacker-controlled SEPARATELY
// from the metadata that declares them:
//
//   - a NON-2-D tensor dies reading Shape()[1] (or trips gatherRows' internal
//     2-D panic, which is deliberately not an error path);
//   - TOO FEW rows index out of range;
//   - TOO MANY rows permute only a prefix and leave the tail unwritten — the model
//     then loads with err == nil and computes wrong attention, the wrong-but-nil-error
//     failure §V29 calls out as the harder one to trace.
//
// The predicate is not restated here: it is [ropeUnpermutePerm], the SAME function
// every quantized twin already gates on (quant_mixtral_gguf.go's mkQPermuted,
// quant_granite_gguf.go's unpermuteQuantRows, quant_cohere_gguf.go's mkQPermuted), so
// the float and quantized loaders accept and reject exactly the same set of files.
// That equality is the property worth having — a second, independently-worded check
// would drift. Callers that additionally know the EXPECTED row count keep their own
// explicit check on top; this one is the floor.
func ropeRowGeom(name string, w *tensor.Tensor, heads int) error {
	if w.Ndim() != 2 {
		return fmt.Errorf("nlp: GGUF %s is %d-D, want a 2-D [out, in] projection", name, w.Ndim())
	}
	if _, err := ropeUnpermutePerm(w.Shape()[0], heads); err != nil {
		return fmt.Errorf("nlp: GGUF %s: %w", name, err)
	}
	return nil
}

// ropeVecGeom is [ropeRowGeom] for the 1-D q/k BIAS the llama lineage permutes with
// [ropeUnpermuteVec] — same predicate, rank 1 instead of rank 2.
func ropeVecGeom(name string, v *tensor.Tensor, heads int) error {
	if v.Ndim() != 1 {
		return fmt.Errorf("nlp: GGUF %s is %d-D, want a 1-D bias", name, v.Ndim())
	}
	if _, err := ropeUnpermutePerm(v.Shape()[0], heads); err != nil {
		return fmt.Errorf("nlp: GGUF %s: %w", name, err)
	}
	return nil
}

// gatherRows reorders the rows of a [heads·hd, cols] tensor per head: dst row
// h·hd + q is copied from src row h·hd + srcPos(hd, q). Returns a fresh F64
// tensor; the per-head row width hd must be even.
//
// The geometry check is an INTERNAL invariant, so it panics rather than returning
// an error: every caller reaching this point has already validated heads against
// the tensor's row count, so a violation here is a GoAI bug, not a bad input file.
// Untrusted metadata is rejected upstream with a clean error instead — see the
// positive-count validation in llamaCfgFromGGUFMeta and the per-tensor ropeGeom
// check in [LlamaFromGGUF] (§V29).
//
// heads is tested BEFORE the division that derives hd: the old order computed
// `rows / heads` first, so a head_count of 0 from a malformed GGUF crashed with an
// integer divide-by-zero one line before its own heads<=0 guard could report it.
func gatherRows(t *tensor.Tensor, heads int, srcPos func(hd, q int) int) *tensor.Tensor {
	if t.Ndim() != 2 {
		panic(fmt.Sprintf("nlp: rope row permute needs a 2-D tensor, got %d-D", t.Ndim()))
	}
	rows, cols := t.Shape()[0], t.Shape()[1]
	if heads <= 0 || rows%heads != 0 || (rows/heads)%2 != 0 {
		panic(fmt.Sprintf("nlp: rope row permute needs rows %d divisible by heads %d with even head_dim", rows, heads))
	}
	hd := rows / heads
	src := cloneF64(t).Storage().F64()
	out := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	dst := out.Storage().F64()
	for h := range heads {
		base := h * hd
		for q := range hd {
			copy(dst[(base+q)*cols:(base+q+1)*cols], src[(base+srcPos(hd, q))*cols:(base+srcPos(hd, q)+1)*cols])
		}
	}
	return out
}

// ropePermuteVec / ropeUnpermuteVec are the 1-D (bias) twins of [ropePermuteRows] /
// [ropeUnpermuteRows]: llama.cpp's converter applies LlamaModel.permute to
// q_proj.bias and k_proj.bias exactly as to the weights (conversion/llama.py
// modify_tensors matches name.endswith(("q_proj.weight", "q_proj.bias"))), so a
// bias-carrying llama-arch file stores its q/k biases in the same interleaved rotary
// row order as the projections (§B67 follow-up).
func ropePermuteVec(t *tensor.Tensor, heads int) *tensor.Tensor {
	return gatherVec(t, heads, func(hd, q int) int {
		i, j := q/2, q%2
		return j*(hd/2) + i
	})
}

func ropeUnpermuteVec(t *tensor.Tensor, heads int) *tensor.Tensor {
	return gatherVec(t, heads, func(hd, q int) int {
		j, i := q/(hd/2), q%(hd/2)
		return 2*i + j
	})
}

// gatherVec is [gatherRows] for a 1-D per-head vector: dst element h·hd + q is copied
// from src element h·hd + srcPos(hd, q). Returns a fresh F64 tensor; the per-head
// width hd must be even.
//
// Like [gatherRows] this panic is an internal invariant (callers validate first), and
// heads is likewise tested BEFORE the division that derives hd — a head_count of 0
// otherwise divided by zero one line ahead of the guard meant to catch it.
func gatherVec(t *tensor.Tensor, heads int, srcPos func(hd, q int) int) *tensor.Tensor {
	if t.Ndim() != 1 {
		panic(fmt.Sprintf("nlp: rope vec permute needs a 1-D tensor, got %d-D", t.Ndim()))
	}
	n := t.Shape()[0]
	if heads <= 0 || n%heads != 0 || (n/heads)%2 != 0 {
		panic(fmt.Sprintf("nlp: rope vec permute needs length %d divisible by heads %d with even head_dim", n, heads))
	}
	hd := n / heads
	src := cloneF64(t).Storage().F64()
	out := tensor.New(tensor.F64, tensor.Shape{n})
	dst := out.Storage().F64()
	for h := range heads {
		base := h * hd
		for q := range hd {
			dst[base+q] = src[base+srcPos(hd, q)]
		}
	}
	return out
}

// stack3D stacks E equal-shape [r, c] tensors into a fresh F64 [E, r, c] tensor —
// the inverse of sub3D, mirroring the converter's torch.stack(dim=0).
func stack3D(parts []*tensor.Tensor) *tensor.Tensor {
	r, c := parts[0].Shape()[0], parts[0].Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{len(parts), r, c})
	dst := out.Storage().F64()
	for e, p := range parts {
		copy(dst[e*r*c:(e+1)*r*c], cloneF64(p).Storage().F64())
	}
	return out
}
