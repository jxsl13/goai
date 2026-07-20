package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GGUF metadata keys used by the deepseek2 architecture beyond the shared llama-family
// set (gguf-py constants.py Keys.LLM / Keys.Attention / Keys.Rope). Arch-prefixed like
// the rest: "deepseek2.attention.q_lora_rank".
const (
	ggufVocabSize  = "vocab_size"                       // Keys.LLM.VOCAB_SIZE
	ggufLeadDense  = "leading_dense_block_count"        // Keys.LLM.LEADING_DENSE_BLOCK_COUNT
	ggufExpFFLen   = "expert_feed_forward_length"       // Keys.LLM.EXPERT_FEED_FORWARD_LENGTH
	ggufExpShared  = "expert_shared_count"              // Keys.LLM.EXPERT_SHARED_COUNT
	ggufExpWScale  = "expert_weights_scale"             // Keys.LLM.EXPERT_WEIGHTS_SCALE
	ggufExpWNorm   = "expert_weights_norm"              // Keys.LLM.EXPERT_WEIGHTS_NORM
	ggufExpGating  = "expert_gating_func"               // Keys.LLM.EXPERT_GATING_FUNC (1=softmax, 2=sigmoid)
	ggufQLoraRank  = "attention.q_lora_rank"            // Keys.Attention.Q_LORA_RANK
	ggufKVLoraRank = "attention.kv_lora_rank"           // Keys.Attention.KV_LORA_RANK
	ggufKeyLenMLA  = "attention.key_length_mla"         // Keys.Attention.KEY_LENGTH_MLA
	ggufValLenMLA  = "attention.value_length_mla"       // Keys.Attention.VALUE_LENGTH_MLA
	ggufScaleType  = "rope.scaling.type"                // Keys.Rope.SCALING_TYPE
	ggufYarnLogMul = "rope.scaling.yarn_log_multiplier" // Keys.Rope.SCALING_YARN_LOG_MUL
)

// DeepSeekV2FromGGUF builds a [DeepSeekV2] from the metadata and (dequantized) tensor
// maps of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture
// is "deepseek2" — the llama.cpp convention for DeepseekV2ForCausalLM checkpoints. The
// MLA geometry comes from the deepseek2.* metadata keys, the weights from the
// token_embd / blk.N.* / output tensors; every projection is transposed from GGUF's
// torch [out, in] layout into GoAI's [in, out].
//
// The deepseek2 conventions this loader implements were verified against llama.cpp
// master (conversion/deepseek.py DeepseekV2Model + conversion/base.py, src/models/
// deepseek2.cpp, src/llama-arch.cpp, src/llama-model.cpp, src/llama-hparams.cpp and
// gguf-py/gguf/constants.py + tensor_mapping.py):
//
//   - MLA tensor map (llama-arch.cpp / tensor_mapping.py): q_a_proj → blk.N.attn_q_a,
//     q_a_layernorm → attn_q_a_norm, q_b_proj → attn_q_b, kv_a_proj_with_mqa →
//     attn_kv_a_mqa, kv_a_layernorm → attn_kv_a_norm. When config.q_lora_rank is null
//     (DeepSeek-V2-Lite) the converter writes a plain blk.N.attn_q instead of the
//     q_a/q_b pair and omits deepseek2.attention.q_lora_rank — a form GoAI's
//     [DeepSeekV2] cannot represent (its query path is fixed to
//     q_b(q_a_layernorm(q_a(h))), see [DeepSeekV2FromHF]), so such files are REJECTED
//     with a precise error rather than misloaded.
//   - kv_b is SPLIT ON DISK by the current converter: DeepseekV2Model.modify_tensors
//     never writes kv_b_proj — it views it per head as [qk_nope | v] row blocks and
//     yields k_b_proj TRANSPOSED per head (k_b.transpose(1, 2), ready for the
//     absorption matmul) plus v_b_proj: blk.N.attn_k_b [heads, kv_lora, qk_nope] and
//     blk.N.attn_v_b [heads, v_head, kv_lora] (numpy order; ggml logs the reversed ne).
//     Nothing is derived at load: llama.cpp requires the split pair exactly when the
//     *_mla metadata keys are set (hparams.is_mla(), llama-hparams.cpp) and only "old
//     legacy GGUF files" carry the unsplit blk.N.attn_kv_b
//     [heads·(qk_nope+v_head), kv_lora] (src/models/deepseek2.cpp load_arch_tensors
//     note). This loader mirrors both: split when key/value_length_mla are present,
//     legacy unsplit otherwise, both reconstructing the fused WkvB — and rejects a
//     file carrying both forms (llama.cpp would fail its all-tensors-used check).
//   - The head count in modify_tensors is the ORIGINAL num_key_value_heads (= n_head
//     for DeepSeek): conversion/base.py write() runs prepare_tensors() BEFORE
//     prepare_metadata()/set_gguf_parameters(), and only the latter forces
//     num_key_value_heads = 1 — so deepseek2.attention.head_count_kv is 1 on disk
//     ("MLA converts into MQA") while attn_k_b/attn_v_b carry all heads.
//   - Metadata semantics (conversion/deepseek.py set_gguf_parameters):
//     attention.key_length = kv_lora_rank + qk_rope_head_dim and
//     attention.value_length = kv_lora_rank are the MQA CACHE widths, NOT head dims;
//     the true MLA head dims are attention.key_length_mla = qk_nope + qk_rope and
//     attention.value_length_mla = v_head_dim, with rope.dimension_count =
//     qk_rope_head_dim (so QKNope = key_length_mla − rope.dimension_count). The
//     latent ranks are attention.q_lora_rank / attention.kv_lora_rank.
//   - NO q/k rope permute, interleaved rotary: DeepseekV2Model.modify_tensors leaves
//     q_a/q_b/kv_a_proj_with_mqa untouched (contrast LlamaModel.permute), and
//     LLM_ARCH_DEEPSEEK2 sits in the LLAMA_ROPE_TYPE_NORM case of
//     llama_model_rope_type (llama-model.cpp) — rope on pairs of CONSECUTIVE values,
//     matching HF DeepseekV2's view_as_complex-style shuffle. GoAI's OpRoPE is
//     split-half, so this loader de-interleaves the pe rows of attn_q_b (per head)
//     and attn_kv_a_mqa (the shared k_pe block) exactly like [DeepSeekV2FromHF]
//     (see [deinterleaveRoPE]).
//   - MoE (src/models/deepseek2.cpp): layers with index < deepseek2.
//     leading_dense_block_count (config.first_k_dense_replace; the converter writes
//     n_layer for MoE-free checkpoints) use dense blk.N.ffn_{gate,up,down}; later
//     layers use the router blk.N.ffn_gate_inp [E, dim], the fused 3-D expert bank
//     blk.N.ffn_{gate,up}_exps [E, moe_ffn, dim] / ffn_down_exps [E, dim, moe_ffn]
//     (torch.stack over mlp.experts.J.*, conversion/deepseek.py merge_expert) and the
//     FUSED shared expert blk.N.ffn_{gate,up,down}_shexp of width
//     expert_shared_count · expert_feed_forward_length. Routing is softmax over all
//     deepseek2.expert_count experts, greedy top expert_used_count, weights scaled by
//     deepseek2.expert_weights_scale (routed_scaling_factor; absent → 1.0) WITHOUT
//     renormalization — exactly [DeepSeekV2.moeFFN]. Files declaring
//     expert_weights_norm = true or expert_gating_func ≠ softmax (a DeepSeek-V3-style
//     sigmoid router, which also carries blk.N.exp_probs_b.bias) are REJECTED: GoAI's
//     DeepSeekV2 implements only V2's un-normalized softmax gating.
//   - YaRN rejected: DeepSeek-V2 release checkpoints use YaRN rope scaling, which the
//     converter writes as deepseek2.rope.scaling.* plus rope.scaling.
//     yarn_log_multiplier (0.1·mscale_all_dim) and llama.cpp folds into a pre-scaled
//     kq_scale (src/models/deepseek2.cpp). GoAI's DeepSeekV2Config has no YaRN form,
//     so a present scaling type other than "none" — or any yarn_log_multiplier — is
//     rejected rather than silently decoded with the wrong frequencies/scale. Without
//     YaRN the graph's kq_scale reduces to 1/√(key_length_mla), GoAI's own
//     softmaxScale default.
//   - Untied head optional: output.weight falls back to token_embd
//     (TENSOR_DUPLICATED in load_arch_tensors), and the RMS epsilon key is
//     attention.layer_norm_rms_epsilon (the converter defaults rms_norm_eps to 1e-6).
func DeepSeekV2FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*DeepSeekV2, error) {
	cfg, isMLA, err := deepseekV2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &DeepSeekV2{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		if _, ok := tensors[p+"attn_q.weight"]; ok {
			return nil, fmt.Errorf("nlp: deepseek2 GGUF carries %sattn_q.weight (a no-q-lora file, e.g. DeepSeek-V2-Lite); GoAI's DeepSeekV2 supports only the compressed attn_q_a/attn_q_b query path", p)
		}
		wqA, err := g("attn_q_a.weight")
		if err != nil {
			return nil, err
		}
		qaNorm, err := g("attn_q_a_norm.weight")
		if err != nil {
			return nil, err
		}
		wqB, err := g("attn_q_b.weight")
		if err != nil {
			return nil, err
		}
		wkvA, err := g("attn_kv_a_mqa.weight")
		if err != nil {
			return nil, err
		}
		kvaNorm, err := g("attn_kv_a_norm.weight")
		if err != nil {
			return nil, err
		}
		kvB, err := deepseekV2KvB(tensors, p, cfg, isMLA) // fused torch [heads·(QKNope+VHead), KVLoraRank]
		if err != nil {
			return nil, err
		}
		wo, err := g("attn_output.weight")
		if err != nil {
			return nil, err
		}
		attnNorm, err := g("attn_norm.weight")
		if err != nil {
			return nil, err
		}
		ffnNorm, err := g("ffn_norm.weight")
		if err != nil {
			return nil, err
		}
		// FFN sublayer: dense SwiGLU for the first leading_dense_block_count layers,
		// else the DeepSeekMoE (router + fused expert bank + fused shared expert).
		var dense *nn.SwiGLU
		var moe *nn.DeepSeekMoE
		if l < cfg.FirstKDense || cfg.NRoutedExperts <= 0 {
			gate, err := g("ffn_gate.weight")
			if err != nil {
				return nil, err
			}
			up, err := g("ffn_up.weight")
			if err != nil {
				return nil, err
			}
			down, err := g("ffn_down.weight")
			if err != nil {
				return nil, err
			}
			// Rank-guard the dense FFN weights before swiGLUFromGGUF transposes them —
			// the float twin of QuantDeepSeekV2FromGGUF's mkQ `len(qt.Shape) != 2` (§B77).
			if err := require2DEach(
				ggufWeight{p + "ffn_gate.weight", gate}, ggufWeight{p + "ffn_up.weight", up}, ggufWeight{p + "ffn_down.weight", down},
			); err != nil {
				return nil, err
			}
			dense = swiGLUFromGGUF(gate, up, down)
		} else {
			if moe, err = deepseekV2MoEFromGGUF(tensors, p, cfg); err != nil {
				return nil, err
			}
		}
		// wqB/wkvA are rank-checked by deinterleaveRoPEChecked below and kvB is built
		// internally; wqA and wo reach transpose2D unchecked — the float twin of
		// QuantDeepSeekV2FromGGUF's mkQ `len(qt.Shape) != 2` (§B77).
		if err := require2DEach(ggufWeight{p + "attn_q_a.weight", wqA}, ggufWeight{p + "attn_output.weight", wo}); err != nil {
			return nil, err
		}
		// De-interleave the pe rows (interleaved on disk, LLAMA_ROPE_TYPE_NORM) into
		// split-half order for GoAI's OpRoPE — the same permutation as DeepSeekV2FromHF.
		qbPerm, err := deinterleaveRoPEChecked(p+"attn_q_b.weight", wqB, cfg.Heads, cfg.QKNope+cfg.QKRope, cfg.QKNope, cfg.QKRope)
		if err != nil {
			return nil, err
		}
		kvaPerm, err := deinterleaveRoPEChecked(p+"attn_kv_a_mqa.weight", wkvA, 1, cfg.KVLoraRank+cfg.QKRope, cfg.KVLoraRank, cfg.QKRope)
		if err != nil {
			return nil, err
		}

		m.Blocks = append(m.Blocks, &DeepSeekV2Block{
			InputNorm:    rmsFromGGUF(attnNorm, cfg.Eps),
			WqA:          transpose2D(wqA),
			QANorm:       rmsFromGGUF(qaNorm, cfg.Eps),
			WqB:          transpose2D(qbPerm),
			WkvA:         transpose2D(kvaPerm),
			KvANorm:      rmsFromGGUF(kvaNorm, cfg.Eps),
			WkvB:         transpose2D(kvB),
			Wo:           transpose2D(wo),
			PostAttnNorm: rmsFromGGUF(ffnNorm, cfg.Eps),
			Dense:        dense,
			MoE:          moe,
		})
	}
	norm, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.FinalNorm = rmsFromGGUF(norm, cfg.Eps)
	head, headName := tok, "token_embd.weight (tied LM head)" // tied when output.weight is absent (TENSOR_DUPLICATED)
	if o, ok := tensors["output.weight"]; ok {
		head, headName = o, "output.weight"
	}
	if err := require2D(headName, head); err != nil { // guards transpose2D below (§B77)
		return nil, err
	}
	m.LmHead = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// deepseekV2CfgFromGGUFMeta parses the deepseek2.* metadata of a GGUF file whose
// general.architecture is "deepseek2" into a DeepSeekV2Config, reporting whether the
// file is the modern MLA-split form (key/value_length_mla present → split
// attn_k_b/attn_v_b tensors) or the legacy unsplit form (attn_kv_b, with
// attention.key_length/value_length holding the PER-HEAD dims as older converters
// wrote them). Vocab is left for [DeepSeekV2FromGGUF] (token_embd-derived). Files GoAI's
// DeepSeekV2 cannot represent are rejected here: missing q_lora_rank (the V2-Lite plain-q
// form), YaRN rope scaling, renormalized gate weights and non-softmax gating functions.
func deepseekV2CfgFromGGUFMeta(meta map[string]any) (cfg DeepSeekV2Config, isMLA bool, err error) {
	const arch = "deepseek2"
	if a, _ := meta[ggufArch].(string); a != arch {
		return cfg, false, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }

	// YaRN (DeepSeek-V2's release scaling) is not representable in DeepSeekV2Config.
	if st, ok := meta[key(ggufScaleType)].(string); ok && st != "none" {
		return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF has %s=%q; GoAI's DeepSeekV2 supports only unscaled RoPE", key(ggufScaleType), st)
	}
	if _, ok := meta[key(ggufYarnLogMul)]; ok {
		return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF has %s (YaRN mscale); GoAI's DeepSeekV2 supports only unscaled RoPE", key(ggufYarnLogMul))
	}

	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return cfg, false, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return cfg, false, err
	}
	ffn, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return cfg, false, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return cfg, false, err
	}
	qLora, err := metaInt(meta, key(ggufQLoraRank))
	if err != nil {
		return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF has no %s (a no-q-lora file, e.g. DeepSeek-V2-Lite); GoAI's DeepSeekV2 supports only the q_lora-compressed query path: %w", key(ggufQLoraRank), err)
	}
	kvLora, err := metaInt(meta, key(ggufKVLoraRank))
	if err != nil {
		return cfg, false, err
	}
	ropeDim, err := metaInt(meta, key(ggufRopeDim))
	if err != nil {
		return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF needs %s (= qk_rope_head_dim): %w", key(ggufRopeDim), err)
	}

	// Head geometry: modern files carry the true MLA head dims under the *_mla keys
	// (attention.key_length/value_length are the MQA cache widths kv_lora+rope /
	// kv_lora); legacy unsplit files carry them under key_length/value_length directly.
	kMLA, kErr := metaInt(meta, key(ggufKeyLenMLA))
	vMLA, vErr := metaInt(meta, key(ggufValLenMLA))
	switch {
	case kErr == nil && vErr == nil:
		isMLA = true
	case kErr == nil || vErr == nil:
		return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF carries only one of %s/%s; llama.cpp requires both or neither", key(ggufKeyLenMLA), key(ggufValLenMLA))
	default:
		if kMLA, err = metaInt(meta, key(ggufKeyLen)); err != nil {
			return cfg, false, fmt.Errorf("nlp: legacy deepseek2 GGUF needs %s (= qk_nope+qk_rope): %w", key(ggufKeyLen), err)
		}
		if vMLA, err = metaInt(meta, key(ggufValLen)); err != nil {
			return cfg, false, fmt.Errorf("nlp: legacy deepseek2 GGUF needs %s (= v_head_dim): %w", key(ggufValLen), err)
		}
	}
	if kMLA <= ropeDim {
		return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF MLA key length %d must exceed rope.dimension_count %d", kMLA, ropeDim)
	}

	cfg = DeepSeekV2Config{
		Dim: dim, Layers: layers, FFN: ffn, Heads: heads,
		QLoraRank: qLora, KVLoraRank: kvLora,
		QKNope: kMLA - ropeDim, QKRope: ropeDim, VHead: vMLA,
		Eps:      metaFloat(meta, key(ggufRMSEps), 1e-6),
		RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000),
		Ctx:      dim, // provisional; overwritten from context_length below
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}

	// MoE geometry. A file without expert_count is fully dense (the converter writes
	// leading_dense_block_count = n_layer for such checkpoints).
	cfg.FirstKDense = cfg.Layers
	if experts, e := metaInt(meta, key(ggufExpertCnt)); e == nil && experts > 0 {
		if norm, ok := meta[key(ggufExpWNorm)].(bool); ok && norm {
			return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF has %s=true; GoAI's DeepSeekV2 implements V2's UN-normalized top-k gating", key(ggufExpWNorm))
		}
		if gf, e := metaInt(meta, key(ggufExpGating)); e == nil && gf != 1 {
			return cfg, false, fmt.Errorf("nlp: deepseek2 GGUF has %s=%d; GoAI's DeepSeekV2 implements only softmax (1) gating, not the DeepSeek-V3 sigmoid router", key(ggufExpGating), gf)
		}
		cfg.NRoutedExperts = experts
		if cfg.TopK, err = metaInt(meta, key(ggufExpertUsed)); err != nil {
			return cfg, false, err
		}
		if cfg.MoEHidden, err = metaInt(meta, key(ggufExpFFLen)); err != nil {
			return cfg, false, err
		}
		if cfg.NSharedExperts, err = metaInt(meta, key(ggufExpShared)); err != nil {
			return cfg, false, err
		}
		cfg.RoutedScale = metaFloat(meta, key(ggufExpWScale), 1.0) // absent → llama.cpp scale_w=false ≡ 1.0
		cfg.FirstKDense = 0
		if lead, e := metaInt(meta, key(ggufLeadDense)); e == nil {
			cfg.FirstKDense = lead
		}
	}
	return cfg, isMLA, nil
}

// deepseekV2KvB reads one block's kv_b projection in whichever form the file carries
// and returns it FUSED in torch layout [heads·(QKNope+VHead), KVLoraRank] — the shape
// [DeepSeekV2FromHF] receives from HF, ready for transpose2D. Modern (isMLA) files
// carry the converter's split pair — attn_k_b [heads, KVLoraRank, QKNope] (per-head
// TRANSPOSED by DeepseekV2Model.modify_tensors for the absorption matmul) and attn_v_b
// [heads, VHead, KVLoraRank] — which is un-transposed and re-fused here; legacy files
// carry the unsplit attn_kv_b. A file carrying both is rejected (llama.cpp loads
// exactly one set and fails its all-tensors-used check otherwise).
func deepseekV2KvB(tensors map[string]*tensor.Tensor, p string, cfg DeepSeekV2Config, isMLA bool) (*tensor.Tensor, error) {
	kb, hasKb := tensors[p+"attn_k_b.weight"]
	vb, hasVb := tensors[p+"attn_v_b.weight"]
	kvb, hasKvb := tensors[p+"attn_kv_b.weight"]
	if (hasKb || hasVb) && hasKvb {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %s carries both the split attn_k_b/attn_v_b and the legacy attn_kv_b", p)
	}
	if !isMLA {
		if !hasKvb {
			return nil, fmt.Errorf("nlp: legacy deepseek2 GGUF missing %sattn_kv_b.weight", p)
		}
		want := tensor.Shape{cfg.Heads * (cfg.QKNope + cfg.VHead), cfg.KVLoraRank}
		if !kvb.Shape().Equal(want) {
			return nil, fmt.Errorf("nlp: deepseek2 GGUF %sattn_kv_b.weight shape %v, want %v", p, kvb.Shape(), want)
		}
		return cloneF64(kvb), nil
	}
	if !hasKb || !hasVb {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF (MLA-split form) missing %sattn_k_b.weight/attn_v_b.weight", p)
	}
	if want := (tensor.Shape{cfg.Heads, cfg.KVLoraRank, cfg.QKNope}); !kb.Shape().Equal(want) {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sattn_k_b.weight shape %v, want %v (per-head transposed)", p, kb.Shape(), want)
	}
	if want := (tensor.Shape{cfg.Heads, cfg.VHead, cfg.KVLoraRank}); !vb.Shape().Equal(want) {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sattn_v_b.weight shape %v, want %v", p, vb.Shape(), want)
	}
	kvHead := cfg.QKNope + cfg.VHead
	fused := tensor.New(tensor.F64, tensor.Shape{cfg.Heads * kvHead, cfg.KVLoraRank})
	dst := fused.Storage().F64()
	rowLen := cfg.KVLoraRank
	for h := range cfg.Heads {
		// k rows: un-transpose the converter's per-head [KVLoraRank, QKNope] back to
		// [QKNope, KVLoraRank] row blocks.
		kh := transpose2D(sub3D(kb, h)) // [QKNope, KVLoraRank]
		copy(dst[(h*kvHead)*rowLen:(h*kvHead+cfg.QKNope)*rowLen], kh.Storage().F64())
		// v rows are stored untransposed: [VHead, KVLoraRank] as-is.
		vh := sub3D(vb, h)
		copy(dst[(h*kvHead+cfg.QKNope)*rowLen:(h+1)*kvHead*rowLen], vh.Storage().F64())
	}
	return fused, nil
}

// deepseekV2MoEFromGGUF builds one block's [nn.DeepSeekMoE] from the GGUF router
// (ffn_gate_inp [E, dim]), fused 3-D expert bank (ffn_{gate,up}_exps [E, moe_ffn, dim],
// ffn_down_exps [E, dim, moe_ffn]) and fused shared expert (ffn_*_shexp, width
// NSharedExperts·MoEHidden). A DeepSeek-V3-style blk.N.exp_probs_b.bias (the sigmoid
// router's score-correction bias) is rejected — GoAI's V2 gating has no such term.
func deepseekV2MoEFromGGUF(tensors map[string]*tensor.Tensor, p string, cfg DeepSeekV2Config) (*nn.DeepSeekMoE, error) {
	if _, ok := tensors[p+"exp_probs_b.bias"]; ok {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF carries %sexp_probs_b.bias (a DeepSeek-V3 score-correction bias); GoAI's DeepSeekV2 implements V2's bias-free softmax gating", p)
	}
	g := func(name string) (*tensor.Tensor, error) {
		t, ok := tensors[p+name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
		}
		return t, nil
	}
	router, err := g("ffn_gate_inp.weight")
	if err != nil {
		return nil, err
	}
	gate, err := g("ffn_gate_exps.weight")
	if err != nil {
		return nil, err
	}
	up, err := g("ffn_up_exps.weight")
	if err != nil {
		return nil, err
	}
	down, err := g("ffn_down_exps.weight")
	if err != nil {
		return nil, err
	}
	e, inter := cfg.NRoutedExperts, cfg.MoEHidden
	if gate.Ndim() != 3 || gate.Shape()[0] != e || gate.Shape()[1] != inter {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sffn_gate_exps.weight shape %v, want [%d,%d,dim]", p, gate.Shape(), e, inter)
	}
	if up.Ndim() != 3 || up.Shape()[0] != e || up.Shape()[1] != inter {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sffn_up_exps.weight shape %v, want [%d,%d,dim]", p, up.Shape(), e, inter)
	}
	if down.Ndim() != 3 || down.Shape()[0] != e || down.Shape()[2] != inter {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sffn_down_exps.weight shape %v, want [%d,dim,%d]", p, down.Shape(), e, inter)
	}
	experts := make([]*nn.SwiGLU, e)
	for j := range e {
		experts[j] = swiGLUFromGGUF(sub3D(gate, j), sub3D(up, j), sub3D(down, j))
	}

	sg, err := g("ffn_gate_shexp.weight")
	if err != nil {
		return nil, err
	}
	su, err := g("ffn_up_shexp.weight")
	if err != nil {
		return nil, err
	}
	sd, err := g("ffn_down_shexp.weight")
	if err != nil {
		return nil, err
	}
	if want := cfg.NSharedExperts * inter; sg.Shape()[0] != want {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sffn_gate_shexp.weight width %d, want expert_shared_count·expert_feed_forward_length = %d", p, sg.Shape()[0], want)
	}
	// The 2-D router and the shared-expert weights reach transpose2D/swiGLUFromGGUF; the
	// routed expert banks are 3-D (checked above). Guard them as the quantized twin does
	// with `len(qt.Shape) != 2` (§B77).
	if err := require2DEach(
		ggufWeight{p + "ffn_gate_inp.weight", router},
		ggufWeight{p + "ffn_gate_shexp.weight", sg}, ggufWeight{p + "ffn_up_shexp.weight", su}, ggufWeight{p + "ffn_down_shexp.weight", sd},
	); err != nil {
		return nil, err
	}

	return &nn.DeepSeekMoE{
		Shared: []*nn.SwiGLU{swiGLUFromGGUF(sg, su, sd)},
		Routed: &nn.SparseMoE{
			Router:  &nn.Linear{W: transpose2D(router)}, // [E,dim] → [dim,E], no bias
			Experts: experts,
			TopK:    cfg.TopK,
		},
	}, nil
}

// DeepSeekV2ToGGUF is the inverse of [DeepSeekV2FromGGUF]: it serializes a DeepSeekV2
// into GGUF metadata + tensor maps under general.architecture "deepseek2", in the
// CURRENT converter's MLA-split layout — kv_b split into blk.N.attn_k_b (per-head
// transposed, [heads, KVLoraRank, QKNope]) and attn_v_b ([heads, VHead, KVLoraRank]),
// the *_mla head-dim keys alongside the MQA-semantics attention.key_length
// (KVLoraRank+QKRope) / value_length (KVLoraRank) with attention.head_count_kv = 1,
// rope.dimension_count = QKRope, and the pe rows of attn_q_b/attn_kv_a_mqa
// RE-INTERLEAVED into the on-disk LLAMA_ROPE_TYPE_NORM order (the exact inverse of the
// load-time [deinterleaveRoPE]). MoE layers write the ffn_gate_inp router, the fused
// 3-D ffn_{gate,up,down}_exps bank and the fused ffn_*_shexp shared expert, with
// deepseek2.expert_count / expert_used_count / expert_shared_count /
// expert_feed_forward_length / expert_weights_scale / leading_dense_block_count
// (= Layers for a fully-dense model, converter-style). Every projection is transposed
// back into torch [out, in]. No rope-scaling keys are written (unscaled RoPE — the
// only form [DeepSeekV2FromGGUF] accepts). Pass the result to gguf.Write via a
// gguf.File.
func DeepSeekV2ToGGUF(m *DeepSeekV2) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "deepseek2"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	lead := c.FirstKDense
	if c.NRoutedExperts <= 0 {
		lead = c.Layers // converter: first_k_dense_replace = n_layer for MoE-free checkpoints
	}
	moeFF := c.MoEHidden
	if moeFF <= 0 {
		moeFF = c.FFN // converter fallback: moe_intermediate_size → intermediate_size
	}
	meta := map[string]any{
		ggufArch:            arch,
		key(ggufEmbLen):     uint32(c.Dim),
		key(ggufBlockCnt):   uint32(c.Layers),
		key(ggufFFLen):      uint32(c.FFN),
		key(ggufHeadCnt):    uint32(c.Heads),
		key(ggufHeadKV):     uint32(1), // "deepseek2 using MLA converts into MQA"
		key(ggufCtxLen):     uint32(c.Ctx),
		key(ggufRMSEps):     float32(c.Eps),
		key(ggufRopeFreq):   float32(ropeBaseOr(c.RopeBase)),
		key(ggufVocabSize):  uint32(c.Vocab),
		key(ggufQLoraRank):  uint32(c.QLoraRank),
		key(ggufKVLoraRank): uint32(c.KVLoraRank),
		key(ggufKeyLen):     uint32(c.KVLoraRank + c.QKRope), // MQA cache widths, not head dims
		key(ggufValLen):     uint32(c.KVLoraRank),
		key(ggufKeyLenMLA):  uint32(c.QKNope + c.QKRope),
		key(ggufValLenMLA):  uint32(c.VHead),
		key(ggufRopeDim):    uint32(c.QKRope),
		key(ggufLeadDense):  uint32(lead),
		key(ggufExpFFLen):   uint32(moeFF),
		key(ggufExpShared):  uint32(c.NSharedExperts),
	}
	if c.NRoutedExperts > 0 {
		meta[key(ggufExpertCnt)] = uint32(c.NRoutedExperts)
		meta[key(ggufExpertUsed)] = uint32(c.TopK)
		meta[key(ggufExpWScale)] = float32(c.routedScale())
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.FinalNorm.Gamma),
		"output.weight":      transpose2D(m.LmHead), // [dim,vocab] → [vocab,dim]
	}
	kvHead := c.QKNope + c.VHead
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.InputNorm.Gamma)
		ts[p+"attn_q_a.weight"] = transpose2D(b.WqA)
		ts[p+"attn_q_a_norm.weight"] = cloneF64(b.QANorm.Gamma)
		// GoAI [in,out] → torch [out,in], then re-interleave the pe rows into the
		// on-disk LLAMA_ROPE_TYPE_NORM (consecutive-pairs) order.
		ts[p+"attn_q_b.weight"] = interleaveRoPE(transpose2D(b.WqB), c.Heads, c.QKNope+c.QKRope, c.QKNope, c.QKRope)
		ts[p+"attn_kv_a_mqa.weight"] = interleaveRoPE(transpose2D(b.WkvA), 1, c.KVLoraRank+c.QKRope, c.KVLoraRank, c.QKRope)
		ts[p+"attn_kv_a_norm.weight"] = cloneF64(b.KvANorm.Gamma)
		// kv_b: split converter-style into the per-head transposed k_b and v_b.
		kvbTorch := transpose2D(b.WkvB) // [heads·(QKNope+VHead), KVLoraRank]
		kParts := make([]*tensor.Tensor, c.Heads)
		vParts := make([]*tensor.Tensor, c.Heads)
		for h := range c.Heads {
			kParts[h] = transpose2D(sliceRows(kvbTorch, h*kvHead, h*kvHead+c.QKNope)) // [KVLoraRank, QKNope]
			vParts[h] = sliceRows(kvbTorch, h*kvHead+c.QKNope, (h+1)*kvHead)          // [VHead, KVLoraRank]
		}
		ts[p+"attn_k_b.weight"] = stack3D(kParts)
		ts[p+"attn_v_b.weight"] = stack3D(vParts)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_norm.weight"] = cloneF64(b.PostAttnNorm.Gamma)
		if b.MoE != nil {
			ts[p+"ffn_gate_inp.weight"] = transpose2D(b.MoE.Routed.Router.W) // [dim,E] → [E,dim]
			gates := make([]*tensor.Tensor, len(b.MoE.Routed.Experts))
			ups := make([]*tensor.Tensor, len(b.MoE.Routed.Experts))
			downs := make([]*tensor.Tensor, len(b.MoE.Routed.Experts))
			for e, ex := range b.MoE.Routed.Experts {
				gates[e] = transpose2D(ex.Wgate) // [dim,moe_ffn] → [moe_ffn,dim]
				ups[e] = transpose2D(ex.Wup)
				downs[e] = transpose2D(ex.Wdown) // [moe_ffn,dim] → [dim,moe_ffn]
			}
			ts[p+"ffn_gate_exps.weight"] = stack3D(gates)
			ts[p+"ffn_up_exps.weight"] = stack3D(ups)
			ts[p+"ffn_down_exps.weight"] = stack3D(downs)
			sh := b.MoE.Shared[0] // GoAI holds the shared experts pre-fused, like the file
			ts[p+"ffn_gate_shexp.weight"] = transpose2D(sh.Wgate)
			ts[p+"ffn_up_shexp.weight"] = transpose2D(sh.Wup)
			ts[p+"ffn_down_shexp.weight"] = transpose2D(sh.Wdown)
		} else {
			ts[p+"ffn_gate.weight"] = transpose2D(b.Dense.Wgate)
			ts[p+"ffn_up.weight"] = transpose2D(b.Dense.Wup)
			ts[p+"ffn_down.weight"] = transpose2D(b.Dense.Wdown)
		}
	}
	return meta, ts
}

// interleaveRoPE is the exact inverse of [deinterleaveRoPE]: it reorders the pe rows of
// a torch [out, in] projection from GoAI's split-half order (evens first, then odds)
// back into the interleaved (consecutive-pairs, LLAMA_ROPE_TYPE_NORM) on-disk order:
// within each of the `heads` blocks of `block` rows, the row at offset peOffset+2i
// takes peOffset+i and peOffset+2i+1 takes peOffset+i+ropeDim/2, for i in
// [0, ropeDim/2). All other rows are copied unchanged; a fresh tensor is returned.
func interleaveRoPE(w *tensor.Tensor, heads, block, peOffset, ropeDim int) *tensor.Tensor {
	out, in := w.Shape()[0], w.Shape()[1]
	res := tensor.New(tensor.F64, tensor.Shape{out, in})
	src := cloneF64(w).Storage().F64()
	dst := res.Storage().F64()
	copy(dst, src)
	half := ropeDim / 2
	for h := range heads {
		base := h*block + peOffset
		for i := range half {
			copy(dst[(base+2*i)*in:(base+2*i+1)*in], src[(base+i)*in:(base+i+1)*in])
			copy(dst[(base+2*i+1)*in:(base+2*i+2)*in], src[(base+i+half)*in:(base+i+half+1)*in])
		}
	}
	return res
}

// deinterleaveRoPEChecked is [deinterleaveRoPE] with the row geometry PINNED first,
// so a malformed file is a clean error instead of a silently mis-permuted model.
//
// deinterleaveRoPE walks `heads` blocks of `block` rows and rewrites each block's pe
// span. It never inspects the tensor it was handed: when heads×block is SMALLER than
// the actual row count it permutes only a prefix and returns a tensor that looks
// perfectly well-formed — the model then loads with err == nil and computes wrong
// attention, the wrong-but-nil-error failure §V29 calls out as the harder one to
// trace. The quantized twin never had this hole (QuantDeepSeekV2FromGGUF pins the
// [out, in] shape and routes through deepseekV2DeinterleavePerm's explicit error), so
// this reuses that SAME predicate rather than restating it: the float and quantized
// loaders now accept and reject exactly the same set of files, which is the property
// worth having — a second, independently-worded check would drift.
//
// Failure modes: a non-2-D tensor, a row count that is not heads×block, a
// non-positive heads or block, or a pe span that does not fit an even rotary width
// inside the block. Valid files are unaffected — the predicate passes whenever the
// on-disk geometry matches the config the same file declares.
func deinterleaveRoPEChecked(name string, w *tensor.Tensor, heads, block, peOffset, ropeDim int) (*tensor.Tensor, error) {
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: GGUF %s is %d-D, want a 2-D [out, in] projection", name, w.Ndim())
	}
	if _, err := deepseekV2DeinterleavePerm(w.Shape()[0], heads, block, peOffset, ropeDim); err != nil {
		return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
	}
	return deinterleaveRoPE(w, heads, block, peOffset, ropeDim), nil
}
