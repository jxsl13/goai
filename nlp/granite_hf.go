package nlp

import "github.com/jxsl13/goai/tensor"

// GraniteFromHF builds a [Llama] from a Hugging Face IBM Granite checkpoint
// (transformers' GraniteForCausalLM). Granite is structurally a plain Llama —
// pre-norm RMSNorm, RoPE, grouped-query attention, a bias-free SwiGLU FFN, no
// biases — plus FOUR scalar multipliers that Granite threads through the forward
// pass:
//
//   - embedding_multiplier: inputs_embeds *= this scalar right after the token
//     lookup (like Gemma's √dim normalizer, but a config value);
//   - attention_multiplier: the pre-softmax attention scale IS this scalar,
//     replacing the default 1/√headDim;
//   - residual_multiplier: every residual add becomes x = residual + sublayer·this
//     (both the attention and the FFN residual);
//   - logits_scaling: the final logits are divided by this scalar.
//
// The weights use the same model.layers.N.self_attn / mlp names as Llama, so the
// loader is [LlamaFromHF] verbatim; the four scalars ride in cfg (set them from
// config.json via [GraniteConfigFromHF]). Because [LlamaConfig]'s Granite fields
// default to 0 ("unset → identity"), a plain-Llama load with the scalars left zero
// is byte-identical to before — Granite is a superset, not a fork. This function is
// a named, documented entry point so Granite reads as a first-class architecture
// rather than a Llama special case.
//
// cfg supplies Heads, KVHeads, Eps, RopeBase, Ctx and the four Granite scalars;
// Dim, Vocab, Layers, Hidden are inferred from the checkpoint tensors.
func GraniteFromHF(ts map[string]*tensor.Tensor, cfg LlamaConfig) (*Llama, error) {
	return LlamaFromHF(ts, cfg)
}
