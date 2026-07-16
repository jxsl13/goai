package nlp

import "github.com/jxsl13/goai/tensor"

// Qwen3MoeFromHF builds a [Mixtral] from a Hugging Face Qwen3-MoE checkpoint
// (transformers' Qwen3MoeForCausalLM). Qwen3-MoE is Mixtral with Qwen3 attention:
// the sparse top-k Mixture-of-Experts FFN is byte-for-byte the Mixtral fused layout
// (`mlp.gate.weight` [E,dim], `mlp.experts.gate_up_proj` [E,2·moe_inter,dim],
// `mlp.experts.down_proj` [E,dim,moe_inter]), while the attention adds Qwen3's per-head
// QK-norm — an RMSNorm over head_dim applied to the queries and keys before RoPE
// (`self_attn.{q,k}_norm.weight`) — on top of the usual bias-free RMSNorm/RoPE/GQA.
// Head_dim may be decoupled from dim/heads (q_proj is [heads·head_dim, dim]); the
// attention ops infer head_dim from the tensor widths, so no extra config is needed.
//
// This is [MixtralFromHF] verbatim: that loader already reads the optional q_norm/k_norm
// gains when present (they are absent in a plain Mixtral checkpoint, so it stays a no-op
// there), and [Mixtral.Forward]/[Mixtral.DecodeStep] apply them via [applyQKNorm]. This
// function is a named, documented entry point so Qwen3-MoE reads as a first-class
// architecture rather than a Mixtral special case.
//
// cfg supplies Heads, KVHeads, TopK, Eps, RopeBase, Ctx; Dim, Vocab, Layers, Hidden
// (moe_inter) and Experts are inferred from the checkpoint.
//
// Limitation: the combine step always renormalizes the top-k router weights
// (nn.SparseMoE, matching transformers with norm_topk_prob=True, Qwen3-MoE's default).
// A norm_topk_prob=False checkpoint would route to the same experts but weight them
// without renormalization and so would NOT match this loader.
func Qwen3MoeFromHF(ts map[string]*tensor.Tensor, cfg MixtralConfig) (*Mixtral, error) {
	return MixtralFromHF(ts, cfg)
}
