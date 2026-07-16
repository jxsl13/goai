package nlp

import (
	"encoding/json"
	"fmt"
)

// LlamaConfigFromHF parses a Hugging Face Llama config.json into a [LlamaConfig]
// suitable for [LlamaFromHF]. It fills the hyperparameters that are NOT
// derivable from the weight tensors — Heads, KVHeads, Eps, RopeBase, Ctx — so a
// caller can do: cfg, _ := LlamaConfigFromHF(cfgBytes); m, _ := LlamaFromHF(ts, cfg).
// The tensor-derivable dims (Dim, Vocab, Layers, Hidden) are left zero for
// LlamaFromHF to infer. Missing rope_theta / rms_norm_eps fall back to the Llama
// defaults (10000, 1e-5).
func LlamaConfigFromHF(configJSON []byte) (LlamaConfig, error) {
	var j struct {
		NumAttentionHeads int     `json:"num_attention_heads"`
		NumKeyValueHeads  *int    `json:"num_key_value_heads"`
		RMSNormEps        float64 `json:"rms_norm_eps"`
		RopeTheta         float64 `json:"rope_theta"` // older configs: top-level
		RopeParameters    *struct {
			RopeTheta float64 `json:"rope_theta"` // transformers ≥5: nested
		} `json:"rope_parameters"`
		MaxPos int `json:"max_position_embeddings"`
	}
	if err := json.Unmarshal(configJSON, &j); err != nil {
		return LlamaConfig{}, fmt.Errorf("nlp: parse Llama config.json: %w", err)
	}
	if j.NumAttentionHeads <= 0 {
		return LlamaConfig{}, fmt.Errorf("nlp: config.json missing num_attention_heads")
	}
	kv := j.NumAttentionHeads
	if j.NumKeyValueHeads != nil {
		kv = *j.NumKeyValueHeads
	}
	eps := j.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	rope := j.RopeTheta
	if rope == 0 && j.RopeParameters != nil {
		rope = j.RopeParameters.RopeTheta
	}
	return LlamaConfig{
		Heads:    j.NumAttentionHeads,
		KVHeads:  kv,
		Eps:      eps,
		RopeBase: rope, // 0 → LlamaFromHF's ropeBaseOr defaults to 10000
		Ctx:      j.MaxPos,
	}, nil
}

// GemmaConfigFromHF parses a Hugging Face Gemma config.json into a [GemmaConfig]
// for [GemmaFromHF] — the non-tensor-derivable Heads, KVHeads, Eps, RopeBase, Ctx
// (Dim, Vocab, Layers, FFN, HeadDim are inferred from the weights). Missing
// rms_norm_eps / rope_theta fall back to the Gemma defaults (1e-6, 10000).
func GemmaConfigFromHF(configJSON []byte) (GemmaConfig, error) {
	var j struct {
		NumAttentionHeads int     `json:"num_attention_heads"`
		NumKeyValueHeads  *int    `json:"num_key_value_heads"`
		RMSNormEps        float64 `json:"rms_norm_eps"`
		RopeTheta         float64 `json:"rope_theta"`
		MaxPos            int     `json:"max_position_embeddings"`
	}
	if err := json.Unmarshal(configJSON, &j); err != nil {
		return GemmaConfig{}, fmt.Errorf("nlp: parse Gemma config.json: %w", err)
	}
	if j.NumAttentionHeads <= 0 {
		return GemmaConfig{}, fmt.Errorf("nlp: config.json missing num_attention_heads")
	}
	kv := j.NumAttentionHeads
	if j.NumKeyValueHeads != nil {
		kv = *j.NumKeyValueHeads
	}
	eps := j.RMSNormEps
	if eps == 0 {
		eps = 1e-6
	}
	return GemmaConfig{
		Heads:    j.NumAttentionHeads,
		KVHeads:  kv,
		Eps:      eps,
		RopeBase: j.RopeTheta, // 0 → GemmaFromHF defaults to 10000
		Ctx:      j.MaxPos,
	}, nil
}

// Gemma2ConfigFromHF parses a Hugging Face Gemma 2 config.json into a
// [Gemma2Config] for [Gemma2FromHF] — including the three Gemma 2 scalars
// (query_pre_attn_scalar, attn_logit_softcapping, final_logit_softcapping) that a
// bare GemmaConfig lacks. Dim, Vocab, Layers, FFN, HeadDim are inferred from the
// weights. Missing rms_norm_eps / rope_theta fall back to the Gemma defaults
// (1e-6, 10000); the soft-cap fields default to 0 (disabled) if absent.
func Gemma2ConfigFromHF(configJSON []byte) (Gemma2Config, error) {
	var j struct {
		NumAttentionHeads  int      `json:"num_attention_heads"`
		NumKeyValueHeads   *int     `json:"num_key_value_heads"`
		RMSNormEps         float64  `json:"rms_norm_eps"`
		RopeTheta          float64  `json:"rope_theta"`
		MaxPos             int      `json:"max_position_embeddings"`
		QueryPreAttnScalar *float64 `json:"query_pre_attn_scalar"`
		AttnLogitSoftcap   float64  `json:"attn_logit_softcapping"`
		FinalLogitSoftcap  float64  `json:"final_logit_softcapping"`
	}
	if err := json.Unmarshal(configJSON, &j); err != nil {
		return Gemma2Config{}, fmt.Errorf("nlp: parse Gemma2 config.json: %w", err)
	}
	if j.NumAttentionHeads <= 0 {
		return Gemma2Config{}, fmt.Errorf("nlp: config.json missing num_attention_heads")
	}
	kv := j.NumAttentionHeads
	if j.NumKeyValueHeads != nil {
		kv = *j.NumKeyValueHeads
	}
	eps := j.RMSNormEps
	if eps == 0 {
		eps = 1e-6
	}
	var qScalar float64
	if j.QueryPreAttnScalar != nil {
		qScalar = *j.QueryPreAttnScalar
	}
	return Gemma2Config{
		Heads:              j.NumAttentionHeads,
		KVHeads:            kv,
		Eps:                eps,
		RopeBase:           j.RopeTheta, // 0 → Gemma2FromHF defaults to 10000
		Ctx:                j.MaxPos,
		QueryPreAttnScalar: qScalar, // 0 → Gemma2 falls back to HeadDim
		AttnLogitCap:       j.AttnLogitSoftcap,
		FinalLogitCap:      j.FinalLogitSoftcap,
	}, nil
}

// MixtralConfigFromHF parses a Hugging Face Mixtral config.json into a
// [MixtralConfig] for [MixtralFromHF] — Heads, KVHeads, TopK, Eps, RopeBase, Ctx
// (Dim, Vocab, Layers, Hidden, Experts are inferred from the weights). Missing
// rms_norm_eps / rope_theta / num_experts_per_tok fall back to (1e-5, 10000, 2).
func MixtralConfigFromHF(configJSON []byte) (MixtralConfig, error) {
	var j struct {
		NumAttentionHeads int     `json:"num_attention_heads"`
		NumKeyValueHeads  *int    `json:"num_key_value_heads"`
		NumExpertsPerTok  int     `json:"num_experts_per_tok"`
		RMSNormEps        float64 `json:"rms_norm_eps"`
		RopeTheta         float64 `json:"rope_theta"`
		MaxPos            int     `json:"max_position_embeddings"`
	}
	if err := json.Unmarshal(configJSON, &j); err != nil {
		return MixtralConfig{}, fmt.Errorf("nlp: parse Mixtral config.json: %w", err)
	}
	if j.NumAttentionHeads <= 0 {
		return MixtralConfig{}, fmt.Errorf("nlp: config.json missing num_attention_heads")
	}
	kv := j.NumAttentionHeads
	if j.NumKeyValueHeads != nil {
		kv = *j.NumKeyValueHeads
	}
	eps := j.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	topK := j.NumExpertsPerTok
	if topK <= 0 {
		topK = 2
	}
	return MixtralConfig{
		Heads:    j.NumAttentionHeads,
		KVHeads:  kv,
		TopK:     topK,
		Eps:      eps,
		RopeBase: j.RopeTheta, // 0 → MixtralFromHF defaults to 10000
		Ctx:      j.MaxPos,
	}, nil
}

// BertConfigFromHF parses a Hugging Face BERT/RoBERTa config.json into a
// [BertConfig] for [BertFromHF] / [RobertaFromHF] — the non-tensor-derivable
// Heads and Eps. Missing layer_norm_eps falls back to 1e-12.
func BertConfigFromHF(configJSON []byte) (BertConfig, error) {
	var j struct {
		NumAttentionHeads int     `json:"num_attention_heads"`
		LayerNormEps      float64 `json:"layer_norm_eps"`
	}
	if err := json.Unmarshal(configJSON, &j); err != nil {
		return BertConfig{}, fmt.Errorf("nlp: parse BERT config.json: %w", err)
	}
	if j.NumAttentionHeads <= 0 {
		return BertConfig{}, fmt.Errorf("nlp: config.json missing num_attention_heads")
	}
	eps := j.LayerNormEps
	if eps == 0 {
		eps = 1e-12
	}
	return BertConfig{Heads: j.NumAttentionHeads, Eps: eps}, nil
}
