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
