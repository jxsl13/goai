package nlp

import "fmt"

// GGUF tokenizer metadata keys (the ggml/llama.cpp convention, §R88).
const (
	ggufTokModel  = "tokenizer.ggml.model"
	ggufTokTokens = "tokenizer.ggml.tokens"
	ggufTokScores = "tokenizer.ggml.scores"
	ggufTokUnkID  = "tokenizer.ggml.unknown_token_id"
)

// UnigramFromGGUF builds a Unigram tokenizer from the metadata map of a parsed GGUF
// model file (gguf.File.Metadata), reading the ggml/llama.cpp tokenizer convention
// (§R88): tokenizer.ggml.tokens (the vocabulary, an array of strings indexed by token
// id) paired with tokenizer.ggml.scores (an array of float32 log-probabilities), and
// tokenizer.ggml.unknown_token_id for the <unk> id. This wires a real .gguf model's
// embedded tokenizer to the Viterbi Unigram of NewUnigram, so weights and tokenizer
// load from one file.
//
// It accepts tokenizer.ggml.model == "t5" (ggml's UGM — a true Unigram tokenizer, the
// exact match for the Viterbi decoding here) and "llama" (SentencePiece, which carries
// the same per-token scores and ▁ meta-symbol). NOTE: for "llama"/SPM, llama.cpp runs
// a greedy best-score-bigram MERGE rather than Viterbi; this loader always applies the
// Viterbi 1-best over the same scores, which is the higher-likelihood segmentation but
// can differ token-for-token from llama.cpp's SPM output (an explicit, documented
// choice — §R88). Byte-level BPE models ("gpt2"/"bpe") are not handled here (they need
// merge rules, not scores) and return an error.
func UnigramFromGGUF(meta map[string]any) (*Unigram, error) {
	model, _ := meta[ggufTokModel].(string)
	switch model {
	case "t5", "llama":
	case "gpt2", "bpe":
		return nil, fmt.Errorf("nlp: GGUF tokenizer.ggml.model=%q is byte-level BPE, not Unigram (needs merges, not scores)", model)
	default:
		return nil, fmt.Errorf("nlp: GGUF tokenizer.ggml.model=%q is not a Unigram/SPM tokenizer", model)
	}

	toks, ok := meta[ggufTokTokens].([]any)
	if !ok || len(toks) == 0 {
		return nil, fmt.Errorf("nlp: GGUF %s missing or empty", ggufTokTokens)
	}
	scores, ok := meta[ggufTokScores].([]any)
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF %s missing (Unigram needs per-token scores)", ggufTokScores)
	}
	if len(scores) != len(toks) {
		return nil, fmt.Errorf("nlp: GGUF tokens (%d) and scores (%d) length mismatch", len(toks), len(scores))
	}

	vocab := make([]UnigramPiece, len(toks))
	for i := range toks {
		piece, ok := toks[i].(string)
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF token %d is %T, want string", i, toks[i])
		}
		score, ok := toks32(scores[i])
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF score %d is %T, want float32", i, scores[i])
		}
		vocab[i] = UnigramPiece{Piece: piece, Score: score}
	}

	var opts []UnigramOption
	if unk, ok := ggufTokenID(meta[ggufTokUnkID]); ok {
		opts = append(opts, WithUnigramUnkID(unk))
	}
	return NewUnigram(vocab, opts...)
}

// toks32 reads a GGUF score (a float32 per spec, but tolerate float64) as float64.
func toks32(v any) (float64, bool) {
	switch s := v.(type) {
	case float32:
		return float64(s), true
	case float64:
		return s, true
	default:
		return 0, false
	}
}

// ggufTokenID reads a GGUF special-token id (a uint32 per spec, but tolerate the other
// integer types a metadata reader may yield) as an int.
func ggufTokenID(v any) (int, bool) {
	switch n := v.(type) {
	case uint32:
		return int(n), true
	case int32:
		return int(n), true
	case uint64:
		return int(n), true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}
