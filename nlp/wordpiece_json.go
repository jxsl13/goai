package nlp

import (
	"encoding/json"
	"fmt"
)

// HuggingFace tokenizer.json interop for WordPiece (the "model":{"type":"WordPiece"} form used by
// BERT/DistilBERT/ELECTRA). Definitional source: the tokenizers library schema — verified by
// round-trip + fuzz (§V15), §V16-exempt like the BPE (§T236) and Unigram (§T237) tokenizer.json
// loaders. Only the WordPiece model core (vocab + unk_token + continuing_subword_prefix +
// max_input_chars_per_word, plus any added_tokens) is read; normalizer/pre_tokenizer sections are
// ignored (basic pre-tokenization is applied upstream, as elsewhere in nlp.WordPiece).

// hfWordPieceJSON is the minimal tokenizer.json shape needed to rebuild a WordPiece tokenizer.
type hfWordPieceJSON struct {
	Model struct {
		Type                    string         `json:"type"`
		Vocab                   map[string]int `json:"vocab"` // token → id
		UnkToken                string         `json:"unk_token"`
		ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
		MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// WordPieceFromJSON builds a WordPiece tokenizer from HuggingFace tokenizer.json bytes. It errors
// on invalid JSON, a non-WordPiece model type, or an empty vocabulary. The unk_token,
// continuing_subword_prefix ("##") and max_input_chars_per_word (100) from the file are applied;
// caller opts override them.
func WordPieceFromJSON(data []byte, opts ...WordPieceOption) (*WordPiece, error) {
	var tj hfWordPieceJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("nlp: parse tokenizer.json: %w", err)
	}
	if tj.Model.Type != "" && tj.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("nlp: tokenizer.json model.type=%q, want WordPiece", tj.Model.Type)
	}
	// place each token at its id; added_tokens (special tokens) share the id space.
	maxID := -1
	for _, id := range tj.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, a := range tj.AddedTokens {
		if a.ID > maxID {
			maxID = a.ID
		}
	}
	if maxID < 0 {
		return nil, fmt.Errorf("nlp: tokenizer.json has an empty vocabulary")
	}
	// hostile-input guard (§B47, same class as the BPE parser): the vocab slice
	// is indexed by id — reject implausibly large ids instead of allocating them.
	if nTok := len(tj.Model.Vocab) + len(tj.AddedTokens); maxID+1 > 4*nTok+1024 {
		return nil, fmt.Errorf("nlp: tokenizer.json max token id %d implausible for %d tokens", maxID, nTok)
	}
	vocab := make([]string, maxID+1)
	for tok, id := range tj.Model.Vocab {
		if id >= 0 {
			vocab[id] = tok
		}
	}
	for _, a := range tj.AddedTokens {
		if a.ID >= 0 {
			vocab[a.ID] = a.Content
		}
	}
	// parsed options first, then the caller's (so the caller can override the file).
	parsed := make([]WordPieceOption, 0, 3+len(opts))
	if p := tj.Model.ContinuingSubwordPrefix; p != "" {
		parsed = append(parsed, WithWordPieceContinuation(p))
	}
	if tj.Model.MaxInputCharsPerWord > 0 {
		parsed = append(parsed, WithWordPieceMaxChars(tj.Model.MaxInputCharsPerWord))
	}
	if u := tj.Model.UnkToken; u != "" {
		if id, ok := tj.Model.Vocab[u]; ok {
			parsed = append(parsed, WithWordPieceUnk(id))
		}
	}
	return NewWordPiece(vocab, append(parsed, opts...)...)
}

// ToJSON serializes the tokenizer to the minimal HuggingFace tokenizer.json WordPiece form, so
// WordPieceFromJSON(w.ToJSON()) round-trips it.
func (w *WordPiece) ToJSON() ([]byte, error) {
	vocab := make(map[string]int, len(w.pieces))
	for id, tok := range w.pieces {
		vocab[tok] = id
	}
	var out hfWordPieceJSON
	out.Model.Type = "WordPiece"
	out.Model.Vocab = vocab
	out.Model.ContinuingSubwordPrefix = w.continuation
	out.Model.MaxInputCharsPerWord = w.maxChars
	if w.unkID >= 0 && w.unkID < len(w.pieces) {
		out.Model.UnkToken = w.pieces[w.unkID]
	}
	return json.Marshal(out)
}
