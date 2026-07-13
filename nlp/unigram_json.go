package nlp

import (
	"encoding/json"
	"fmt"
)

// HuggingFace tokenizer.json interop for the SentencePiece Unigram model (the
// "model":{"type":"Unigram"} form used by Llama, Mistral and T5). Definitional source:
// the tokenizers library schema — verified by round-trip + fuzz (§V15). Only the model
// core (the scored vocabulary and unk_id) is read; normalizer/pre_tokenizer/decoder
// sections are ignored (the tokenizer applies SentencePiece whitespace escaping itself).

// hfUnigramJSON is the minimal tokenizer.json shape for a Unigram model: vocab is a list
// of [piece, score] pairs, and unk_id (when present) is the index of the <unk> piece.
type hfUnigramJSON struct {
	Model struct {
		Type  string              `json:"type"`
		UnkID *int                `json:"unk_id"`
		Vocab [][]json.RawMessage `json:"vocab"` // each entry: [ "piece", score ]
	} `json:"model"`
}

// UnigramFromJSON builds a SentencePiece Unigram tokenizer from HuggingFace
// tokenizer.json bytes. It errors on invalid JSON, a non-Unigram model type, an empty
// vocabulary, or a malformed vocab entry.
func UnigramFromJSON(data []byte, opts ...UnigramOption) (*Unigram, error) {
	var tj hfUnigramJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("nlp: parse tokenizer.json: %w", err)
	}
	if tj.Model.Type != "" && tj.Model.Type != "Unigram" {
		return nil, fmt.Errorf("nlp: tokenizer.json model.type=%q, want Unigram", tj.Model.Type)
	}
	vocab := make([]UnigramPiece, len(tj.Model.Vocab))
	for i, e := range tj.Model.Vocab {
		if len(e) != 2 {
			return nil, fmt.Errorf("nlp: Unigram vocab entry %d has %d fields, want [piece,score]", i, len(e))
		}
		var piece string
		if err := json.Unmarshal(e[0], &piece); err != nil {
			return nil, fmt.Errorf("nlp: Unigram vocab entry %d piece: %w", i, err)
		}
		var score float64
		if err := json.Unmarshal(e[1], &score); err != nil {
			return nil, fmt.Errorf("nlp: Unigram vocab entry %d score: %w", i, err)
		}
		vocab[i] = UnigramPiece{Piece: piece, Score: score}
	}
	// unk_id from the file is applied first, so caller opts can still override it.
	if tj.Model.UnkID != nil {
		opts = append([]UnigramOption{WithUnigramUnkID(*tj.Model.UnkID)}, opts...)
	}
	return NewUnigram(vocab, opts...)
}

// ToJSON serializes the tokenizer to the minimal HuggingFace tokenizer.json Unigram form
// (model.type "Unigram" with unk_id and the [piece, score] vocab), so
// UnigramFromJSON(u.ToJSON()) round-trips it.
func (u *Unigram) ToJSON() ([]byte, error) {
	vocab := make([][]any, len(u.pieces))
	for i, p := range u.pieces {
		vocab[i] = []any{p.Piece, p.Score}
	}
	unk := u.unkID
	out := struct {
		Model struct {
			Type  string  `json:"type"`
			UnkID *int    `json:"unk_id"`
			Vocab [][]any `json:"vocab"`
		} `json:"model"`
	}{}
	out.Model.Type = "Unigram"
	out.Model.UnkID = &unk
	out.Model.Vocab = vocab
	return json.Marshal(out)
}
