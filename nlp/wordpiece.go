package nlp

import (
	"fmt"
	"strings"
)

// WordPiece is the subword tokenizer of BERT (Devlin, Chang, Lee & Toutanova 2019; the wordpiece
// model of Schuster & Nakajima 2012 and Wu et al. 2016 GNMT §4). It completes this library's
// tokenizer set alongside byte-level BPE (GPT-2/Llama-3, nlp.Tokenizer) and Unigram/SentencePiece
// (Llama/Mistral/T5, nlp.Unigram §R87). Given a pre-tokenized word, WordPiece encodes it by GREEDY
// LONGEST-MATCH-FIRST (MaxMatch): from the start of the word it takes the LONGEST prefix present in
// the vocabulary, emits it, and continues from the remainder — distinct from BPE's learned merges
// and Unigram's probabilistic Viterbi. Continuation pieces (any not at the word start) are looked
// up with a "##" prefix (e.g. "playing" with {play, ##ing} → [play, ##ing]); if at some position
// no substring down to a single character matches, the WHOLE word becomes one [UNK]; words longer
// than MaxChars are [UNK] without tokenizing.
//
// Basic pre-tokenization (punctuation splitting, lowercasing, accent stripping) and normalization
// are assumed applied upstream (as with Unigram's NFKC); Encode splits the input on whitespace and
// runs WordPiece per word.
type WordPiece struct {
	id           map[string]int // piece → id
	pieces       []string       // id → piece
	unkID        int            // id emitted for an unmatchable / too-long word
	continuation string         // continuation-piece prefix ("##")
	maxChars     int            // words longer than this (in runes) → unk
}

// WordPieceOption configures a WordPiece tokenizer (functional-options idiom, §C12).
type WordPieceOption func(*WordPiece)

// WithWordPieceUnk sets the id emitted for a word that cannot be tokenized (no subword match,
// or longer than the max-chars limit).
//
// In plain terms: the "unknown word" token id. Boundary behavior — any valid vocab id; point
// it at your real unknown token so downstream code recognizes gaps. SPECIAL VALUE / default:
// the id of the "[UNK]" piece if the vocabulary has one, else 0 (research-grounded: the
// BERT/WordPiece convention of a dedicated [UNK] symbol).
func WithWordPieceUnk(id int) WordPieceOption { return func(w *WordPiece) { w.unkID = id } }

// WithWordPieceContinuation sets the prefix marking a subword that continues the previous one
// (rather than starting a new word).
//
// In plain terms: the little marker glued to the front of mid-word pieces ("play", "##ing").
// Boundary behavior — must match the marker your vocabulary was built with, or nothing
// tokenizes. Default "##" (research-grounded: the BERT/WordPiece continuation prefix).
func WithWordPieceContinuation(p string) WordPieceOption {
	return func(w *WordPiece) { w.continuation = p }
}

// WithWordPieceMaxChars caps the length (in runes) of a single word; longer words are emitted
// as the unknown token instead of being tokenized.
//
// In plain terms: a safety valve — WordPiece's greedy matching is quadratic in word length, so
// pathologically long "words" (e.g. a 5000-char blob) are skipped rather than stalling the
// tokenizer. Boundary behavior — too small drops legitimate long words to unk; very large
// removes the protection. SPECIAL VALUE: non-positive is ignored (keeps the current value).
//
// Default 100 (research-grounded: the BERT/Hugging Face max_input_chars_per_word default).
func WithWordPieceMaxChars(n int) WordPieceOption {
	return func(w *WordPiece) {
		if n > 0 {
			w.maxChars = n
		}
	}
}

// NewWordPiece builds a WordPiece tokenizer from vocab (id = index). Continuation pieces carry the
// "##" prefix in vocab. Defaults: continuation "##", maxChars 100, unk = id of "[UNK]" if present.
func NewWordPiece(vocab []string, opts ...WordPieceOption) (*WordPiece, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("nlp: WordPiece needs a non-empty vocabulary")
	}
	w := &WordPiece{
		id:           make(map[string]int, len(vocab)),
		pieces:       append([]string(nil), vocab...),
		continuation: "##",
		maxChars:     100,
	}
	for i, p := range vocab {
		if _, dup := w.id[p]; dup {
			return nil, fmt.Errorf("nlp: WordPiece duplicate piece %q", p)
		}
		w.id[p] = i
	}
	if id, ok := w.id["[UNK]"]; ok {
		w.unkID = id
	}
	for _, o := range opts {
		o(w)
	}
	return w, nil
}

// Encode tokenizes text into ids: it splits on whitespace and applies greedy longest-match-first
// WordPiece to each word.
func (w *WordPiece) Encode(text string) []int {
	var out []int
	for _, word := range strings.Fields(text) {
		out = append(out, w.encodeWord(word)...)
	}
	return out
}

// encodeWord runs MaxMatch on a single word.
func (w *WordPiece) encodeWord(word string) []int {
	runes := []rune(word)
	if len(runes) > w.maxChars {
		return []int{w.unkID}
	}
	var toks []int
	start := 0
	for start < len(runes) {
		// longest substring runes[start:end] in the vocab (with "##" if not word-initial).
		end := len(runes)
		matchedID, matchedLen := -1, 0
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = w.continuation + sub
			}
			if id, ok := w.id[sub]; ok {
				matchedID, matchedLen = id, end-start
				break
			}
			end--
		}
		if matchedID < 0 {
			return []int{w.unkID} // no piece at this position ⇒ the whole word is unk
		}
		toks = append(toks, matchedID)
		start += matchedLen
	}
	return toks
}

// Decode concatenates the pieces of ids: a continuation piece ("##…") is appended to the current
// word (prefix stripped), a non-continuation piece starts a new space-separated word.
func (w *WordPiece) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(w.pieces) {
			continue
		}
		p := w.pieces[id]
		if strings.HasPrefix(p, w.continuation) && b.Len() > 0 {
			b.WriteString(strings.TrimPrefix(p, w.continuation))
			continue
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(p)
	}
	return b.String()
}
