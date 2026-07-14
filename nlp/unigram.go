package nlp

import (
	"fmt"
	"math"
	"strings"
)

// spaceMeta is SentencePiece's whitespace meta-symbol ▁ (U+2581, LOWER ONE EIGHTH
// BLOCK): spaces are escaped to it so segmentation and detokenization are lossless.
const spaceMeta = "▁"

// unkPenalty is SentencePiece's kUnkPenalty: an unmatched character costs
// min(scores)−unkPenalty, so the Viterbi path only falls back to <unk> when no real
// piece covers a position.
const unkPenalty = 10.0

// UnigramPiece is one vocabulary entry of a Unigram tokenizer: a subword string and
// its score, the log-probability log p(piece) of the unigram language model.
type UnigramPiece struct {
	Piece string  // the subword (may include the ▁ space meta-symbol)
	Score float64 // log p(piece)
}

// Unigram is the SentencePiece Unigram-LM subword tokenizer (Kudo 2018, "Subword
// Regularization", arXiv:1804.10959, §3.1, §R87) — the tokenizer of the Llama, Mistral
// and T5 model families. The vocabulary is a fixed set of pieces each carrying a
// score = log p(piece) under a unigram language model P(x)=∏ᵢ p(xᵢ). Inference-time
// encoding is the 1-best (most-probable) segmentation, found by Viterbi dynamic
// programming that MAXIMIZES the sum of piece log-probabilities — not greedy
// longest-match, which can pick a lower-probability tiling.
//
// DP recurrence over the input's Unicode characters, best[i] = max score of any
// segmentation of the first i characters:
//
//	best[i] = max over pieces p ending at i of  best[i−len(p)] + score(p)
//
// with backpointers to recover the pieces. A position no piece covers takes a single
// character as <unk> at score min(scores)−kUnkPenalty (kUnkPenalty=10), so the DP is
// always feasible. Whitespace is escaped to ▁ and a leading ▁ is prepended
// (add_dummy_prefix) before segmentation; Decode concatenates the pieces and maps ▁
// back to spaces, dropping the dummy — lossless when the vocabulary covers every
// character. Input is assumed already Unicode-normalized (SentencePiece applies NFKC
// upstream; this tokenizer segments the string it is given).
type Unigram struct {
	id       map[string]int // piece → id
	pieces   []UnigramPiece // id → piece+score
	unkID    int            // id emitted for an unmatched character
	minScore float64        // min piece score (base of the unk penalty)
	maxRunes int            // longest piece in runes (bounds the DP inner loop)
	dummy    bool           // prepend a ▁ (add_dummy_prefix)
}

// UnigramOption configures a Unigram tokenizer (functional-options idiom, §C12).
type UnigramOption func(*Unigram)

// WithUnigramUnkID sets the id emitted for a character no piece in the vocabulary covers.
//
// In plain terms: the "unknown" token id, used when a character can't be built from any known
// piece. Boundary behavior — any valid vocab id. SPECIAL VALUE / default: the id of a piece
// literally named "<unk>" if present, else 0 (research-grounded: the SentencePiece Unigram
// convention of a dedicated <unk> symbol).
func WithUnigramUnkID(id int) UnigramOption { return func(u *Unigram) { u.unkID = id } }

// WithUnigramDummyPrefix toggles prepending a ▁ (the SentencePiece space marker) before
// encoding, so a word at the START of the text tokenizes the same way it would mid-sentence.
//
// In plain terms: SentencePiece marks spaces with ▁ and treats them as part of the following
// word; without this, "Hello" at the very start would tokenize differently from " Hello" in
// the middle. Turning it on makes the two consistent. Boundary behavior — a boolean; off only
// if your vocabulary was trained without add_dummy_prefix.
//
// Default true (research-grounded: SentencePiece's add_dummy_prefix default).
func WithUnigramDummyPrefix(on bool) UnigramOption { return func(u *Unigram) { u.dummy = on } }

// NewUnigram builds a Unigram tokenizer from a scored vocabulary (piece id = index).
// Scores are log-probabilities (≤ 0, larger = more frequent). It errors on an empty
// vocabulary or duplicate pieces.
func NewUnigram(vocab []UnigramPiece, opts ...UnigramOption) (*Unigram, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("nlp: Unigram needs a non-empty vocabulary")
	}
	u := &Unigram{
		id:       make(map[string]int, len(vocab)),
		pieces:   append([]UnigramPiece(nil), vocab...),
		minScore: math.Inf(1),
		dummy:    true,
	}
	for i, p := range vocab {
		if _, dup := u.id[p.Piece]; dup {
			return nil, fmt.Errorf("nlp: Unigram duplicate piece %q", p.Piece)
		}
		u.id[p.Piece] = i
		if r := len([]rune(p.Piece)); r > u.maxRunes {
			u.maxRunes = r
		}
		if p.Score < u.minScore {
			u.minScore = p.Score
		}
	}
	if id, ok := u.id["<unk>"]; ok {
		u.unkID = id
	}
	for _, o := range opts {
		o(u)
	}
	return u, nil
}

// preprocess applies SentencePiece's whitespace escaping: spaces → ▁ and, when
// enabled, a leading ▁ (add_dummy_prefix).
func (u *Unigram) preprocess(text string) string {
	s := strings.ReplaceAll(text, " ", spaceMeta)
	if u.dummy {
		s = spaceMeta + s
	}
	return s
}

// Encode segments text into token ids by the 1-best Viterbi over the unigram LM.
func (u *Unigram) Encode(text string) []int {
	runes := []rune(u.preprocess(text))
	n := len(runes)
	if n == 0 {
		return nil
	}
	unkScore := u.minScore - unkPenalty
	best := make([]float64, n+1)
	start := make([]int, n+1) // start position of the piece ending at i
	pid := make([]int, n+1)   // id of that piece
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
	}
	for i := 1; i <= n; i++ {
		lo := max(0, i-u.maxRunes)
		for j := lo; j < i; j++ {
			id, ok := u.id[string(runes[j:i])]
			if !ok {
				continue
			}
			if sc := best[j] + u.pieces[id].Score; sc > best[i] {
				best[i], start[i], pid[i] = sc, j, id
			}
		}
		// <unk> single-character fallback (only wins where no piece covers)
		if sc := best[i-1] + unkScore; sc > best[i] {
			best[i], start[i], pid[i] = sc, i-1, u.unkID
		}
	}
	// backtrack
	var rev []int
	for i := n; i > 0; i = start[i] {
		rev = append(rev, pid[i])
	}
	ids := make([]int, len(rev))
	for i, id := range rev {
		ids[len(rev)-1-i] = id
	}
	return ids
}

// Decode reconstructs text from token ids: concatenate the pieces, map ▁ back to
// spaces, and drop the leading dummy space. Lossless when the vocabulary covered
// every input character (no <unk> was emitted).
func (u *Unigram) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(u.pieces) {
			b.WriteString(u.pieces[id].Piece)
		}
	}
	s := strings.ReplaceAll(b.String(), spaceMeta, " ")
	if u.dummy {
		s = strings.TrimPrefix(s, " ")
	}
	return s
}
