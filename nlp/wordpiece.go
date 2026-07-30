package nlp

import (
	"fmt"
	"strings"
	"unicode/utf8"
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
	trie         *unigramTrie   // byte trie over the vocabulary for greedy longest-match
	contNode     int32          // trie node reached by descending the continuation prefix
	hasCont      bool           // whether the continuation prefix exists in the trie

	specials specialSet // markers parsed only by EncodeSpecial (§B60)

	// decCont[id] is the piece with the continuation prefix already stripped, and isCont[id] whether
	// it had one. Decode ran HasPrefix and then TrimPrefix per TOKEN — the same comparison twice —
	// for a split fixed at construction, since `continuation` is option-only and pieces never change.
	decCont []string
	isCont  []bool
}

// buildWordPieceTrie inserts every vocabulary piece (including the "##…" continuation
// forms) into a byte trie, sharing common prefixes.
func buildWordPieceTrie(pieces []string) *unigramTrie {
	t := &unigramTrie{edge: make(map[uint64]int32, len(pieces)*4), id: make([]int32, 1, len(pieces)*4)}
	t.id[0] = -1 // root
	for pid, p := range pieces {
		node := int32(0)
		for i := 0; i < len(p); i++ {
			key := uint64(node)<<8 | uint64(p[i])
			child, ok := t.edge[key]
			if !ok {
				child = int32(len(t.id))
				t.id = append(t.id, -1)
				t.edge[key] = child
			}
			node = child
		}
		t.id[node] = int32(pid)
	}
	t.finalize()
	return t
}

// descend walks node down by the bytes of s, returning the reached node (false if a
// byte has no edge).
func (t *unigramTrie) descend(node int32, s string) (int32, bool) {
	for i := 0; i < len(s); i++ {
		child, ok := t.child(node, s[i])
		if !ok {
			return 0, false
		}
		node = child
	}
	return node, true
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
//
// SPECIAL VALUE: the empty string is IGNORED (keeps the current value), matching
// [WithWordPieceMaxChars]'s treatment of a non-positive cap. The marker is what
// separates a mid-word piece from a word-initial one, so an empty one cannot carry
// that meaning: strings.HasPrefix(piece, "") is true for EVERY piece, which made
// [WordPiece.Decode] treat the whole sequence as one continued word and emit it with
// no spaces at all, while Tokenize looked mid-word pieces up unprefixed and collided
// them with the word-initial entries. Since no vocabulary can be built around an
// empty marker, Go's zero-value string is read as "not configured" rather than
// applied. A non-empty prefix is stored verbatim, so every working configuration is
// unaffected.
func WithWordPieceContinuation(p string) WordPieceOption {
	return func(w *WordPiece) {
		if p == "" {
			return
		}
		w.continuation = p
	}
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
	w.trie = buildWordPieceTrie(w.pieces)
	w.contNode, w.hasCont = w.trie.descend(0, w.continuation)
	w.buildDecodeTable()
	return w, nil
}

// buildDecodeTable splits every piece on the continuation prefix once. Construction-time only:
// `continuation` is set by option inside NewWordPiece and pieces never change, and Decode does not
// consult specials, so unlike SPM and Unigram this table needs no rebuild.
func (w *WordPiece) buildDecodeTable() {
	cont := make([]string, len(w.pieces))
	is := make([]bool, len(w.pieces))
	for id, p := range w.pieces {
		if strings.HasPrefix(p, w.continuation) {
			is[id] = true
			cont[id] = strings.TrimPrefix(p, w.continuation) // aliases p's tail, no allocation
		}
	}
	w.decCont, w.isCont = cont, is
}

// Encode tokenizes text into ids: it splits on whitespace and applies greedy longest-match-first
// WordPiece to each word.
func (w *WordPiece) Encode(text string) []int {
	var out []int
	for word := range strings.FieldsSeq(text) { // FieldsSeq avoids allocating the full []string of words
		out = w.encodeWordInto(out, word)
	}
	return out
}

// encodeWordInto runs MaxMatch on a single word, appending its piece ids directly to
// out (returned regrown). Emitting into the caller's slice instead of a fresh per-word
// []int — the old encodeWord built one and Encode spread it in — drops one heap
// allocation per word (T953). On an unmatchable position the whole word becomes a
// single [UNK]: any pieces already appended for this word are first rolled back
// (out[:wordStart]), exactly as the old encodeWord discarded its partial result before
// returning []int{unkID}.
func (w *WordPiece) encodeWordInto(out []int, word string) []int {
	if utf8.RuneCountInString(word) > w.maxChars {
		return append(out, w.unkID)
	}
	// Byte-trie greedy longest-match. From each byte start bs, descend the trie one byte
	// at a time (from the root, or from the "##" continuation node when bs>0), tracking
	// the LONGEST piece that ends at a rune boundary — one pass instead of re-building and
	// re-hashing string(runes[start:end]) (+ "##") for every candidate length (§T625). The
	// rune-boundary guard keeps candidates rune-aligned, exactly as the old []rune slice did.
	hasSpecials := len(w.specials.byText) > 0
	wordStart := len(out) // rollback point if this word proves unmatchable
	bs := 0
	for bs < len(word) {
		node, have := int32(0), true
		if bs > 0 {
			node, have = w.contNode, w.hasCont
		}
		matchedID, matchedEnd := -1, -1
		if have {
			cur := node
			for be := bs; be < len(word); be++ {
				child, ok := w.trie.child(cur, word[be])
				if !ok {
					break // no piece shares this prefix
				}
				cur = child
				if be+1 < len(word) && !utf8.RuneStart(word[be+1]) {
					continue // mid-rune: a piece may only end on a rune boundary
				}
				id := w.trie.id[cur]
				if id < 0 {
					continue // this prefix is not itself a piece; a longer one may be
				}
				// Registered special markers are never produced by ordinary segmentation
				// (§B60): so a blocked special is skipped, and a shorter/longer non-blocked
				// piece is taken instead. EncodeSpecial is the only path that emits them.
				if hasSpecials {
					sub := word[bs : be+1]
					if bs > 0 {
						sub = w.continuation + sub
					}
					if w.specials.blocked(sub) {
						continue
					}
				}
				matchedID, matchedEnd = int(id), be+1
			}
		}
		if matchedID < 0 {
			return append(out[:wordStart], w.unkID) // no piece here ⇒ whole word is one unk
		}
		out = append(out, matchedID)
		bs = matchedEnd
	}
	return out
}

// Decode concatenates the pieces of ids: a continuation piece ("##…") is appended to the current
// word (prefix stripped), a non-continuation piece starts a new space-separated word.
func (w *WordPiece) Decode(ids []int) string {
	var b strings.Builder
	b.Grow(len(ids) * 4) // pre-size (§T929): avoid the log(n) growth-buffer churn; Grow is capacity-only, output byte-identical
	for _, id := range ids {
		if id < 0 || id >= len(w.pieces) {
			if t, ok := w.specials.textForID(id); ok {
				if b.Len() > 0 {
					b.WriteByte(' ')
				}
				b.WriteString(t) // added/special token appended past the base vocab
			}
			continue
		}
		p := w.pieces[id]
		// A continuation piece joins the previous word; at position 0 there is nothing to join, and
		// the original wrote the piece WITH its prefix there — so both forms are kept.
		if w.isCont != nil {
			if w.isCont[id] && b.Len() > 0 {
				b.WriteString(w.decCont[id])
				continue
			}
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(p)
			continue
		}
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
