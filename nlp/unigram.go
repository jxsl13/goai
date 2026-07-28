package nlp

import (
	"fmt"
	"math"
	"strings"
	"sync"
)

// spaceMeta is SentencePiece's whitespace meta-symbol ▁ (U+2581, LOWER ONE EIGHTH
// BLOCK): spaces are escaped to it so segmentation carries word boundaries and
// detokenization can put them back. The escaping has no INVERSE, so it is lossless
// only for input that contains no literal ▁ — see [ContainsSpaceMeta].
const spaceMeta = "▁"

// SpaceMeta is SentencePiece's whitespace meta-symbol ▁ (U+2581, LOWER ONE EIGHTH
// BLOCK) — the character [Unigram] and [SPM] substitute for a space before
// segmenting, and map back to a space when decoding.
//
// In plain terms: SentencePiece has no concept of "words separated by spaces". It
// rewrites every space as this little block character and treats it as an ordinary
// part of the following word, which is why tokens print as "▁hello". You mostly only
// need the constant to check whether your own text already contains one — see
// [ContainsSpaceMeta].
const SpaceMeta = spaceMeta

// ContainsSpaceMeta reports whether text contains a literal ▁ (U+2581), the one class
// of input the SentencePiece tokenizers ([Unigram], [SPM]) cannot round-trip.
//
// In plain terms: ▁ is the stand-in for a space, so a ▁ you typed yourself is
// indistinguishable from a space you typed — encoding "a▁b" and "a b" gives the same
// token ids, and both decode to "a b". Nothing errors; the character is just quietly
// gone. Call this before encoding if that would matter.
//
// Professional: ▁ is SentencePiece's escape character and the escaping has no inverse,
// in the reference implementation as much as here (its default nmt_nfkc normalizer
// maps U+2581 to U+0020 outright, and byte fallback never fires because ▁ is always in
// the vocabulary). The published guarantee, Decode(Encode(Normalize(t))) == Normalize(t),
// holds only because Normalize is itself where the information is lost — the stronger
// claim, Decode(Encode(t)) == t, is false for this input. GoAI does not invent an escape
// that would disagree with SentencePiece, HuggingFace and llama.cpp about what "a▁b"
// tokenizes to; it makes the case checkable instead:
//
//	if nlp.ContainsSpaceMeta(text) {
//		// this text will come back with its ▁ turned into spaces
//	}
//
// Two things this deliberately does NOT flag, because neither is lossy: a ▁ inside a
// registered special marker's name (DeepSeek's "<｜begin▁of▁sentence｜>"), which
// EncodeSpecial emits as one id and Decode returns verbatim; and the ▁ that appears in
// the PIECES of an ordinary encoding, which is just the escaped form of your spaces.
// It answers a question about the INPUT text only.
func ContainsSpaceMeta(text string) bool { return strings.Contains(text, spaceMeta) }

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
// (add_dummy_prefix) before segmentation; Decode maps ▁ back to spaces and drops the
// dummy. Input is assumed already Unicode-normalized (SentencePiece applies NFKC
// upstream; this tokenizer segments the string it is given).
//
// # Round-tripping is conditional, and one condition cannot be lifted
//
// Decode(Encode(text)) == text needs the vocabulary to cover every character of text
// (else <unk> is emitted and the character is gone) AND text to contain no literal ▁
// (U+2581). The second condition is inherited from SentencePiece and is NOT fixable
// here: ▁ is the escape character, so escaping "a b" and passing through "a▁b" both
// yield "▁a▁b" and the same ids. The reference implementation goes further and maps a
// literal U+2581 to a space outright — its default nmt_nfkc normalizer lists 0x2581
// among the "code points considered as whitespace" (builder.cc) — and its Decode is an
// unconditional replace of ▁ with " ". Byte fallback never rescues it either, because
// ▁ is always in the vocabulary and so never unknown. No escape is specified by
// SentencePiece, HuggingFace tokenizers, or llama.cpp, and inventing one here would
// make GoAI disagree with all three about what "a▁b" tokenizes to — a worse failure
// than the one it fixes. So the loss stays, and [ContainsSpaceMeta] makes it
// DETECTABLE instead of silent.
//
// The lossy case is narrower than it looks, because it applies to ORDINARY text only.
// A ▁ inside a special marker's name — DeepSeek's "<｜begin▁of▁sentence｜>" — is
// handled: register it with [Unigram.AddSpecialTokens], and [Unigram.EncodeSpecial]
// emits its id as a unit while [Unigram.Decode] returns its name verbatim.
type Unigram struct {
	id       map[string]int // piece → id
	pieces   []UnigramPiece // id → piece+score
	unkID    int            // id emitted for an unmatched character
	minScore float64        // min piece score (base of the unk penalty)
	maxRunes int            // longest piece in runes (bounds the DP inner loop)
	dummy    bool           // prepend a ▁ (add_dummy_prefix)
	trie     *unigramTrie   // byte trie over the vocabulary for the Viterbi DP

	specials specialSet // markers parsed only by EncodeSpecial (§B60)
}

// unigramTrie is a byte trie over the vocabulary pieces. The Viterbi DP walks it
// forward from each start position, descending one byte at a time: extending a match
// from length L to L+1 is a single edge step reusing the length-L node, instead of
// re-hashing the whole length-(L+1) substring in the piece map. It also prunes — once
// a prefix has no outgoing edge, no longer piece can start there, so the inner loop
// stops early. Edges live in ONE flat map keyed by (parentNode<<8 | byte); id[node] is
// the piece id ending at that node, or -1.
type unigramTrie struct {
	edge map[uint64]int32 // build-time only; niled after finalize
	id   []int32
	// CSR child index (built by finalize): node's children live in
	// childByte/childNode[childOff[node]:childOff[node+1]], sorted by byte so
	// child() can early-exit. Replaces the per-descend map[uint64]int32 hash.
	childOff  []int32
	childByte []byte
	childNode []int32
}

// finalize converts the build-time edge map into the CSR child index and frees
// the map. Node ids and id[] are unchanged, so token ids are bit-identical.
func (t *unigramTrie) finalize() {
	numNodes := len(t.id)
	t.childOff = make([]int32, numNodes+1)
	for key := range t.edge {
		t.childOff[int32(key>>8)+1]++
	}
	for i := 0; i < numNodes; i++ {
		t.childOff[i+1] += t.childOff[i]
	}
	ne := t.childOff[numNodes]
	t.childByte = make([]byte, ne)
	t.childNode = make([]int32, ne)
	cur := make([]int32, numNodes)
	copy(cur, t.childOff[:numNodes])
	for key, node := range t.edge {
		parent := int32(key >> 8)
		pos := cur[parent]
		t.childByte[pos] = byte(key)
		t.childNode[pos] = node
		cur[parent]++
	}
	// sort each node's (small) child run ascending by byte for early-exit lookup
	for i := 0; i < numNodes; i++ {
		lo, hi := t.childOff[i], t.childOff[i+1]
		for a := lo + 1; a < hi; a++ {
			bb, nn := t.childByte[a], t.childNode[a]
			k := a - 1
			for k >= lo && t.childByte[k] > bb {
				t.childByte[k+1], t.childNode[k+1] = t.childByte[k], t.childNode[k]
				k--
			}
			t.childByte[k+1], t.childNode[k+1] = bb, nn
		}
	}
	t.edge = nil
}

// child returns the node reached from `node` by byte `b` (false if none). A
// contiguous, sorted, cache-friendly scan over the node's children replaces the
// (node<<8|byte) map hash on the tokenizer's hot descent.
func (t *unigramTrie) child(node int32, b byte) (int32, bool) {
	lo, hi := t.childOff[node], t.childOff[node+1]
	for i := lo; i < hi; i++ {
		cb := t.childByte[i]
		if cb == b {
			return t.childNode[i], true
		}
		if cb > b {
			break
		}
	}
	return 0, false
}

// buildUnigramTrie inserts every piece byte-by-byte, sharing common prefixes.
func buildUnigramTrie(pieces []UnigramPiece) *unigramTrie {
	t := &unigramTrie{edge: make(map[uint64]int32, len(pieces)*4), id: make([]int32, 1, len(pieces)*4)}
	t.id[0] = -1 // root
	for pid, p := range pieces {
		node := int32(0)
		for i := 0; i < len(p.Piece); i++ {
			key := uint64(node)<<8 | uint64(p.Piece[i])
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
	u.trie = buildUnigramTrie(u.pieces)
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

// unigramScratch holds Encode's Viterbi DP working arrays, reused across calls
// (T951): off (rune-boundary byte offsets), best (DP scores), start/pid
// (backpointers) and rev (the reversed-ids scratch). None escape Encode — only the
// freshly-built ids slice does — so pooling them is bit-identical, provided best[0]
// is re-zeroed each call (the pooled buffer is dirty and the DP reads best[0] as the
// empty-prefix score; every other cell is written before it is read).
type unigramScratch struct {
	off, start, pid, rev []int
	best                 []float64
}

var unigramScratchPool = sync.Pool{New: func() any { return new(unigramScratch) }}

// Encode segments text into token ids by the 1-best Viterbi over the unigram LM.
//
// It treats text as LITERAL and never emits a registered special marker's id, so it is
// the right entry point for untrusted input; [Unigram.EncodeSpecial] is the one
// that parses markers. Spaces are escaped to ▁ first — which means a literal ▁ already
// in text is indistinguishable from a space and will decode back as one, silently.
// [ContainsSpaceMeta] detects that input; the [Unigram] type doc explains why no
// escape is possible.
func (u *Unigram) Encode(text string) []int {
	s := u.preprocess(text)
	// Rune-boundary byte offsets: off[k] is the byte index of rune k, off[n] == len(s).
	// The DP then slices candidate pieces as s[off[j]:off[i]] — a substring that SHARES
	// s's backing (no allocation), so the u.id lookup and blocked() check cost no per-
	// candidate string conversion (the old string(runes[j:i]) allocated O(runes·maxlen)
	// strings just to key the map — the §T625 anti-pattern).
	sp := unigramScratchPool.Get().(*unigramScratch)
	defer unigramScratchPool.Put(sp)
	off := sp.off[:0]
	for bi := range s {
		off = append(off, bi)
	}
	n := len(off)
	off = append(off, len(s))
	sp.off = off // retain the (possibly grown) backing for the next call
	if n == 0 {
		return nil
	}
	unkScore := u.minScore - unkPenalty
	if cap(sp.best) < n+1 { // grow the DP trio together, mirroring radixScratch (T944)
		sp.best = make([]float64, n+1)
		sp.start = make([]int, n+1)
		sp.pid = make([]int, n+1)
	}
	best := sp.best[:n+1]
	start := sp.start[:n+1] // start position of the piece ending at i
	pid := sp.pid[:n+1]     // id of that piece
	best[0] = 0             // pooled buffer is dirty; the DP reads best[0] as the empty-prefix score
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
	}
	// Forward Viterbi over the byte trie. For each start j (increasing) extend pieces
	// to ends i>j by descending the trie one rune at a time, then take the <unk>
	// single-character step j→j+1. best[j] is final by the time j is used as a start
	// (every update to it comes from a start < j), and for any end i the updates arrive
	// in exactly the old order — pieces from j=lo..i-1 ascending, then the <unk> from
	// i-1 last — so the strict-`>` tie-break, and thus the segmentation, is unchanged.
	tr := u.trie
	for j := 0; j < n; j++ {
		if bj := best[j]; bj != math.Inf(-1) {
			node := int32(0)
			maxI := min(n, j+u.maxRunes)
		extend:
			for i := j + 1; i <= maxI; i++ {
				for b := off[i-1]; b < off[i]; b++ { // descend the bytes of rune i-1
					child, has := tr.child(node, s[b])
					if !has {
						break extend // no piece shares this prefix → stop extending
					}
					node = child
				}
				id := tr.id[node]
				if id < 0 {
					continue // this prefix is not itself a piece; a longer one may be
				}
				piece := s[off[j]:off[i]]
				if u.specials.blocked(piece) {
					continue
				}
				if sc := bj + u.pieces[id].Score; sc > best[i] {
					best[i], start[i], pid[i] = sc, j, int(id)
				}
			}
		}
		// <unk> single-character fallback (only wins where no piece covers)
		if sc := best[j] + unkScore; sc > best[j+1] {
			best[j+1], start[j+1], pid[j+1] = sc, j, u.unkID
		}
	}
	// backtrack
	rev := sp.rev[:0]
	for i := n; i > 0; i = start[i] {
		rev = append(rev, pid[i])
	}
	sp.rev = rev // retain the (possibly grown) backing for the next call
	ids := make([]int, len(rev))
	for i, id := range rev {
		ids[len(rev)-1-i] = id
	}
	return ids
}

// Decode reconstructs text from token ids: map each piece's ▁ back to spaces,
// concatenate, and drop the leading dummy space.
//
// Registered special markers (see [Unigram.AddSpecialTokens]) are emitted VERBATIM,
// with no ▁→space mapping. That distinction is not cosmetic: DeepSeek's control
// tokens are literally named "<｜begin▁of▁sentence｜>", where the U+2581 belongs to
// the marker's NAME rather than standing in for a space. Unescaping it would hand
// back "<｜begin of sentence｜>" — a corrupted marker — for a perfectly correct token
// id (§B74).
//
// This mirrors llama.cpp's token_to_piece, which switches on the token's ATTRIBUTE
// rather than post-processing one joined string: UNKNOWN, CONTROL and USER_DEFINED are
// copied straight out, and only NORMAL tokens get llama_unescape_whitespace. Those are
// the same three GGUF token types this package registers as markers (see ggufSpecials),
// so the two agree token for token.
//
// Losslessness — Decode(Encode(text)) == text — requires BOTH of:
//
//   - the vocabulary covers every character of text (no <unk> was emitted), and
//   - text contains no literal ▁ (U+2581). Encode escapes " " to ▁ but has no
//     inverse escape, exactly as SentencePiece itself has none, so a literal ▁ in
//     the input is indistinguishable from a space by the time segmentation runs and
//     comes back as a space. [ContainsSpaceMeta] tests for this up front.
func (u *Unigram) Decode(ids []int) string {
	var b strings.Builder
	b.Grow(len(ids) * 4) // pre-size (§T929): avoid the log(n) growth-buffer churn; Grow is capacity-only, output byte-identical
	for _, id := range ids {
		if id < 0 || id >= len(u.pieces) {
			if t, ok := u.specials.textForID(id); ok {
				b.WriteString(t) // added/special token appended past the base vocab
			}
			continue
		}
		p := u.pieces[id].Piece
		if u.specials.blocked(p) {
			b.WriteString(p) // control/user-defined marker: ▁ is part of its name
			continue
		}
		writeUnescapedMeta(&b, p)
	}
	s := b.String()
	if u.dummy {
		s = strings.TrimPrefix(s, " ")
	}
	return s
}
