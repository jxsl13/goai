package nlp

import (
	"fmt"
	"strings"
)

// SPM is the SentencePiece BPE-mode encoder of the Llama model family (Llama 1/2,
// Mistral, TinyLlama, …), matching llama.cpp's `llm_tokenizer_spm`: the text is split
// into characters (after ▁ whitespace escaping) and the neighboring pair whose
// concatenation is the highest-scoring vocabulary piece is merged, repeatedly, until
// no adjacent pair forms a known piece; anything not in the vocabulary falls back to
// the <0xNN> byte pieces.
//
// This is NOT the Unigram-LM Viterbi of nlp.Unigram. The llama-family GGUF vocabularies
// carry scores that are NEGATIVE MERGE RANKS (piece id 259+i has score −i), not
// log-probabilities: under Viterbi's score-sum maximization, two short frequent pieces
// ("▁T"=−61 + "he"=−6 = −67) beat the correct single piece ("▁The"=−156) and the
// encoding shatters into sub-word fragments the model never saw in training (§B59 —
// observed as incoherent Mistral-7B output; TinyLlama GGUFs mask the bug because their
// scores are all zero). Greedy best-score-pair merging reproduces the BPE process those
// ranks came from, token-for-token equal to llama.cpp.
type SPM struct {
	id     map[string]int // piece → id
	pieces []UnigramPiece // id → piece+score
	unkID  int            // id emitted when even byte fallback fails
	dummy  bool           // prepend a ▁ (add_dummy_prefix)
	byteID [256]int       // id of the <0xNN> byte piece, −1 when absent

	specials specialSet // markers parsed only by EncodeSpecial (§B60)
}

// SPMOption configures an SPM tokenizer (functional-options idiom, §C12).
type SPMOption func(*SPM)

// WithSPMUnkID sets the id emitted for a byte not covered by the vocabulary's <0xNN>
// pieces (with the standard 256 byte pieces present it is never used).
//
// In plain terms: the "unknown" token id of last resort. Boundary behavior — any valid
// vocab id. SPECIAL VALUE / default: the id of a piece literally named "<unk>" if
// present, else 0 (the SentencePiece convention).
func WithSPMUnkID(id int) SPMOption { return func(s *SPM) { s.unkID = id } }

// WithSPMDummyPrefix toggles prepending a ▁ (the SentencePiece space marker) before
// encoding, so a word at the START of the text tokenizes the same way it would
// mid-sentence.
//
// In plain terms: SentencePiece treats a space as part of the following word; the dummy
// prefix makes "Hello" at the start match " Hello" in the middle. Boundary behavior — a
// boolean; off only for vocabularies trained without add_dummy_prefix.
//
// Default true (research-grounded: SentencePiece's add_dummy_prefix default, which the
// Llama family uses).
func WithSPMDummyPrefix(on bool) SPMOption { return func(s *SPM) { s.dummy = on } }

// NewSPM builds an SPM tokenizer from a scored vocabulary (piece id = index). Scores
// order the merges (higher merges first); for llama-family GGUF vocabularies they are
// negative merge ranks. It errors on an empty vocabulary or duplicate pieces.
func NewSPM(vocab []UnigramPiece, opts ...SPMOption) (*SPM, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("nlp: SPM needs a non-empty vocabulary")
	}
	s := &SPM{
		id:     make(map[string]int, len(vocab)),
		pieces: append([]UnigramPiece(nil), vocab...),
		dummy:  true,
	}
	for i := range s.byteID {
		s.byteID[i] = -1
	}
	for i, p := range vocab {
		if _, dup := s.id[p.Piece]; dup {
			return nil, fmt.Errorf("nlp: SPM duplicate piece %q", p.Piece)
		}
		s.id[p.Piece] = i
		// same <0xHH> gate as Decode (T931): skip the reflection-based fmt.Sscanf for the
		// vast majority of vocab pieces that are not byte tokens (one-time build cost).
		if pc := p.Piece; len(pc) == 6 && pc[0] == '<' && pc[1] == '0' && pc[2] == 'x' {
			var b int
			if n, _ := fmt.Sscanf(pc, "<0x%02X>", &b); n == 1 {
				s.byteID[b] = i
			}
		}
	}
	if id, ok := s.id["<unk>"]; ok {
		s.unkID = id
	}
	for _, o := range opts {
		o(s)
	}
	return s, nil
}

// Encode segments text into token ids by greedy best-score adjacent-pair merging
// (llama.cpp llm_tokenizer_spm semantics; ties break to the leftmost pair). The merge
// runs in O(N log N) via a priority queue over a linked list of symbols (spmBounds) —
// SPM merges over the WHOLE input (no pre-tokenization), so the old full-rescan loop
// was O(N²) and quadratic on long prompts (8 KB: 199 ms → 1 ms; ~600× at 33 KB).
//
// It treats text as LITERAL and never emits a registered special marker's id, so it is
// the right entry point for untrusted input (§B60); [SPM.EncodeSpecial] is the one that
// parses markers. Spaces are escaped to ▁ first — which means a literal ▁ already in
// text is indistinguishable from a space and will decode back as one, silently.
// [ContainsSpaceMeta] detects that input; SentencePiece specifies no escape for it and
// neither does GoAI (see [ContainsSpaceMeta] for why inventing one would be worse).
func (s *SPM) Encode(text string) []int {
	str := strings.ReplaceAll(text, " ", spaceMeta)
	if s.dummy {
		str = spaceMeta + str
	}
	if str == "" {
		return nil
	}
	// Symbols as boundary byte-offsets into str: symbol k spans str[bounds[k]:bounds[k+1]],
	// and since merges only combine ADJACENT symbols, every symbol (and every candidate
	// pair) is a contiguous substring of str. So the merge-candidate is the alloc-free
	// slice str[bounds[i]:bounds[i+2]] instead of the concatenation syms[i]+syms[i+1],
	// which allocated a fresh string per adjacent pair per merge round (§T625, the
	// BPE-merge anti-pattern, here shared with SPM).
	bounds := s.spmBounds(str)
	var ids []int
	for k := 0; k+1 < len(bounds); k++ {
		sym := str[bounds[k]:bounds[k+1]]
		// blocked() again for the degenerate case of a single-character marker, which
		// merging never had to build (§B60); it falls through to byte fallback.
		if id, ok := s.id[sym]; ok && !s.specials.blocked(sym) {
			ids = append(ids, id)
			continue
		}
		for b := bounds[k]; b < bounds[k+1]; b++ { // byte fallback for uncovered characters
			if id := s.byteID[str[b]]; id >= 0 {
				ids = append(ids, id)
			} else {
				ids = append(ids, s.unkID)
			}
		}
	}
	return ids
}

// spmCand is one adjacent-pair merge candidate on the priority queue. lgen/rgen pin the left/right
// symbols' generations at push time so a stale entry (a symbol has since merged) is detected and
// skipped on pop (lazy deletion) — the standard SentencePiece merge queue.
type spmCand struct {
	score       float64
	pos         int // left symbol's start byte offset — the leftmost-wins tie-break
	left, right int // symbol slot indices
	lgen, rgen  int
	id          int
}

// spmLess reports that candidate a outranks b — higher score, or (equal score) the leftmost. This
// is a strict total order (pos is a symbol's unique start offset), so the heap pop order is
// deterministic and matches the naive scan's "first symbol with the maximum score wins".
func spmLess(a, b spmCand) bool {
	if a.score != b.score {
		return a.score > b.score
	}
	return a.pos < b.pos
}

// A hand-written binary max-priority heap over []spmCand. Avoids container/heap's any-boxing (one
// heap alloc per Push — 876 allocs in the profile for a 1 KB input); the typed version pushes into
// the reused backing array with zero per-element allocation.
func spmHeapPush(h []spmCand, c spmCand) []spmCand {
	h = append(h, c)
	for i := len(h) - 1; i > 0; {
		p := (i - 1) / 2
		if !spmLess(h[i], h[p]) {
			break
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
	return h
}

func spmHeapPop(h []spmCand) (spmCand, []spmCand) {
	top := h[0]
	n := len(h) - 1
	h[0] = h[n]
	h = h[:n]
	for i := 0; ; {
		l, r, best := 2*i+1, 2*i+2, i
		if l < n && spmLess(h[l], h[best]) {
			best = l
		}
		if r < n && spmLess(h[r], h[best]) {
			best = r
		}
		if best == i {
			break
		}
		h[i], h[best] = h[best], h[i]
		i = best
	}
	return top, h
}

// spmBounds runs the highest-score-first BPE merge as an O(N log N) priority-queue over a linked list
// of symbols, returning the final symbol byte-boundaries. Replaces the O(N²) full-rescan merge (each
// round re-scanned every adjacent pair — catastrophic on long inputs since SPM merges over the WHOLE
// string, not per pre-token). Segmentation is IDENTICAL to the naive scan: same global-max-score
// choice each step, same leftmost tie-break, so encode parity holds (TestSPM*).
func (s *SPM) spmBounds(str string) []int {
	var starts []int
	for i := range str { // rune-boundary starts
		starts = append(starts, i)
	}
	m := len(starts)
	if m == 0 {
		return []int{0}
	}
	start := make([]int, m)
	end := make([]int, m)
	prev := make([]int, m)
	next := make([]int, m)
	gen := make([]int, m)
	alive := make([]bool, m)
	for k := 0; k < m; k++ {
		start[k] = starts[k]
		if k+1 < m {
			end[k] = starts[k+1]
			next[k] = k + 1
		} else {
			end[k] = len(str)
			next[k] = -1
		}
		prev[k] = k - 1
		alive[k] = true
	}
	var h []spmCand
	tryPush := func(a int) {
		b := next[a]
		if b < 0 {
			return
		}
		merged := str[start[a]:end[b]] // alloc-free map key
		id, ok := s.id[merged]
		if !ok || s.specials.blocked(merged) {
			return
		}
		h = spmHeapPush(h, spmCand{score: s.pieces[id].Score, pos: start[a], left: a, right: b, lgen: gen[a], rgen: gen[b], id: id})
	}
	for k := 0; k < m; k++ {
		tryPush(k)
	}
	for len(h) > 0 {
		var c spmCand
		c, h = spmHeapPop(h)
		a, b := c.left, c.right
		if !alive[a] || !alive[b] || next[a] != b || gen[a] != c.lgen || gen[b] != c.rgen {
			continue // stale (a or b already merged) — lazy delete
		}
		end[a] = end[b] // extend a to cover b
		alive[b] = false
		next[a] = next[b]
		if next[b] >= 0 {
			prev[next[b]] = a
		}
		gen[a]++
		tryPush(a) // a with its new right neighbor
		if prev[a] >= 0 {
			tryPush(prev[a]) // a's left neighbor with the grown a
		}
	}
	// symbol 0 is never a right operand (nothing to its left), so it always survives as the head.
	bounds := make([]int, 0, m+1)
	for n := 0; n >= 0; n = next[n] {
		bounds = append(bounds, start[n])
	}
	return append(bounds, len(str))
}

// Decode reconstructs text from token ids and drops the leading dummy space. Each
// piece is expanded by its KIND, matching llama.cpp's token_to_piece, which switches
// on the token ATTRIBUTE (BYTE / UNKNOWN+CONTROL+USER_DEFINED / NORMAL) rather than
// post-processing one concatenated string:
//
//   - <0xNN> byte pieces → the raw byte, never whitespace-unescaped (a byte-fallback
//     run reconstructs arbitrary UTF-8, and its bytes mean themselves);
//   - registered special markers (see [SPM.AddSpecialTokens]) → VERBATIM. DeepSeek's
//     control tokens are named "<｜begin▁of▁sentence｜>", where U+2581 is part of the
//     NAME; unescaping it would return a corrupted "<｜begin of sentence｜>" for a
//     perfectly correct token id (§B74);
//   - everything else (NORMAL pieces) → ▁ mapped back to spaces.
//
// Losslessness — Decode(Encode(text)) == text — requires BOTH that the vocabulary can
// represent every character (the <0xNN> byte pieces normally guarantee this) and that
// text contains no literal ▁ (U+2581): Encode escapes " " to ▁ with no inverse
// escape, exactly as SentencePiece itself has none, so a literal ▁ arrives at
// segmentation indistinguishable from a space and returns as one. [ContainsSpaceMeta]
// tests for this up front.
func (s *SPM) Decode(ids []int) string {
	var b strings.Builder
	b.Grow(len(ids) * 4) // pre-size (§T929): avoid the log(n) growth-buffer churn; Grow is capacity-only, output byte-identical
	for _, id := range ids {
		if id < 0 || id >= len(s.pieces) {
			if t, ok := s.specials.textForID(id); ok {
				b.WriteString(t) // added/special token appended past the base vocab
			}
			continue
		}
		p := s.pieces[id].Piece
		// BYTE piece <0xHH>: gate the (allocating, reflection-based) fmt.Sscanf behind a
		// cheap prefix+length check. A successful n==1 REQUIRES the "<0x" literal to match,
		// so this is bit-identical — but it skips the Sscanf and its ~5 allocs for the 99%+
		// of tokens that are ordinary subwords, not byte pieces (was 1.55M allocs/decode).
		if len(p) == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' {
			var v int
			if n, _ := fmt.Sscanf(p, "<0x%02X>", &v); n == 1 {
				b.WriteByte(byte(v)) // BYTE: raw, no unescaping
				continue
			}
		}
		if s.specials.blocked(p) {
			b.WriteString(p) // CONTROL / USER_DEFINED: ▁ is part of its name
			continue
		}
		writeUnescapedMeta(&b, p)
	}
	out := b.String()
	if s.dummy {
		out = strings.TrimPrefix(out, " ")
	}
	return out
}

// writeUnescapedMeta writes p into b with every spaceMeta (▁) replaced by a space,
// straight into the builder. It is the alloc-free equivalent of
// b.WriteString(strings.ReplaceAll(p, spaceMeta, " ")) — which allocated a fresh
// string for every ▁-bearing piece (52k allocs in the T931 SPM decode profile).
// Byte-identical: same replacement, same output.
func writeUnescapedMeta(b *strings.Builder, p string) {
	for {
		j := strings.Index(p, spaceMeta)
		if j < 0 {
			b.WriteString(p)
			return
		}
		b.WriteString(p[:j])
		b.WriteByte(' ')
		p = p[j+len(spaceMeta):]
	}
}

// SPMFromGGUF builds the SPM (llama-family SentencePiece BPE) tokenizer from the
// metadata map of a parsed GGUF model file, reading the same keys as UnigramFromGGUF
// (tokenizer.ggml.tokens / .scores / .unknown_token_id). It requires
// tokenizer.ggml.model == "llama".
//
// Use THIS loader — not UnigramFromGGUF — for llama-family models whose GGUF carries
// real SentencePiece scores (negative merge ranks): Viterbi over merge ranks fragments
// the encoding and degrades generation (§B59). UnigramFromGGUF remains correct for true
// Unigram-LM vocabularies ("t5"/UGM), whose scores are log-probabilities.
func SPMFromGGUF(meta map[string]any) (*SPM, error) {
	model, _ := meta[ggufTokModel].(string)
	if model != "llama" {
		return nil, fmt.Errorf("nlp: GGUF tokenizer.ggml.model=%q is not the llama SPM family", model)
	}
	toks, ok := meta[ggufTokTokens].([]any)
	if !ok || len(toks) == 0 {
		return nil, fmt.Errorf("nlp: GGUF %s missing or empty", ggufTokTokens)
	}
	scores, ok := meta[ggufTokScores].([]any)
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF %s missing (SPM orders merges by score)", ggufTokScores)
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
	var opts []SPMOption
	if unk, ok := ggufTokenID(meta[ggufTokUnkID]); ok {
		opts = append(opts, WithSPMUnkID(unk))
	}
	s, err := NewSPM(vocab, opts...)
	if err != nil {
		return nil, err
	}
	// Control / user-defined / unknown tokens become the EncodeSpecial marker set (§B60).
	// Plain Encode is unaffected: it still treats every marker as literal text.
	texts := make([]string, len(vocab))
	for i, p := range vocab {
		texts[i] = p.Piece
	}
	s.AddSpecialTokens(ggufSpecials(meta, texts))
	return s, nil
}
