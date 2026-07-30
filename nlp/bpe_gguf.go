package nlp

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
	"unsafe"
)

// bytesToUnicode builds the reversible GPT-2 byte→Unicode map (Radford et al. 2019,
// OpenAI gpt-2 encoder.py bytes_to_unicode; §R90). Printable byte ranges (33–126,
// 161–172, 174–255) map to the same code point; every other byte b maps to 256+n for
// n = 0,1,… assigned in byte order (so space 0x20→U+0120 'Ġ', newline 0x0A→U+010A
// 'Ċ'). This lets a byte-level BPE work over ordinary printable strings while staying
// losslessly invertible. Returns the byte→rune table and the rune→byte inverse.
func bytesToUnicode() (b2u [256]rune, u2b map[rune]byte) {
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	u2b = make(map[rune]byte, 256)
	n := 0
	for b := range 256 {
		r := rune(b)
		if !printable(b) {
			r = rune(256 + n)
			n++
		}
		b2u[b] = r
		u2b[r] = byte(b)
	}
	return b2u, u2b
}

// BPETokenizer is a GPT-2 / HuggingFace byte-level BPE tokenizer (Sennrich et al. 2016
// BPE §R33; GPT-2 byte-level variant, Radford 2019, §R90) — the tokenizer of the
// Llama-3, Qwen and Mistral GGUF model families ("gpt2"/"bpe"). Text is pre-tokenized
// (the GPT-2 regex), each pre-token's raw bytes are mapped through bytesToUnicode, and
// BPE merges are applied by RANK: the vocabulary's merge list is an ordered set of
// symbol pairs, and at each step the adjacent pair with the lowest rank (earliest in
// the list) is merged, until none remains. Because all 256 byte code points are base
// tokens, decode∘encode is byte-exact for any input (§V15).
type BPETokenizer struct {
	vocab     map[string]int    // byte-mapped symbol → id
	decoder   map[int]string    // id → byte-mapped symbol
	mergeRank map[[2]string]int // {left, right} → merge priority (list index). A struct key, not
	//                            "left right", so the O(L²) merge scan looks up a pair with ZERO
	//                            string allocation (no per-pair concat, no string() — T625 class).
	b2u      [256]rune
	u2b      map[rune]byte
	decSlice []string // id → symbol as a dense slice (Decode fast path; nil until built)
	u2bSlice []int16  // rune → byte (−1 = unmapped) for Decode's byte-inversion, sized to the max u2b rune
	// rawSlice holds each id's FINAL raw bytes — decSlice[id] already pushed through u2b — so
	// Decode never has to rebuild the byte-mapped text and decode it back. Fixed at construction:
	// decSlice[id] is a constant string and u2b never changes after NewBPE.
	rawSlice [][]byte
	unkID    int
	hasUnk   bool
	specials specialSet // markers parsed only by EncodeSpecial (§B60)
}

// NewBPE builds a byte-level BPE tokenizer from a vocabulary (byte-mapped token strings
// indexed by id) and an ordered merges list ("left right" per entry, lowest index =
// highest priority) — the form GGUF stores. It errors on an empty vocabulary.
func NewBPE(vocab []string, merges []string, opts ...BPEOption) (*BPETokenizer, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("nlp: BPE needs a non-empty vocabulary")
	}
	t := &BPETokenizer{
		vocab:     make(map[string]int, len(vocab)),
		decoder:   make(map[int]string, len(vocab)),
		mergeRank: make(map[[2]string]int, len(merges)),
	}
	t.b2u, t.u2b = bytesToUnicode()
	for id, tok := range vocab {
		t.vocab[tok] = id
		t.decoder[id] = tok
	}
	for i, m := range merges {
		l, r, ok := strings.Cut(m, " ")
		if !ok {
			continue // no separator: a malformed merge that the {left,right} lookup can never match
		}
		key := [2]string{l, r}
		if _, dup := t.mergeRank[key]; !dup {
			t.mergeRank[key] = i // earlier = higher priority
		}
	}
	for _, o := range opts {
		o(t)
	}
	t.buildDecodeSlices()
	return t, nil
}

// BPEOption configures a BPETokenizer (functional-options idiom, §C12).
type BPEOption func(*BPETokenizer)

// WithBPEUnkID sets the id emitted when a symbol has no entry in the vocabulary.
//
// In plain terms: the "unknown token" id. This is RARE for a byte-level BPE vocabulary (GPT-2
// style), which can represent any byte and so has no true unknowns. Boundary behavior — any
// valid vocab id. SPECIAL VALUE / default: unset — unknown symbols are simply SKIPPED rather
// than mapped to an unk id (the right behavior for a complete byte-level vocab, §R90); set this
// only for a vocabulary that genuinely needs an explicit unknown token.
func WithBPEUnkID(id int) BPEOption {
	return func(t *BPETokenizer) { t.unkID, t.hasUnk = id, true }
}

// ggufPart is one token boundary during the GGUF/JSON byte-level merge: start = its
// byte offset into the immutable mapped string. A part k spans mapped[parts[k].start :
// parts[k+1].start], and rank caches the merge rank of the pair (part k, part k+1) so
// each merge recomputes only the two neighbours it touches (mirrors Tokenizer.bpePart).
// Unlike bpePart it carries no id: GGUF keeps mergeRank and vocab as SEPARATE maps
// (a merge rank is a list index, NOT a token id), so the final id is resolved by a
// vocab lookup at emit, not tracked through the merge.
type ggufPart struct{ start, rank int }

// pairRankAt returns the merge rank of the adjacent pair (part k, part k+1), or
// math.MaxInt when part k has no right neighbour or the pair is not in mergeRank. The
// two operands are SUBSTRINGS of mapped, so building the [2]string map key allocates
// nothing (the Go map lookup hashes the substrings in place) — the whole merge does
// ZERO string allocations (T625/§T938 class).
func (t *BPETokenizer) pairRankAt(mapped string, parts []ggufPart, k int) int {
	if k+2 >= len(parts) {
		return math.MaxInt
	}
	if rk, ok := t.mergeRank[[2]string{
		mapped[parts[k].start:parts[k+1].start],
		mapped[parts[k+1].start:parts[k+2].start],
	}]; ok {
		return rk
	}
	return math.MaxInt
}

// bpeInto merges one pre-token's byte-mapped string into symbols by rank, APPENDING the
// piece's token ids to out and returning the grown out plus the parts scratch for reuse on
// the next piece (caller-owned scratch, mirrors Tokenizer.bpeMergeInto, §T939). Encode
// threads one out (the whole result) and one parts scratch across every piece, so a text of
// N pieces does O(1) heap allocations for the merge instead of O(N) — the per-piece parts
// slice + the returned ids slice were the residual ~68k allocs/op after T938's byte-offset
// merge. Numerically identical to the old bpe — same merge order, same boundaries.
//
// Byte-OFFSET merge (§T938): parts are boundary offsets into the immutable mapped string,
// never per-rune substrings, and a merge DROPS a boundary rather than concatenating two
// strings. Every pair rank comes from substrings of mapped (pairRankAt), so the merge does
// ZERO string allocations. Merge order — repeatedly the LEFTMOST lowest-rank adjacent pair —
// and the final token boundaries are bit-identical to the old scan (§V15/§V16), so token ids
// are unchanged for every input. mapped is NOT retained: only map-key lookups on mapped[a:b]
// substrings, which never retain the key — so the caller may safely reuse its mapped builder.
func (t *BPETokenizer) bpeInto(mapped string, out []int, parts []ggufPart) ([]int, []ggufPart) {
	// parts[k].start = byte offset of part k; the trailing sentinel {len(mapped)}
	// makes part k span mapped[parts[k].start : parts[k+1].start]. Initial parts are
	// per-RUNE — range mapped yields each rune's leading byte offset. parts[:0] reuses the
	// caller's backing array (grows only when a piece is longer than any previously seen).
	parts = parts[:0]
	for i := range mapped {
		parts = append(parts, ggufPart{start: i, rank: math.MaxInt})
	}
	parts = append(parts, ggufPart{start: len(mapped), rank: math.MaxInt})

	// Seed every adjacent-pair rank once, tracking the leftmost lowest. A strict <
	// (first wins on ties) with minRank = MaxInt reproduces the old ascending scan
	// exactly: an unmergeable pair (rank MaxInt) is never selected.
	minRank, minI := math.MaxInt, -1
	for k := 0; k < len(parts)-1; k++ {
		parts[k].rank = t.pairRankAt(mapped, parts, k)
		if parts[k].rank < minRank {
			minRank, minI = parts[k].rank, k
		}
	}
	for minRank != math.MaxInt {
		i := minI
		parts = append(parts[:i+1], parts[i+2:]...) // merge parts i,i+1: drop boundary i+1
		if i > 0 {                                  // only the two pairs touching the merge changed
			parts[i-1].rank = t.pairRankAt(mapped, parts, i-1)
		}
		parts[i].rank = t.pairRankAt(mapped, parts, i)
		minRank, minI = math.MaxInt, -1 // rescan the cached ranks for the new leftmost min
		for k := 0; k < len(parts)-1; k++ {
			if parts[k].rank < minRank {
				minRank, minI = parts[k].rank, k
			}
		}
	}

	for k := 0; k < len(parts)-1; k++ {
		if id, ok := t.vocab[mapped[parts[k].start:parts[k+1].start]]; ok {
			out = append(out, id)
		} else if t.hasUnk {
			out = append(out, t.unkID)
		}
	}
	return out, parts
}

// Encode turns text into token ids: GPT-2 pre-tokenization → byte→Unicode mapping →
// rank-ordered BPE merges → vocab lookup.
//
// It ranges gpt2SplitSeq (an iterator, no pieces []string) and threads one ids (the whole
// result), one parts merge scratch, AND one mapped byte→Unicode buffer across every piece
// (§T939/§T954, mirrors Tokenizer.Encode). mapped is a reused []byte truncated per piece; the
// string handed to bpeInto ALIASES it via unsafe.String (no copy). That is sound because bpeInto
// only does map-key lookups on mapped[a:b] substrings and never retains them, so the alias is
// dead before the next piece overwrites the buffer — turning the old per-piece strings.Builder
// allocation (Reset nils its backing, so each piece grew a fresh one: ~82% of Encode's allocs)
// into O(1) heap traffic for the whole text. mapped stays contiguous so the merge key
// [2]string{mapped[a:b],mapped[b:c]} aliases its backing and the pair lookup allocates nothing.
func (t *BPETokenizer) Encode(text string) []int {
	// Reserve for the token count up front. ids grew by doubling from empty on every call,
	// which an alloc_space profile put at 99.93% of the bytes this encode allocates — 616MB
	// across 200 iterations of the GGUF benchmark. BPE emits at most one token per byte and
	// typically about one per three to four bytes of English, so len(text)/3 covers the common
	// case in one allocation while a token-dense input simply falls back to doubling from a
	// higher floor. The +8 keeps very short inputs off the growth path entirely.
	//
	// This is an ESTIMATE, not a bound, which is why it is a capacity rather than a length: over
	// -reserving wastes a fraction of one allocation, under-reserving costs the doublings it
	// would otherwise have paid anyway.
	ids := make([]int, 0, len(text)/3+8)
	var parts []ggufPart // merge scratch, reused across pieces
	var mapped []byte    // byte→Unicode buffer, reused (truncated) across pieces
	for piece := range gpt2SplitSeq(text) {
		mapped = mapped[:0]
		for i := 0; i < len(piece); i++ {
			mapped = utf8.AppendRune(mapped, t.b2u[piece[i]]) // map each raw byte
		}
		// unsafe.String aliases the reused buffer (no copy); safe because bpeInto never
		// retains the string past the call, so the next piece may overwrite mapped (§T954).
		ids, parts = t.bpeInto(unsafe.String(unsafe.SliceData(mapped), len(mapped)), ids, parts)
	}
	return ids
}

// buildDecodeSlices flattens the two id/rune-keyed maps Decode walks into dense slices:
// decoder (dense ids 0..vocab−1) → []string, and u2b (runes ≤ ~323 with gaps) → []int16
// with −1 for the unmapped runes. Both replace a per-element map hash in Decode with a
// bounds-checked slice load; gaps/out-of-range resolve to the same skip the maps' !ok did.
func (t *BPETokenizer) buildDecodeSlices() {
	maxID := -1
	for id := range t.decoder {
		if id > maxID {
			maxID = id
		}
	}
	ds := make([]string, maxID+1)
	for id, tok := range t.decoder {
		ds[id] = tok
	}
	t.decSlice = ds

	maxR := rune(-1)
	for r := range t.u2b {
		if r > maxR {
			maxR = r
		}
	}
	u2s := make([]int16, maxR+1)
	for i := range u2s {
		u2s[i] = -1
	}
	for r, b := range t.u2b {
		u2s[r] = int16(b)
	}
	t.u2bSlice = u2s

	// Per-id raw bytes, built once. Decode used to write every symbol into a strings.Builder, take
	// its String(), and then RUNE-DECODE the whole output to invert the byte map — two full passes
	// over the result plus the intermediate itself, all to recompute a per-id constant. One slab
	// backs every row so this is one allocation, and the rows are capped so no append can reach
	// into the next.
	total := 0
	for _, sym := range ds {
		total += len(sym)
	}
	slab := make([]byte, 0, total)
	raw := make([][]byte, len(ds))
	for id, sym := range ds {
		start := len(slab)
		for _, r := range sym {
			if uint(r) < uint(len(u2s)) {
				if b := u2s[r]; b >= 0 {
					slab = append(slab, byte(b))
				}
			}
		}
		raw[id] = slab[start:len(slab):len(slab)]
	}
	t.rawSlice = raw
}

// Decode reconstructs the original text: token ids → byte-mapped symbols → invert the
// byte→Unicode map back to raw bytes. Byte-exact for any input the vocabulary covers.
func (t *BPETokenizer) Decode(ids []int) string {
	// Fast path: every id's raw bytes are already known, so this is a size prepass plus one append
	// pass. What it replaces is a strings.Builder over the byte-mapped symbols, a String() to
	// materialize it, and a full UTF-8 rune decode of that intermediate to invert the byte map —
	// two passes over the output and an allocation, none of which depended on the ids.
	//
	// Byte-identical: the fused loop converted rune by rune and dropped unmapped runes, and so does
	// the table build; vocabulary symbols are whole UTF-8 strings, so no rune ever straddles a
	// token boundary and per-symbol conversion is the same conversion.
	if rs := t.rawSlice; rs != nil {
		total := 0
		for _, id := range ids {
			if uint(id) < uint(len(rs)) {
				total += len(rs[id])
			}
		}
		out := make([]byte, 0, total)
		for _, id := range ids {
			if uint(id) < uint(len(rs)) {
				out = append(out, rs[id]...)
			}
		}
		return string(out)
	}
	var mapped strings.Builder
	mapped.Grow(len(ids) * 4) // pre-size (§T929): skip the log(n) builder growth churn
	if ds := t.decSlice; ds != nil {
		for _, id := range ids {
			if uint(id) < uint(len(ds)) {
				mapped.WriteString(ds[id])
			}
		}
	} else {
		//perfscan:ignore PS3003 decSlice is the map→slice fast path; this is the defensive fallback
		for _, id := range ids {
			if s, ok := t.decoder[id]; ok {
				mapped.WriteString(s)
			}
		}
	}
	m := mapped.String()
	out := make([]byte, 0, len(m)) // one byte per rune ≤ len(m) bytes — pre-sized, no append growth
	if u2s := t.u2bSlice; u2s != nil {
		for _, r := range m {
			if uint(r) < uint(len(u2s)) {
				if b := u2s[r]; b >= 0 {
					out = append(out, byte(b))
				}
			}
		}
	} else {
		//perfscan:ignore PS3003 u2bSlice is the map→slice fast path; this is the defensive fallback
		for _, r := range m {
			if b, ok := t.u2b[r]; ok {
				out = append(out, b)
			}
		}
	}
	return string(out)
}

// BPEFromGGUF builds a byte-level BPE tokenizer from the metadata map of a parsed GGUF
// model file (§R88): tokenizer.ggml.model must be "gpt2" or "bpe", with
// tokenizer.ggml.tokens (the byte-mapped vocabulary) and tokenizer.ggml.merges (the
// ordered "left right" merge rules). This wires a real .gguf model's embedded BPE
// tokenizer — the Llama-3 / Qwen / Mistral family — to NewBPE, so weights and tokenizer
// load from one file. SentencePiece/Unigram models ("t5"/"llama") use UnigramFromGGUF.
func BPEFromGGUF(meta map[string]any) (*BPETokenizer, error) {
	model, _ := meta[ggufTokModel].(string)
	switch model {
	case "gpt2", "bpe":
	case "t5", "llama":
		return nil, fmt.Errorf("nlp: GGUF tokenizer.ggml.model=%q is SentencePiece/Unigram, use UnigramFromGGUF", model)
	default:
		return nil, fmt.Errorf("nlp: GGUF tokenizer.ggml.model=%q is not a BPE tokenizer", model)
	}
	toks, ok := meta[ggufTokTokens].([]any)
	if !ok || len(toks) == 0 {
		return nil, fmt.Errorf("nlp: GGUF %s missing or empty", ggufTokTokens)
	}
	vocab := make([]string, len(toks))
	for i := range toks {
		s, ok := toks[i].(string)
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF token %d is %T, want string", i, toks[i])
		}
		vocab[i] = s
	}
	// merges are required for BPE; tolerate absence as an atomic (no-merge) vocab
	var merges []string
	if raw, ok := meta["tokenizer.ggml.merges"].([]any); ok {
		merges = make([]string, 0, len(raw))
		for i, m := range raw {
			s, ok := m.(string)
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF merge %d is %T, want string", i, m)
			}
			merges = append(merges, s)
		}
	}
	var opts []BPEOption
	if unk, ok := ggufTokenID(meta[ggufTokUnkID]); ok {
		opts = append(opts, WithBPEUnkID(unk))
	}
	t, err := NewBPE(vocab, merges, opts...)
	if err != nil {
		return nil, err
	}
	// Control / user-defined / unknown tokens become the EncodeSpecial marker set (§B60).
	// Plain Encode is unaffected: it still treats every marker as literal text.
	t.AddSpecialTokens(ggufSpecials(meta, vocab))
	return t, nil
}
