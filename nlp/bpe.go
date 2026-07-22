package nlp

import (
	"iter"
	"math"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Tokenizer is a byte-level BPE tokenizer compatible with tiktoken's gpt2
// encoding (§T37). The rank table (byte-sequence → id) doubles as the merge
// priority (tiktoken's byte_pair_merge). Since all 256 single bytes are base
// tokens, decode(encode(x)) == x for ANY input — the tokenizer's round-trip
// invariant (§V15), guaranteed by construction. Algorithm: BPE (Sennrich et al.
// 2016, §R33), byte-level variant.
type Tokenizer struct {
	ranks    map[string]int // token bytes → id / merge rank
	decoder  map[int]string // id → token bytes
	decSlice []string       // id → token bytes as a dense slice (Decode fast path; nil until buildPairRank)
	specials specialSet     // markers parsed only by EncodeSpecial (§B60)
	// pairRank[a<<8|b] = rank of the two-byte token {a,b}, or math.MaxInt if not a token. The
	// initial merge pass looks up every adjacent BYTE pair; a direct array index skips the string
	// hash + compare that dominated the profile (mapaccess2_faststr). len 65536 or nil (map fallback).
	pairRank []int
	// byteRank[b] = id of the single-byte token b, or math.MaxInt. Lets the merge track each part's
	// token id (initial = single byte; after a merge = the merge rank) so the final emit reads
	// parts[i].id instead of re-hashing ranks[piece[a:b]] per token (mapaccess1_faststr). len 256 or nil.
	byteRank []int
}

// buildPairRank fills the two-byte AND single-byte fast-lookup tables from ranks (call once after
// ranks is populated). Both are built together so t.pairRank != nil ⇔ t.byteRank != nil.
func (t *Tokenizer) buildPairRank() {
	pr := make([]int, 65536)
	for i := range pr {
		pr[i] = math.MaxInt
	}
	// byteRank is ZERO-initialized (not maxInt): the id it feeds replaces the old
	// mapaccess1 final emit t.ranks[string(1 byte)], which yields 0 on a miss — so a
	// byte with no single-byte token must read 0 here too, keeping the emit bit-identical.
	br := make([]int, 256)
	for tok, id := range t.ranks {
		switch len(tok) {
		case 1:
			br[tok[0]] = id
		case 2:
			pr[int(tok[0])<<8|int(tok[1])] = id
		}
	}
	t.pairRank = pr
	t.byteRank = br

	// Dense id→bytes slice for Decode: the ids are the tiktoken ranks (0..vocab−1), so a
	// slice index replaces the per-token mapaccess1_fast64 that dominated Decode. Gaps and
	// out-of-range ids resolve to "" exactly as the map's zero value did.
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
}

// LoadGPT2 reads a tiktoken-exported rank file (base64(bytes) rank per line).
func LoadGPT2(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readTiktoken(f)
}

// bpePart is one token boundary during the merge: start = its byte offset into
// the contiguous piece; rank = the merge rank of the pair (this token, the next),
// i.e. ranks[piece[start : parts[i+1].next.start]] — or maxInt when not mergeable;
// id = this token's own id = ranks[piece[start : next.start]], tracked as the piece
// grows (initial single byte, then each merge's rank) so the final emit skips a map
// lookup per token. id is only maintained when the fast tables are built (byteRank != nil).
type bpePart struct{ start, rank, id int }

// bpeMerge applies byte-pair merging to a single piece's bytes, returning token
// ids. Repeatedly merges the adjacent pair whose concatenation has the lowest
// rank until none is mergeable; every remaining part is a base/merged token.
//
// This mirrors tiktoken's byte_pair_merge (§R33): instead of a list of copied
// byte-slices re-concatenated on every candidate pair, it keeps only the byte
// OFFSETS into the immutable piece and ranks each pair via ranks[piece[a:c]].
// The Go compiler special-cases m[string(byteSlice)] to look up WITHOUT
// allocating the string, so the whole merge does ZERO map-key allocations — vs
// the old string(parts[i])+string(parts[i+1]), which allocated three strings per
// candidate pair per iteration. Merge order (leftmost lowest rank) and the final
// token boundaries are bit-identical to the old scan, so encode parity holds.
func (t *Tokenizer) bpeMerge(piece []byte) []int {
	ids, _ := t.bpeMergeInto(piece, nil, nil)
	return ids
}

// bpeMergeInto is bpeMerge with caller-owned scratch: it APPENDS the piece's token ids to out and
// returns the grown out plus the parts buffer for reuse on the next piece. Encode threads one out
// (the whole result) and one parts scratch across every piece, so a text of N pieces does O(1)
// heap allocations for the merge instead of O(N) (the parts slice + per-piece ids slice were 40%+60%
// of encode's allocations in the profile). Numerically identical to bpeMerge — same merge order.
func (t *Tokenizer) bpeMergeInto(piece []byte, out []int, parts []bpePart) ([]int, []bpePart) {
	if len(piece) <= 1 {
		if len(piece) == 0 {
			return out, parts
		}
		return append(out, t.ranks[string(piece)]), parts
	}
	// parts[i] = {start offset of token i, rank of merging token i with i+1}; the
	// trailing sentinel {len, maxInt} makes token i span piece[parts[i].start :
	// parts[i+1].start]. Build all adjacent-pair ranks once, tracking the min.
	if cap(parts) >= len(piece)+1 {
		parts = parts[:len(piece)+1]
	} else {
		parts = make([]bpePart, len(piece)+1)
	}
	minRank, minI := math.MaxInt, -1
	for i := 0; i < len(piece)-1; i++ {
		r, id := math.MaxInt, 0
		if t.pairRank != nil {
			r = t.pairRank[int(piece[i])<<8|int(piece[i+1])] // direct index, no string hash
			id = t.byteRank[piece[i]]                        // this token's id (a single byte for now)
		} else {
			if v, ok := t.ranks[string(piece[i:i+2])]; ok {
				r = v
			}
			id = t.ranks[string(piece[i:i+1])] // mapaccess1: 0 on miss, matching the old emit
		}
		parts[i] = bpePart{i, r, id}
		if r < minRank { // leftmost lowest rank (strict < ⇒ first wins), as before
			minRank, minI = r, i
		}
	}
	// last real token: no right neighbor (rank maxInt); its id is its single byte.
	lastID := 0
	if t.byteRank != nil {
		lastID = t.byteRank[piece[len(piece)-1]]
	} else {
		lastID = t.ranks[string(piece[len(piece)-1:])]
	}
	parts[len(piece)-1] = bpePart{len(piece) - 1, math.MaxInt, lastID}
	parts[len(piece)] = bpePart{len(piece), math.MaxInt, 0}

	// getRank(i) = rank of the pair centered on parts[i] AFTER a merge there =
	// ranks[piece[parts[i].start : parts[i+3].start]] (i+3 because the merge that
	// triggers this recompute removes one boundary to parts[i]'s right).
	getRank := func(i int) int {
		if i+3 < len(parts) {
			if v, ok := t.ranks[string(piece[parts[i].start:parts[i+3].start])]; ok {
				return v
			}
		}
		return math.MaxInt
	}

	for minRank != math.MaxInt {
		i := minI
		parts[i].id = minRank // the merged token's own id = ranks[merged bytes] = this merge's rank
		if i > 0 {
			parts[i-1].rank = getRank(i - 1)
		}
		parts[i].rank = getRank(i)
		parts = append(parts[:i+1], parts[i+2:]...) // drop the merged-away boundary
		// rescan for the new leftmost-min pair (exclude the trailing sentinel).
		minRank, minI = math.MaxInt, -1
		for j := 0; j < len(parts)-1; j++ {
			if parts[j].rank < minRank {
				minRank, minI = parts[j].rank, j
			}
		}
	}

	for i := 0; i < len(parts)-1; i++ {
		out = append(out, parts[i].id) // tracked = ranks[piece[parts[i].start:parts[i+1].start]], no re-hash
	}
	return out, parts
}

// Encode turns text into token ids: GPT-2 pre-tokenization → byte-pair merge per
// piece.
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	var parts []bpePart // merge scratch, reused across pieces
	var pb []byte       // piece bytes, reused across pieces (avoids a []byte(piece) alloc per piece)
	for piece := range gpt2SplitSeq(text) {
		pb = append(pb[:0], piece...)
		ids, parts = t.bpeMergeInto(pb, ids, parts)
	}
	return ids
}

// Decode reconstructs the exact original text (byte-level → round-trip exact).
func (t *Tokenizer) Decode(ids []int) string {
	// Pre-size the builder by ESTIMATE: without Grow, WriteString doubles the buffer
	// as it fills, leaving log(n) throw-away buffers per decode (the profile's gcDrain).
	// An exact pre-pass over decoder[id] lengths costs a second O(n) map sweep that
	// outweighs the churn it saves (measured slower), so estimate ~4 bytes/token (GPT-2
	// average) — one or two grows instead of log(n), and no extra lookups. Grow only
	// affects capacity, so the decoded string is byte-identical either way.
	var b strings.Builder
	b.Grow(len(ids) * 4)
	if ds := t.decSlice; ds != nil {
		for _, id := range ids {
			if uint(id) < uint(len(ds)) { // in-range → its bytes; out-of-range → "" (map-miss parity)
				b.WriteString(ds[id])
			}
		}
	} else {
		for _, id := range ids {
			b.WriteString(t.decoder[id])
		}
	}
	return b.String()
}

// gpt2Split replicates tiktoken's gpt2 pre-tokenizer regex
// ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)
// as a manual scanner (Go's RE2 has no lookahead). It slices the ORIGINAL byte
// string — never []rune, which would corrupt invalid UTF-8 — so decode∘encode is
// byte-exact for any input (§V15). Invalid bytes decode to RuneError and
// classify as "other", becoming byte tokens. Matches tiktoken on valid UTF-8.
// gpt2Split materializes the pre-tokens into a slice (the []string form used by BPETokenizer.Encode
// and the tests). Tokenizer.Encode ranges gpt2SplitSeq directly to avoid this slice allocation.
func gpt2Split(text string) []string {
	var out []string
	for p := range gpt2SplitSeq(text) {
		out = append(out, p)
	}
	return out
}

// gpt2SplitSeq yields each GPT-2 pre-token as a SUBSTRING of text without materializing a slice — the
// pieces are the exact same boundaries gpt2Split produces (the pre-tokenizer regex reimplemented as a
// hand scan). Streaming lets Tokenizer.Encode process one pre-token at a time (no []string alloc,
// which the profile showed at 36% of encode's allocations).
// isSpaceR/isLetterR/isNumberR classify a rune, resolving ASCII (the overwhelming
// majority of real text) with a branchless range check and only consulting the
// Unicode tables for a multibyte rune. Each is bit-for-bit equal to the unicode.IsX
// it replaces on ASCII: IsSpace is {\t..\r, ' '}, IsLetter is A–Z/a–z, IsNumber is
// 0–9 there. gpt2SplitSeq's split points — and thus the tokenization — are unchanged.
func isSpaceR(r rune) bool {
	if r < 0x80 {
		return r == ' ' || (r >= '\t' && r <= '\r')
	}
	return unicode.IsSpace(r)
}

func isLetterR(r rune) bool {
	if r < 0x80 {
		c := r | 0x20 // fold ASCII case; digits/punct fold outside a–z
		return c >= 'a' && c <= 'z'
	}
	return unicode.IsLetter(r)
}

func isNumberR(r rune) bool {
	if r < 0x80 {
		return r >= '0' && r <= '9'
	}
	return unicode.IsNumber(r)
}

func gpt2SplitSeq(text string) iter.Seq[string] {
	return func(yield func(string) bool) {
		n := len(text)
		// ASCII fast path: a one-byte rune needs no UTF-8 decode (the decode was
		// ~4% of Encode); multibyte bytes fall through to the real decoder.
		dec := func(i int) (rune, int) {
			if b := text[i]; b < utf8.RuneSelf {
				return rune(b), 1
			}
			return utf8.DecodeRuneInString(text[i:])
		}
		i := 0
		for i < n {
			// 1. contractions (ASCII apostrophe)
			if text[i] == '\'' {
				if m := contraction(text, i); m > 0 {
					if !yield(text[i : i+m]) {
						return
					}
					i += m
					continue
				}
			}
			// 2/3/4: optional literal-space prefix + letters | numbers | others
			j := i
			if text[j] == ' ' {
				j++
			}
			if j < n {
				r, sz := dec(j)
				if !isSpaceR(r) {
					switch {
					case isLetterR(r):
						for j < n {
							if r, sz = dec(j); !isLetterR(r) {
								break
							}
							j += sz
						}
					case isNumberR(r):
						for j < n {
							if r, sz = dec(j); !isNumberR(r) {
								break
							}
							j += sz
						}
					default:
						for j < n {
							if r, sz = dec(j); isSpaceR(r) || isLetterR(r) || isNumberR(r) {
								break
							}
							j += sz
						}
					}
					if !yield(text[i:j]) {
						return
					}
					i = j
					continue
				}
			}
			// 5/6: whitespace run. \s+(?!\S) hands the last whitespace rune to the
			// following token when a non-space follows; a trailing run is kept whole.
			j = i
			lastStart := i
			count := 0
			for j < n {
				r, sz := dec(j)
				if !isSpaceR(r) {
					break
				}
				lastStart = j
				j += sz
				count++
			}
			if j < n && count > 1 {
				j = lastStart // leave the last whitespace rune for the next token
			}
			if !yield(text[i:j]) {
				return
			}
			i = j
		}
	}
}

var contractions = []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

// contraction returns the byte length of a matching contraction at i, or 0.
func contraction(text string, i int) int {
	for _, c := range contractions {
		if strings.HasPrefix(text[i:], c) {
			return len(c)
		}
	}
	return 0
}
