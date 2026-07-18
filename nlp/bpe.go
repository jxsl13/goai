package nlp

import (
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
	specials specialSet     // markers parsed only by EncodeSpecial (§B60)
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
// i.e. ranks[piece[start : parts[i+1].next.start]] — or maxInt when not mergeable.
type bpePart struct{ start, rank int }

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
	if len(piece) <= 1 {
		if len(piece) == 0 {
			return nil
		}
		return []int{t.ranks[string(piece)]}
	}
	// parts[i] = {start offset of token i, rank of merging token i with i+1}; the
	// trailing sentinel {len, maxInt} makes token i span piece[parts[i].start :
	// parts[i+1].start]. Build all adjacent-pair ranks once, tracking the min.
	parts := make([]bpePart, len(piece)+1)
	minRank, minI := math.MaxInt, -1
	for i := 0; i < len(piece)-1; i++ {
		r := math.MaxInt
		if v, ok := t.ranks[string(piece[i:i+2])]; ok {
			r = v
		}
		parts[i] = bpePart{i, r}
		if r < minRank { // leftmost lowest rank (strict < ⇒ first wins), as before
			minRank, minI = r, i
		}
	}
	parts[len(piece)-1] = bpePart{len(piece) - 1, math.MaxInt}
	parts[len(piece)] = bpePart{len(piece), math.MaxInt}

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

	ids := make([]int, len(parts)-1)
	for i := 0; i < len(parts)-1; i++ {
		ids[i] = t.ranks[string(piece[parts[i].start:parts[i+1].start])]
	}
	return ids
}

// Encode turns text into token ids: GPT-2 pre-tokenization → byte-pair merge per
// piece.
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, piece := range gpt2Split(text) {
		ids = append(ids, t.bpeMerge([]byte(piece))...)
	}
	return ids
}

// Decode reconstructs the exact original text (byte-level → round-trip exact).
func (t *Tokenizer) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(t.decoder[id])
	}
	return b.String()
}

// gpt2Split replicates tiktoken's gpt2 pre-tokenizer regex
// ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)
// as a manual scanner (Go's RE2 has no lookahead). It slices the ORIGINAL byte
// string — never []rune, which would corrupt invalid UTF-8 — so decode∘encode is
// byte-exact for any input (§V15). Invalid bytes decode to RuneError and
// classify as "other", becoming byte tokens. Matches tiktoken on valid UTF-8.
func gpt2Split(text string) []string {
	var out []string
	n := len(text)
	dec := func(i int) (rune, int) { return utf8.DecodeRuneInString(text[i:]) }
	i := 0
	for i < n {
		// 1. contractions (ASCII apostrophe)
		if text[i] == '\'' {
			if m := contraction(text, i); m > 0 {
				out = append(out, text[i:i+m])
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
			if !unicode.IsSpace(r) {
				switch {
				case unicode.IsLetter(r):
					for j < n {
						if r, sz = dec(j); !unicode.IsLetter(r) {
							break
						}
						j += sz
					}
				case unicode.IsNumber(r):
					for j < n {
						if r, sz = dec(j); !unicode.IsNumber(r) {
							break
						}
						j += sz
					}
				default:
					for j < n {
						if r, sz = dec(j); unicode.IsSpace(r) || unicode.IsLetter(r) || unicode.IsNumber(r) {
							break
						}
						j += sz
					}
				}
				out = append(out, text[i:j])
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
			if !unicode.IsSpace(r) {
				break
			}
			lastStart = j
			j += sz
			count++
		}
		if j < n && count > 1 {
			j = lastStart // leave the last whitespace rune for the next token
		}
		out = append(out, text[i:j])
		i = j
	}
	return out
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
