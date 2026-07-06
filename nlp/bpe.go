package nlp

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"math"
	"os"
	"strconv"
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
	ranks   map[string]int // token bytes → id / merge rank
	decoder map[int]string // id → token bytes
}

// LoadGPT2 reads a tiktoken-exported rank file (base64(bytes) rank per line).
func LoadGPT2(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	t := &Tokenizer{ranks: map[string]int{}, decoder: map[int]string{}}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		sp := strings.IndexByte(line, ' ')
		if sp < 0 {
			continue
		}
		b, err := base64.StdEncoding.DecodeString(line[:sp])
		if err != nil {
			return nil, fmt.Errorf("bpe: bad base64 %q: %w", line[:sp], err)
		}
		id, err := strconv.Atoi(line[sp+1:])
		if err != nil {
			return nil, fmt.Errorf("bpe: bad rank %q: %w", line[sp+1:], err)
		}
		t.ranks[string(b)] = id
		t.decoder[id] = string(b)
	}
	return t, sc.Err()
}

// bpeMerge applies byte-pair merging to a single piece's bytes, returning token
// ids. Repeatedly merges the adjacent pair whose concatenation has the lowest
// rank until none is mergeable; every remaining part is a base/merged token.
func (t *Tokenizer) bpeMerge(piece []byte) []int {
	if len(piece) == 1 {
		return []int{t.ranks[string(piece)]}
	}
	parts := make([][]byte, len(piece))
	for i := range piece {
		parts[i] = piece[i : i+1]
	}
	for len(parts) > 1 {
		minRank, minI := math.MaxInt, -1
		for i := 0; i < len(parts)-1; i++ {
			if r, ok := t.ranks[string(parts[i])+string(parts[i+1])]; ok && r < minRank {
				minRank, minI = r, i
			}
		}
		if minI < 0 {
			break
		}
		merged := append(append([]byte{}, parts[minI]...), parts[minI+1]...)
		parts[minI] = merged
		parts = append(parts[:minI+1], parts[minI+2:]...)
	}
	ids := make([]int, len(parts))
	for i, p := range parts {
		ids[i] = t.ranks[string(p)]
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
