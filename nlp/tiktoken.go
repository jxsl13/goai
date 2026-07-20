package nlp

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// tiktoken rank-file interop for the byte-level BPE Tokenizer (§T37). The format (used by OpenAI's
// gpt2/cl100k_base/o200k_base encodings) is one "base64(token-bytes) rank" pair per line; the rank
// doubles as the merge priority. Definitional source: the tiktoken file layout / reference — verified
// by round-trip + fuzz (§V15), §V16-exempt like the GGUF and tokenizer.json loaders. LoadGPT2 reads
// it from a path; these add byte-slice parsing and serialization so it round-trips in memory.

// readTiktoken parses a tiktoken rank stream into a Tokenizer. Blank lines and lines without a
// space are skipped; a malformed base64 token or non-integer rank is an error.
func readTiktoken(r io.Reader) (*Tokenizer, error) {
	t := &Tokenizer{ranks: map[string]int{}, decoder: map[int]string{}}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		sp := strings.IndexByte(line, ' ')
		if sp < 0 {
			continue
		}
		b, err := base64.StdEncoding.DecodeString(line[:sp])
		if err != nil {
			return nil, fmt.Errorf("nlp: tiktoken bad base64 %q: %w", line[:sp], err)
		}
		id, err := strconv.Atoi(line[sp+1:])
		if err != nil {
			return nil, fmt.Errorf("nlp: tiktoken bad rank %q: %w", line[sp+1:], err)
		}
		t.ranks[string(b)] = id
		t.decoder[id] = string(b)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	t.buildPairRank()
	return t, nil
}

// TiktokenFromBytes parses an in-memory tiktoken rank file into a byte-level BPE Tokenizer — the
// byte-slice counterpart of LoadGPT2.
func TiktokenFromBytes(data []byte) (*Tokenizer, error) {
	return readTiktoken(bytes.NewReader(data))
}

// ToTiktoken serializes the tokenizer back to the tiktoken rank-file format (base64(bytes) rank per
// line, ascending by rank), so TiktokenFromBytes(t.ToTiktoken()) round-trips it.
func (t *Tokenizer) ToTiktoken() []byte {
	type entry struct {
		bytes string
		id    int
	}
	es := make([]entry, 0, len(t.decoder))
	for id, b := range t.decoder {
		es = append(es, entry{b, id})
	}
	sort.Slice(es, func(a, b int) bool { return es[a].id < es[b].id })
	var buf bytes.Buffer
	for _, e := range es {
		buf.WriteString(base64.StdEncoding.EncodeToString([]byte(e.bytes)))
		buf.WriteByte(' ')
		buf.WriteString(strconv.Itoa(e.id))
		buf.WriteByte('\n')
	}
	return buf.Bytes()
}
