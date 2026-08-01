package gguf

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// writeSynthModel builds a model whose SHAPE matches what a real load pays for, writes it to a
// temp file, and returns the path.
//
// The metadata is the point, not decoration. A llama-family header is dominated by the tokenizer
// arrays — tokens, scores, token types — and every string in them is read as a length followed by
// its bytes. That cost is constant in model size, so it dominates loading a small model and is
// invisible in any benchmark that writes an empty metadata map, which is what every existing
// benchmark here does.
func writeSynthModel(tb testing.TB, nTensors, dim, vocab int) string {
	tb.Helper()
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	types := make([]any, vocab)
	for i := range vocab {
		toks[i] = "tok" + string(rune('a'+i%26)) + string(rune('0'+(i/26)%10))
		scores[i] = float32(i%97) * 0.01
		types[i] = int32(i % 6)
	}
	f := &File{Version: 3, Tensors: make(map[string]*tensor.Tensor, nTensors), Metadata: map[string]any{
		"general.architecture":       "llama",
		"tokenizer.ggml.tokens":      toks,
		"tokenizer.ggml.scores":      scores,
		"tokenizer.ggml.token_type":  types,
		"llama.embedding_length":     uint32(dim),
		"llama.block_count":          uint32(nTensors),
		"llama.attention.head_count": uint32(8),
	}}
	quant := make(map[string]QuantType, nTensors)
	for t := range nTensors {
		name := "blk." + string(rune('a'+t%26)) + string(rune('0'+t/26)) + ".weight"
		w := tensor.New(tensor.F32, tensor.Shape{dim, dim})
		s := w.Storage().F32()
		for i := range s {
			s[i] = float32((i*7+t)%101-50) * 0.02
		}
		f.Tensors[name] = w
		quant[name] = Q4_K
	}
	path := filepath.Join(tb.TempDir(), "synth.gguf")
	fh, err := os.Create(path)
	if err != nil {
		tb.Fatal(err)
	}
	if err := WriteQuantized(fh, f, quant); err != nil {
		fh.Close()
		tb.Fatal(err)
	}
	if err := fh.Close(); err != nil {
		tb.Fatal(err)
	}
	return path
}

// BenchmarkReadFileSynth measures the whole load path — open, header parse, tensor decode — on a
// model this test builds itself.
//
// It exists because BenchmarkReadFileModel SKIPS: it wants a checked-in tinyllama gguf that the
// repository does not carry, so the entire load path had no instrumentation at all and every
// candidate against it was unmeasurable. This one needs nothing but a temp directory.
//
// Two shapes, because the load has two independently sized halves. The header cost scales with the
// vocabulary and is constant in model size; the decode cost scales with the tensor bytes. A single
// shape would let a change to one half hide behind the other.
func BenchmarkReadFileSynth(b *testing.B) {
	for _, c := range []struct {
		name                 string
		nTensors, dim, vocab int
	}{
		{"header-heavy", 8, 256, 32000},
		{"tensor-heavy", 64, 512, 512},
	} {
		b.Run(c.name, func(b *testing.B) {
			path := writeSynthModel(b, c.nTensors, c.dim, c.vocab)
			if _, err := ReadFile(path); err != nil { // warm the page cache, validate
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := ReadFile(path); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestParseTruncatedDataSection pins the error path the presized read introduced. The data section
// is now allocated from the tensor table and filled with io.ReadFull, so a file whose table
// promises more bytes than the file holds fails HERE, naming the shortfall, instead of being
// carried forward as a short section and reported later as a per-tensor range error.
//
// Both halves matter: the parse must fail, and it must not panic or allocate on the promise of a
// table it cannot satisfy.
func TestParseTruncatedDataSection(t *testing.T) {
	path := writeSynthModel(t, 4, 128, 64)
	full, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Read(bytes.NewReader(full)); err != nil {
		t.Fatalf("the untruncated model must parse: %v", err)
	}
	// Drop the last quarter: enough to fall inside the data section, not the header.
	cut := len(full) - len(full)/4
	if _, err := Read(bytes.NewReader(full[:cut])); err == nil {
		t.Fatal("a truncated data section parsed without error")
	}
}
