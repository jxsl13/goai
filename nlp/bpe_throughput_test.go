package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// t882Corpus builds the fixed ≈1 MB benchmark corpus (1,000,116 bytes). The
// python tiktoken companion (internal/benchcompare/tokenizer_compare.py) rebuilds
// this byte-for-byte so both tokenizers run on the identical input.
func t882Corpus() string {
	const unit = "The quick brown fox jumps over the lazy dog. Tokenization throughput " +
		"matters for LLM serving; a pure-Go BPE competes with tiktoken's Rust across megabytes. "
	var sb strings.Builder
	for sb.Len() < 1_000_000 {
		sb.WriteString(unit)
	}
	return sb.String()
}

// BenchmarkGPT2Encode measures BPE encode throughput (bytes/s). On M2 Pro GoAI's
// pure-Go GPT-2 BPE encodes the 1 MB corpus at ≈28.2 MB/s (≈6.7M tok/s), versus
// tiktoken 0.13.0's Rust core at 18.8 MB/s — GoAI ≈1.50× faster — with the
// identical 237,208-token output on both sides (bit-exact parity, T882). The
// tiktoken figure is end-to-end through its Python binding (it materializes the
// token list a Python caller consumes); GoAI returns a native slice. Measured by
// internal/benchcompare/tokenizer_compare.py; this benchmark guards the Go side.
func BenchmarkGPT2Encode(b *testing.B) {
	tk, err := nlp.LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		b.Skipf("gpt2 ranks unavailable (run make golden): %v", err)
	}
	text := t882Corpus()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.Encode(text)
	}
}

// BenchmarkGPT2Decode measures BPE decode throughput on the same corpus's token
// stream. On M2 Pro GoAI decodes at ≈470 MB/s (≈111M tok/s) versus tiktoken's
// 392 MB/s — GoAI ≈1.20× faster (T882). Decode reconstructs the original bytes
// exactly (§V15 round-trip).
func BenchmarkGPT2Decode(b *testing.B) {
	tk, err := nlp.LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		b.Skipf("gpt2 ranks unavailable (run make golden): %v", err)
	}
	text := t882Corpus()
	ids := tk.Encode(text)
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.Decode(ids)
	}
}
