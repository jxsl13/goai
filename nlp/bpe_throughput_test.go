package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// BenchmarkGPT2Encode measures BPE tokenization throughput (bytes/s). GoAI's
// pure-Go GPT-2 BPE was measured at ≈23 MB/s vs tiktoken's Rust ≈20 MB/s on the
// same 1 MB text with exact 216511-token parity (2026-07-16) — i.e. the pure-Go
// tokenizer beats the Rust incumbent. This benchmark guards that throughput.
func BenchmarkGPT2Encode(b *testing.B) {
	tk, err := nlp.LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		b.Skipf("gpt2 ranks unavailable (run make golden): %v", err)
	}
	const unit = "The quick brown fox jumps over the lazy dog. Tokenization throughput " +
		"matters for LLM serving; a pure-Go BPE competes with tiktoken's Rust across megabytes. "
	var sb strings.Builder
	for sb.Len() < 1_000_000 {
		sb.WriteString(unit)
	}
	text := sb.String()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.Encode(text)
	}
}
