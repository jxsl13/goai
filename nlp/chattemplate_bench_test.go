package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// BenchmarkChatRender measures rendering a realistic multi-turn conversation through the
// llama3 chat template (the shared turn-style renderer). Pre-sizing the strings.Builder
// with estChatSize replaces the log(n) doubling reallocations with a single allocation.
func BenchmarkChatRender(b *testing.B) {
	tpl, err := nlp.NewChatTemplate("llama3")
	if err != nil {
		b.Fatal(err)
	}
	msgs := []nlp.ChatMessage{{Role: "system", Content: strings.Repeat("You are a helpful assistant. ", 8)}}
	for i := 0; i < 10; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs = append(msgs, nlp.ChatMessage{Role: role, Content: strings.Repeat("Some conversational content here. ", 6+i)})
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := tpl.Render(msgs, nlp.WithGenerationPrompt()); err != nil {
			b.Fatal(err)
		}
	}
}
