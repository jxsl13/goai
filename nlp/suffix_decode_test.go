package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// TestSuffixDecodeGreedyEqualsGenerate: SuffixDecode, like PromptLookupDecode, must be LOSSLESS under
// greedy — the frequency-scored draft only accelerates; model verification keeps the output token-for-token
// identical to plain greedy Generate (the §R89 / arXiv:2411.04975 guarantee).
func TestSuffixDecodeGreedyEqualsGenerate(t *testing.T) {
	model, tokens := loadGPTModel(t)
	base := tokens[:min(3, len(tokens))]
	prompt := append(append([]int(nil), base...), base...) // recurring n-gram → drafter engages
	const n = 12

	want, err := model.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	got, stats, err := nlp.SuffixDecode(model, prompt, n, 3, 10, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("SuffixDecode produced %d tokens, greedy Generate %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d — greedy SuffixDecode must equal greedy Generate", i, got[i], want[i])
		}
	}
	t.Logf("SuffixDecode lossless; proposed=%d accepted=%d", stats.Proposed, stats.Accepted)
}
