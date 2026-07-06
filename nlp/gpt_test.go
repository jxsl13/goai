package nlp_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type gptGolden struct {
	Config struct {
		Vocab  int     `json:"vocab"`
		Ctx    int     `json:"ctx"`
		Dim    int     `json:"dim"`
		Heads  int     `json:"heads"`
		Layers int     `json:"layers"`
		Eps    float64 `json:"eps"`
	} `json:"config"`
	Tokens []int     `json:"tokens"`
	Logits []float64 `json:"logits"`
	Torch  string    `json:"torch"`
}

func loadGPT(t *testing.T) (*nlp.GPT, gptGolden) {
	t.Helper()
	raw, err := os.ReadFile("testdata/gpt.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g gptGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	ts, _, err := safetensors.LoadFile("testdata/gpt.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.FromSafetensors(nlp.GPTConfig{
		Vocab: g.Config.Vocab, Ctx: g.Config.Ctx, Dim: g.Config.Dim,
		Heads: g.Config.Heads, Layers: g.Config.Layers, Eps: g.Config.Eps,
	}, ts)
	if err != nil {
		t.Fatal(err)
	}
	return model, g
}

// §V1/§G2: the full LLM forward (embed → 2×[LN→causal MHA→res, LN→FFN→res] →
// LN → head) matches real torch f64 logits at rtol 1e-12.
func TestGPTLogitsParity(t *testing.T) {
	model, g := loadGPT(t)
	logits, err := model.Forward(backend.NewContext(), g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	if !logits.Shape().Equal(tensor.Shape{len(g.Tokens), g.Config.Vocab}) {
		t.Fatalf("logits shape %v", logits.Shape())
	}
	for i := range g.Logits {
		idx := tensor.Unravel(i, logits.Shape())
		got := logits.AtF64(idx...)
		if math.Abs(got-g.Logits[i]) > 1e-12*math.Max(1, math.Abs(g.Logits[i])) {
			t.Fatalf("logit[%d]: got %.17g want %.17g (torch %s)", i, got, g.Logits[i], g.Torch)
		}
	}
}

// Causality: changing a future token must not change logits at earlier
// positions — the defining property of decoder self-attention.
func TestGPTCausality(t *testing.T) {
	model, g := loadGPT(t)
	ctx := backend.NewContext()

	base, err := model.Forward(ctx, g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	mutated := append([]int{}, g.Tokens...)
	mutated[len(mutated)-1] = (mutated[len(mutated)-1] + 7) % g.Config.Vocab
	alt, err := model.Forward(ctx, mutated)
	if err != nil {
		t.Fatal(err)
	}
	// positions before the mutated one: bit-identical
	for i := range len(g.Tokens) - 1 {
		for v := range g.Config.Vocab {
			if base.AtF64(i, v) != alt.AtF64(i, v) {
				t.Fatalf("future token leaked into position %d (causality broken)", i)
			}
		}
	}
	// the mutated position itself must differ (sanity that the change matters)
	var diff bool
	last := len(g.Tokens) - 1
	for v := range g.Config.Vocab {
		if base.AtF64(last, v) != alt.AtF64(last, v) {
			diff = true
			break
		}
	}
	if !diff {
		t.Error("mutated token produced identical logits — test not sensitive")
	}
}

func TestGPTErrors(t *testing.T) {
	model, g := loadGPT(t)
	ctx := backend.NewContext()
	if _, err := model.Forward(ctx, nil); err == nil {
		t.Error("empty prompt must error")
	}
	if _, err := model.Forward(ctx, []int{0, 1, 2, 3, 4, 5, 6, 7, 8}); err == nil {
		t.Error("prompt beyond ctx must error")
	}
	if _, err := model.Forward(ctx, []int{g.Config.Vocab + 1}); err == nil {
		t.Error("token beyond vocab must error")
	}
	// missing tensor in the map
	if _, err := nlp.FromSafetensors(nlp.GPTConfig{Layers: 1}, map[string]*tensor.Tensor{}); err == nil {
		t.Error("missing tensors must error")
	}
}
