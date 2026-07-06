package nlp_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type gptTrainGolden struct {
	Config struct {
		Vocab, Ctx, Dim, Heads, Layers int
		Eps                            float64
	} `json:"config"`
	Tokens  []int                `json:"tokens"`
	Targets []float64            `json:"targets"`
	Loss    float64              `json:"loss"`
	Grads   map[string][]float64 `json:"grads"`
}

// paramNames mirrors nlp.GPT.Params() ordering ↔ the golden grad keys.
func paramNames(layers int) []string {
	names := []string{"tok_emb", "pos_emb"}
	for l := range layers {
		p := fmt.Sprintf("blocks.%d.", l)
		names = append(names,
			p+"ln1.gamma", p+"ln1.beta",
			p+"attn.wq", p+"attn.wk", p+"attn.wv", p+"attn.wo",
			p+"ln2.gamma", p+"ln2.beta",
			p+"ffn.w1", p+"ffn.b1", p+"ffn.w2", p+"ffn.b2",
		)
	}
	return append(names, "lnf.gamma", "lnf.beta", "head")
}

// §T34/§G5: the FULL GPT backward — embedding, LayerNorm, fused attention, FFN,
// residuals, head — matches real torch gradients for every one of the 29 weights
// at f64 rtol 1e-9. This validates the entire training path against the oracle.
func TestGPTBackwardVsTorch(t *testing.T) {
	raw, err := os.ReadFile("testdata/gpt.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g gptTrainGolden
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

	tape := autograd.NewTape()
	ctx := tape.Context()
	logits, err := model.Forward(ctx, g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	targets := tensor.FromFloat64(tensor.Shape{len(g.Targets)}, g.Targets)
	loss, err := nn.CrossEntropy(ctx, logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(loss.AtF64()-g.Loss) > 1e-12*math.Max(1, math.Abs(g.Loss)) {
		t.Fatalf("loss %.12g vs torch %.12g", loss.AtF64(), g.Loss)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}

	names := paramNames(g.Config.Layers)
	params := model.Params()
	if len(names) != len(params) {
		t.Fatalf("param count %d != names %d", len(params), len(names))
	}
	for k, p := range params {
		want := g.Grads[names[k]]
		grad := tape.Grad(p)
		if grad == nil {
			t.Fatalf("%s: nil gradient (path not differentiable)", names[k])
		}
		if grad.Numel() != len(want) {
			t.Fatalf("%s: grad numel %d != golden %d", names[k], grad.Numel(), len(want))
		}
		for i := range want {
			got := grad.AtF64(tensor.Unravel(i, grad.Shape())...)
			if math.Abs(got-want[i]) > 1e-9*math.Max(1, math.Abs(want[i])) {
				t.Fatalf("%s grad[%d]: got %.12g want %.12g (torch)", names[k], i, got, want[i])
			}
		}
	}
	t.Logf("full GPT backward matches torch for all %d weights", len(params))
}

// §G5: the GPT trains — CrossEntropy loss on the fixed targets drops sharply
// under AdamW + grad clipping, proving the end-to-end loop learns.
func TestGPTTrains(t *testing.T) {
	raw, _ := os.ReadFile("testdata/gpt.json")
	var g gptTrainGolden
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
	targets := tensor.FromFloat64(tensor.Shape{len(g.Targets)}, g.Targets)
	opt := nn.NewAdamW(model.Params(), 0.01, 0.01)

	var first, last float64
	for step := range 100 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := model.Forward(ctx, g.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(model.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first*0.2 {
		t.Fatalf("GPT did not train: loss %.4f → %.4f", first, last)
	}
	t.Logf("GPT trained: CE %.4f → %.4f", first, last)
}
