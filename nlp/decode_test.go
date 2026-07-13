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

func loadGPTModel(t *testing.T) (*nlp.GPT, []int) {
	t.Helper()
	raw, err := os.ReadFile("testdata/gpt.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Config struct {
			Vocab, Ctx, Dim, Heads, Layers int
			Eps                            float64
		} `json:"config"`
		Tokens []int `json:"tokens"`
	}
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
	return model, g.Tokens
}

// §T35: incremental KV-cache decoding must reproduce the full-forward logits at
// every position (bit-close: same causal attention math, f64).
func TestKVCacheParity(t *testing.T) {
	model, tokens := loadGPTModel(t)
	ctx := backend.NewContext()

	full, err := model.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}

	cache := model.NewCache()
	for pos, tok := range tokens {
		step, err := model.DecodeStep(ctx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		if cache.Len() != pos+1 {
			t.Fatalf("cache len %d after step %d", cache.Len(), pos)
		}
		vocab := full.Shape()[1]
		for v := range vocab {
			got := step.AtF64(0, v)
			want := full.AtF64(pos, v)
			if math.Abs(got-want) > 1e-11*math.Max(1, math.Abs(want)) {
				t.Fatalf("pos %d cls %d: incremental %.14g vs full %.14g", pos, v, got, want)
			}
		}
	}
	t.Logf("KV-cache decode matches full forward across %d positions", len(tokens))
}

// The generalized MHA (sq<sk) still leaves the full-attention path unchanged:
// a single query against N keys attends to all of them.
func TestMHASingleQueryAttendsAll(t *testing.T) {
	ctx := backend.NewContext()
	// 1 query, 3 keys, 1 head, dmodel 2, non-causal → weights sum to a valid avg
	q := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{0, 0}) // zero query → uniform softmax
	k := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 0, 0, 1, -1, 0})
	v := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{3, 0, 0, 6, 9, 0})
	out, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: 1, Causal: false})
	if err != nil {
		t.Fatal(err)
	}
	// zero query → equal weights 1/3 → out = mean of v = [(3+0+9)/3,(0+6+0)/3]=[4,2]
	if math.Abs(out[0].AtF64(0, 0)-4) > 1e-12 || math.Abs(out[0].AtF64(0, 1)-2) > 1e-12 {
		t.Errorf("single-query attention = (%v,%v), want (4,2)", out[0].AtF64(0, 0), out[0].AtF64(0, 1))
	}
}

// §T361/§V: Generate's WithBackend option routes the decode loop to the chosen backend
// and (greedy = deterministic) produces identical tokens across backends — verifying the
// option is plumbed and the routing is correct. Decode is dispatch-bound, so callers use
// this to put small-model decode on the faster cpu (§T360/§T361).
func TestGenerateWithBackend(t *testing.T) {
	model, prompt := loadGPTModel(t)
	cpuB, _ := backend.Get(backend.CPU)
	refB, _ := backend.Get(backend.Ref)

	base, err := model.Generate(prompt, 4, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	onCPU, err := model.Generate(prompt, 4, nlp.Greedy(), nlp.WithBackend(cpuB))
	if err != nil {
		t.Fatalf("Generate WithBackend(cpu): %v", err)
	}
	onRef, err := model.Generate(prompt, 4, nlp.Greedy(), nlp.WithBackend(refB))
	if err != nil {
		t.Fatalf("Generate WithBackend(ref): %v", err)
	}
	if len(onCPU) != len(base) || len(onRef) != len(base) {
		t.Fatalf("length mismatch: base %d, cpu %d, ref %d", len(base), len(onCPU), len(onRef))
	}
	for i := range base {
		if onCPU[i] != base[i] || onRef[i] != base[i] {
			t.Fatalf("token %d differs across backends: base %d, cpu %d, ref %d", i, base[i], onCPU[i], onRef[i])
		}
	}
}
