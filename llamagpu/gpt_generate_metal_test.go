//go:build darwin && cgo

package llamagpu

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func gptGenerateParityModel(t testing.TB, cfg nlp.GPTConfig) *nlp.GPT {
	t.Helper()
	seed := 0
	values := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			seed++
			d[i] = float32((seed*7)%23-11) * 0.02
		}
		return x
	}
	gain := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 0.9 + float32(i%5)*0.02
		}
		return x
	}
	bias := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%3-1) * 0.05
		}
		return x
	}
	ffn := 4 * cfg.Dim
	ts := map[string]*tensor.Tensor{
		"tok_emb": values(cfg.Vocab, cfg.Dim), "pos_emb": values(cfg.Ctx, cfg.Dim),
		"head": values(cfg.Dim, cfg.Vocab), "lnf.gamma": gain(cfg.Dim), "lnf.beta": bias(cfg.Dim),
	}
	for layer := range cfg.Layers {
		prefix := fmt.Sprintf("blocks.%d.", layer)
		ts[prefix+"ln1.gamma"] = gain(cfg.Dim)
		ts[prefix+"ln1.beta"] = bias(cfg.Dim)
		ts[prefix+"attn.wq"] = values(cfg.Dim, cfg.Dim)
		ts[prefix+"attn.wk"] = values(cfg.Dim, cfg.Dim)
		ts[prefix+"attn.wv"] = values(cfg.Dim, cfg.Dim)
		ts[prefix+"attn.wo"] = values(cfg.Dim, cfg.Dim)
		ts[prefix+"ln2.gamma"] = gain(cfg.Dim)
		ts[prefix+"ln2.beta"] = bias(cfg.Dim)
		ts[prefix+"ffn.w1"] = values(cfg.Dim, ffn)
		ts[prefix+"ffn.b1"] = bias(ffn)
		ts[prefix+"ffn.w2"] = values(ffn, cfg.Dim)
		ts[prefix+"ffn.b2"] = bias(cfg.Dim)
	}
	m, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestGPTGenerateLogitsReusePreservesTokensAndCache(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, Layers: 2, Eps: 1e-5}
	m := gptGenerateParityModel(t, cfg)
	candidate, err := newGPTDecoder(m, metalGPTOps())
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Release()
	control, err := newGPTDecoder(m, metalGPTOps())
	if err != nil {
		t.Fatal(err)
	}
	defer control.Release()
	control.eagerGenerateResultControl = true

	prompt := []int{5, 12, 3}
	const maxNew = 8
	got, err := candidate.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	want, err := control.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("token count = %d, historical control %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, historical control %d", i, got[i], want[i])
		}
	}

	probe := (got[len(got)-1] + 7) % cfg.Vocab
	gotLogits, err := candidate.Step(probe, len(got))
	if err != nil {
		t.Fatal(err)
	}
	wantLogits, err := control.Step(probe, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i := range gotLogits {
		if math.Float32bits(gotLogits[i]) != math.Float32bits(wantLogits[i]) {
			t.Fatalf("continuation logit[%d] = %v, historical control %v", i, gotLogits[i], wantLogits[i])
		}
	}
}
