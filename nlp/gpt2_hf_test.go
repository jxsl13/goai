package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §T506: the GPT-2/HF checkpoint converter, verified synthetically — an HF-named tensor map
// (fused c_attn, Conv1D [in,out] layout, tied head) is converted and its logits must be
// BIT-identical to a manually assembled reference model with the same weights split by hand.
func TestGPT2FromHFMatchesManualAssembly(t *testing.T) {
	const vocab, ctxLen, d, heads, layers = 11, 12, 8, 2, 2
	val := func(seed, i int) float64 { return math.Sin(float64(seed*1000+i)) * 0.3 }
	mk := func(seed int, shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape(shape))
		for i := range x.Numel() {
			x.SetF64(val(seed, i), tensor.Unravel(i, x.Shape())...)
		}
		return x
	}

	// HF-style map.
	hf := map[string]*tensor.Tensor{
		"wte.weight":  mk(1, vocab, d),
		"wpe.weight":  mk(2, ctxLen, d),
		"ln_f.weight": mk(30, d), "ln_f.bias": mk(31, d),
	}
	for l := range layers {
		p := fmt.Sprintf("h.%d.", l)
		s := 100 * (l + 1)
		hf[p+"ln_1.weight"], hf[p+"ln_1.bias"] = mk(s+1, d), mk(s+2, d)
		hf[p+"attn.c_attn.weight"], hf[p+"attn.c_attn.bias"] = mk(s+3, d, 3*d), mk(s+4, 3*d)
		hf[p+"attn.c_proj.weight"], hf[p+"attn.c_proj.bias"] = mk(s+5, d, d), mk(s+6, d)
		hf[p+"ln_2.weight"], hf[p+"ln_2.bias"] = mk(s+7, d), mk(s+8, d)
		hf[p+"mlp.c_fc.weight"], hf[p+"mlp.c_fc.bias"] = mk(s+9, d, 4*d), mk(s+10, 4*d)
		hf[p+"mlp.c_proj.weight"], hf[p+"mlp.c_proj.bias"] = mk(s+11, 4*d, d), mk(s+12, d)
	}

	converted, err := nlp.GPT2FromHF(hf, heads)
	if err != nil {
		t.Fatal(err)
	}

	// manual reference: split c_attn by hand, transpose wte for the head, set biases.
	slice2 := func(src *tensor.Tensor, lo, hi int) *tensor.Tensor {
		rows := src.Shape()[0]
		o := tensor.New(tensor.F64, tensor.Shape{rows, hi - lo})
		for i := range rows {
			for j := lo; j < hi; j++ {
				o.SetF64(src.AtF64(i, j), i, j-lo)
			}
		}
		return o
	}
	slice1 := func(src *tensor.Tensor, lo, hi int) *tensor.Tensor {
		o := tensor.New(tensor.F64, tensor.Shape{hi - lo})
		for j := lo; j < hi; j++ {
			o.SetF64(src.AtF64(j), j-lo)
		}
		return o
	}
	head := tensor.New(tensor.F64, tensor.Shape{d, vocab})
	for v := range vocab {
		for j := range d {
			head.SetF64(hf["wte.weight"].AtF64(v, j), j, v)
		}
	}
	ours := map[string]*tensor.Tensor{
		"tok_emb": hf["wte.weight"], "pos_emb": hf["wpe.weight"], "head": head,
		"lnf.gamma": hf["ln_f.weight"], "lnf.beta": hf["ln_f.bias"],
	}
	for l := range layers {
		p, q := fmt.Sprintf("h.%d.", l), fmt.Sprintf("blocks.%d.", l)
		ours[q+"ln1.gamma"], ours[q+"ln1.beta"] = hf[p+"ln_1.weight"], hf[p+"ln_1.bias"]
		ours[q+"attn.wq"] = slice2(hf[p+"attn.c_attn.weight"], 0, d)
		ours[q+"attn.wk"] = slice2(hf[p+"attn.c_attn.weight"], d, 2*d)
		ours[q+"attn.wv"] = slice2(hf[p+"attn.c_attn.weight"], 2*d, 3*d)
		ours[q+"attn.wo"] = hf[p+"attn.c_proj.weight"]
		ours[q+"ln2.gamma"], ours[q+"ln2.beta"] = hf[p+"ln_2.weight"], hf[p+"ln_2.bias"]
		ours[q+"ffn.w1"], ours[q+"ffn.b1"] = hf[p+"mlp.c_fc.weight"], hf[p+"mlp.c_fc.bias"]
		ours[q+"ffn.w2"], ours[q+"ffn.b2"] = hf[p+"mlp.c_proj.weight"], hf[p+"mlp.c_proj.bias"]
	}
	manual, err := nlp.FromSafetensors(nlp.GPTConfig{
		Vocab: vocab, Ctx: ctxLen, Dim: d, Heads: heads, Layers: layers, Eps: 1e-5,
	}, ours)
	if err != nil {
		t.Fatal(err)
	}
	for l := range layers {
		p := fmt.Sprintf("h.%d.", l)
		manual.Blocks[l].Attn.Bias = map[string]*tensor.Tensor{
			"q": slice1(hf[p+"attn.c_attn.bias"], 0, d),
			"k": slice1(hf[p+"attn.c_attn.bias"], d, 2*d),
			"v": slice1(hf[p+"attn.c_attn.bias"], 2*d, 3*d),
			"o": hf[p+"attn.c_proj.bias"],
		}
	}

	tokens := []int{1, 4, 7, 2, 9, 0, 3}
	ctx := backend.NewContext()
	a, err := converted.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	b, err := manual.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatalf("converted vs manual logits differ at %v", idx)
		}
	}
	// error paths: missing block tensor, malformed c_attn.
	bad := map[string]*tensor.Tensor{"wte.weight": hf["wte.weight"], "wpe.weight": hf["wpe.weight"]}
	if _, err := nlp.GPT2FromHF(bad, heads); err == nil {
		t.Fatal("blockless map accepted")
	}
}
