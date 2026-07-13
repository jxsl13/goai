package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func randMat(shape tensor.Shape, seed int) *tensor.Tensor {
	t := tensor.New(tensor.F64, shape)
	for i := range t.Numel() {
		idx := tensor.Unravel(i, shape)
		t.SetF64(math.Sin(float64(i*13+seed)*0.37+0.2)*0.5, idx...)
	}
	return t
}

// §V16 tier-1 (§T119): heads-aware RoPE rotates each head's slice independently — it
// equals applying the single-head RoPE to each [seq,headDim] slice on its own.
func TestRoPEMultiHeadEqualsPerHead(t *testing.T) {
	const seq, heads, dk = 5, 3, 4
	q := randMat(tensor.Shape{seq, heads * dk}, 1)
	ctx := backend.NewContext()

	full, err := backend.Execute(ctx, backend.OpRoPE, []*tensor.Tensor{q}, backend.RoPEAttrs{Base: 10000, Heads: heads})
	if err != nil {
		t.Fatal(err)
	}
	for h := 0; h < heads; h++ {
		slice := tensor.New(tensor.F64, tensor.Shape{seq, dk})
		for p := range seq {
			for d := range dk {
				slice.SetF64(q.AtF64(p, h*dk+d), p, d)
			}
		}
		ref, err := backend.Execute(ctx, backend.OpRoPE, []*tensor.Tensor{slice}, backend.RoPEAttrs{Base: 10000})
		if err != nil {
			t.Fatal(err)
		}
		for p := range seq {
			for d := range dk {
				got, want := full[0].AtF64(p, h*dk+d), ref[0].AtF64(p, d)
				if math.Abs(got-want) > 1e-12 {
					t.Fatalf("head %d [%d,%d]: multi-head %.12g vs per-head %.12g", h, p, d, got, want)
				}
			}
		}
	}
}

func tinyLlama(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 6, Ctx: 8, Dim: 8, Heads: 2, KVHeads: 1, Layers: 2, Hidden: 16, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// The assembled Llama produces [seq,vocab] logits and is deterministic.
func TestLlamaForwardShapeDeterministic(t *testing.T) {
	m := tinyLlama(t)
	tokens := []int{1, 3, 2, 5}
	a, err := m.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	if !a.Shape().Equal(tensor.Shape{len(tokens), 6}) {
		t.Fatalf("logits shape %v, want [%d,6]", a.Shape(), len(tokens))
	}
	b, _ := m.Forward(backend.NewContext(), tokens)
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatal("Llama forward is not deterministic")
		}
	}
}

// §V2 / §V16 tier-1: the whole Llama forward is differentiable end-to-end — central
// finite differences match the analytic gradients the tape produces, proving RMSNorm,
// the RoPE-on-Q/K attention (with GQA), SwiGLU and the residuals are all wired and
// backpropagated correctly.
func TestLlamaGradCheck(t *testing.T) {
	m := tinyLlama(t)
	tokens := []int{1, 4, 2}

	loss := func() float64 {
		out, err := m.Forward(backend.NewContext(), tokens)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out.Numel() {
			s += out.AtF64(tensor.Unravel(i, out.Shape())...)
		}
		return s
	}

	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out); err != nil {
		t.Fatal(err)
	}

	const h = 1e-6
	// spot-check a few parameters across the different sublayers
	b0 := m.Blocks[0]
	targets := []*tensor.Tensor{m.TokEmb, b0.Wq, b0.Wv, b0.AttnNorm.Gamma, b0.FFN.Wgate, m.Norm.Gamma, m.Out}
	for _, p := range targets {
		grad := tape.Grad(p)
		if grad == nil {
			t.Fatalf("param %v received no gradient", p.Shape())
		}
		n := p.Numel()
		for _, flat := range []int{0, n / 2, n - 1} {
			idx := tensor.Unravel(flat, p.Shape())
			o := p.AtF64(idx...)
			p.SetF64(o+h, idx...)
			lp := loss()
			p.SetF64(o-h, idx...)
			lm := loss()
			p.SetF64(o, idx...)
			num, ana := (lp-lm)/(2*h), grad.AtF64(idx...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("param %v[%d]: numeric %.8g vs analytic %.8g", p.Shape(), flat, num, ana)
			}
		}
	}
}

// §V16 tier-1: grouped-query attention is exercised — a Llama with KVHeads < Heads
// builds Wk/Wv of the reduced width and runs end-to-end.
func TestLlamaGQARuns(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 5, Ctx: 8, Dim: 12, Heads: 6, KVHeads: 2, Layers: 1, Hidden: 24, Eps: 1e-5,
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	// Wk maps dim → kvHeads·headDim = 2·2 = 4
	if got := m.Blocks[0].Wk.Shape(); !got.Equal(tensor.Shape{12, 4}) {
		t.Fatalf("GQA Wk shape %v, want [12,4]", got)
	}
	if _, err := m.Forward(backend.NewContext(), []int{0, 1, 2, 3}); err != nil {
		t.Fatalf("GQA Llama forward: %v", err)
	}
}

// A Llama model assembled from RMSNorm, RoPE attention, GQA and SwiGLU turns a prompt
// into next-token logits.
func ExampleLlama() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 32, Ctx: 16, Dim: 16, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5,
	}, 1)
	logits, _ := m.Forward(backend.NewContext(), []int{5, 9, 14})
	fmt.Println(logits.Shape())
	// Output: (3, 32)
}
