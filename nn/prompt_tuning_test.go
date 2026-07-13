package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: Forward prepends the k-row soft prompt to the seq-row input, giving
// [k+seq, d] — the first k rows are exactly P, the rest are the input (§R127).
func TestPromptTuningForward(t *testing.T) {
	const k, d, seq = 2, 3, 4
	pt, err := nn.NewPromptTuning(tensor.F64, k, d, 5)
	if err != nil {
		t.Fatal(err)
	}
	ed := make([]float64, seq*d)
	for i := range ed {
		ed[i] = float64(i) + 0.5
	}
	embeds := tensor.FromFloat64(tensor.Shape{seq, d}, ed)
	out, err := pt.Forward(backend.NewContext(), embeds)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{k + seq, d}) {
		t.Fatalf("shape %v, want [%d %d]", out.Shape(), k+seq, d)
	}
	for r := range k { // first k rows == P
		for c := range d {
			if out.AtF64(r, c) != pt.P.AtF64(r, c) {
				t.Errorf("prompt row %d col %d: %g != P %g", r, c, out.AtF64(r, c), pt.P.AtF64(r, c))
			}
		}
	}
	for r := range seq { // remaining rows == input embeddings
		for c := range d {
			if out.AtF64(k+r, c) != embeds.AtF64(r, c) {
				t.Errorf("input row %d col %d: %g != embeds %g", r, c, out.AtF64(k+r, c), embeds.AtF64(r, c))
			}
		}
	}
	if pt.NumVirtualTokens() != k {
		t.Errorf("NumVirtualTokens %d, want %d", pt.NumVirtualTokens(), k)
	}
	if p := pt.Params(); len(p) != 1 || p[0] != pt.P {
		t.Errorf("Params should be exactly [P]")
	}
}

// §V2: finite differences of ½·Σ(prompted)² w.r.t. the soft prompt P match the
// analytic gradient (the gradient flows into P through the concat, so P trains).
func TestPromptTuningGradcheck(t *testing.T) {
	const k, d, seq = 2, 3, 3
	embeds := tensor.FromFloat64(tensor.Shape{seq, d}, []float64{0.5, -1, 2, 0.3, -0.7, 1.1, -0.4, 0.9, 0.2})
	loss := func(ctx *backend.Context, p *tensor.Tensor) (*tensor.Tensor, error) {
		// Forward is Concat(P, embeds, axis 0); call it directly so P is the tape leaf.
		out, err := backend.Execute(ctx, backend.OpConcat, []*tensor.Tensor{p, embeds}, backend.ConcatAttrs{Axis: 0})
		if err != nil {
			return nil, err
		}
		sq, err := mustOp(ctx, backend.OpMul, out[0], out[0])
		if err != nil {
			return nil, err
		}
		return mustOp(ctx, backend.OpSum, sq)
	}
	pd := []float64{0.2, -0.6, 1.3, 0.8, -0.1, 0.4}
	lossAt := func(v []float64) float64 {
		l, err := loss(backend.NewContext(), tensor.FromFloat64(tensor.Shape{k, d}, v))
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64() / 2
	}
	pT := tensor.FromFloat64(tensor.Shape{k, d}, append([]float64(nil), pd...))
	tape := autograd.NewTape()
	l, err := loss(tape.Context(), pT)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(pT)
	data := append([]float64(nil), pd...)
	const h = 1e-6
	for i := range data {
		o := data[i]
		data[i] = o + h
		lp := lossAt(data)
		data[i] = o - h
		lm := lossAt(data)
		data[i] = o
		num := (lp - lm) / (2 * h)
		ana := 0.5 * g.AtF64(tensor.Unravel(i, pT.Shape())...)
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
}

// §V16 tier-1 (behavioral): training only the soft prompt drives the prompted
// output toward a target — the first k output rows (which ARE P) converge to the
// target prompt while the frozen input rows are untouched.
func TestPromptTuningTrains(t *testing.T) {
	const k, d, seq = 2, 2, 2
	pt, _ := nn.NewPromptTuning(tensor.F64, k, d, 1)
	embeds := tensor.FromFloat64(tensor.Shape{seq, d}, []float64{1, 2, 3, 4})
	targetP := []float64{0.5, -0.5, 1.5, -1.5}
	target := tensor.FromFloat64(tensor.Shape{k + seq, d}, []float64{0.5, -0.5, 1.5, -1.5, 1, 2, 3, 4})

	const lr = 0.2
	first, last := 0.0, 0.0
	for step := range 300 {
		tape := autograd.NewTape()
		out, err := pt.Forward(tape.Context(), embeds)
		if err != nil {
			t.Fatal(err)
		}
		diff, _ := mustOp(tape.Context(), backend.OpSub, out, target)
		sq, _ := mustOp(tape.Context(), backend.OpMul, diff, diff)
		l, _ := mustOp(tape.Context(), backend.OpSum, sq)
		if err := tape.Backward(l); err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = l.AtF64()
		}
		last = l.AtF64()
		g := tape.Grad(pt.P)
		for i := range k * d {
			idx := tensor.Unravel(i, pt.P.Shape())
			pt.P.SetF64(pt.P.AtF64(idx...)-lr*g.AtF64(idx...), idx...)
		}
	}
	if last > 1e-6 || last >= first {
		t.Errorf("prompt did not converge: first=%.6g last=%.6g", first, last)
	}
	for i, w := range targetP {
		idx := tensor.Unravel(i, pt.P.Shape())
		if math.Abs(pt.P.AtF64(idx...)-w) > 1e-3 {
			t.Errorf("P[%d]=%.5f, want %.1f", i, pt.P.AtF64(idx...), w)
		}
	}
}

func TestPromptTuningErrors(t *testing.T) {
	if _, err := nn.NewPromptTuning(tensor.F64, 0, 4, 1); err == nil {
		t.Error("non-positive k must error")
	}
	pt, _ := nn.NewPromptTuning(tensor.F64, 2, 3, 1)
	if _, err := pt.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4, 5})); err == nil {
		t.Error("d_model mismatch must error")
	}
	if _, err := pt.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4, 3, 1})); err == nil {
		t.Error("non rank-2 must error")
	}
}

// Prompt tuning (Lester et al. 2021) prepends a learned soft prompt of k virtual
// tokens to the input embeddings; only the prompt trains. Here a 3-token prompt is
// prepended to a length-5, 8-dim input, giving an 8-row sequence.
func ExamplePromptTuning() {
	pt, _ := nn.NewPromptTuning(tensor.F64, 3, 8, 42)
	embeds := tensor.New(tensor.F64, tensor.Shape{5, 8})
	out, _ := pt.Forward(backend.NewContext(), embeds)
	fmt.Println(out.Shape())
	// Output: (8, 8)
}
