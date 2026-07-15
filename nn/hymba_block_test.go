package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the Hymba block maps [T,dim]→[T,dim] and exposes every learnable
// tensor: 8 bare tensors (ConvW, ConvB, ALog, Dskip, both norm γ, β₁, β₂) + 6
// Linears × 2 (InProj, DtLow, BProj, CProj, DtProj, OutProj) = 20.
func TestHymbaBlockShapes(t *testing.T) {
	const dim, heads, ssmState, seq = 8, 2, 3, 5
	b, err := nn.NewHymbaBlock(tensor.F64, dim, heads, ssmState, 7)
	if err != nil {
		t.Fatal(err)
	}
	if b.DInner != dim {
		t.Errorf("DInner=%d, want fusion width == dim %d", b.DInner, dim)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, seq, dim)
	out, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), seq, dim)
	}
	if got, want := len(b.Params()), 20; got != want {
		t.Errorf("Params count %d, want %d", got, want)
	}
}

// §V16: the constructor rejects a dim not divisible by heads, non-positive
// dims, a negative meta-token count, and Forward rejects a wrong input width.
func TestHymbaBlockErrors(t *testing.T) {
	if _, err := nn.NewHymbaBlock(tensor.F64, 7, 2, 3, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewHymbaBlock(tensor.F64, 0, 1, 3, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewHymbaBlock(tensor.F64, 8, 2, 0, 1); err == nil {
		t.Error("ssmState=0: want error")
	}
	if _, err := nn.NewHymbaBlock(tensor.F64, 8, 2, 3, 1, nn.WithHymbaMetaTokens(-1)); err == nil {
		t.Error("negative meta tokens: want error")
	}
	b, _ := nn.NewHymbaBlock(tensor.F64, 8, 2, 3, 1)
	if _, err := b.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
}

// §V16 (b) BOTH BRANCHES CONTRIBUTE: zeroing β₁ (killing the attention stream)
// and zeroing β₂ (killing the SSM stream) each change the output versus the
// full fusion, and the two single-branch outputs differ from each other — the
// two streams are genuinely fused, neither is ignored.
func TestHymbaBothBranchesContribute(t *testing.T) {
	const dim, heads, ssmState, seq, seed = 8, 2, 3, 5, 11
	rng := rand.New(rand.NewPCG(5, 6))
	x := randMat(rng, seq, dim)
	forward := func(kill *int) *tensor.Tensor { // kill: nil=full, 1=zero β₁, 2=zero β₂
		b, err := nn.NewHymbaBlock(tensor.F64, dim, heads, ssmState, seed)
		if err != nil {
			t.Fatal(err)
		}
		if kill != nil && *kill == 1 {
			nn.Zeros(b.Beta1)
		}
		if kill != nil && *kill == 2 {
			nn.Zeros(b.Beta2)
		}
		out, err := b.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	one, two := 1, 2
	full, noAttn, noSSM := forward(nil), forward(&one), forward(&two)
	maxDiff := func(a, b *tensor.Tensor) float64 {
		var m float64
		for i := range a.Numel() {
			c := tensor.Unravel(i, a.Shape())
			if d := math.Abs(a.AtF64(c...) - b.AtF64(c...)); d > m {
				m = d
			}
		}
		return m
	}
	if d := maxDiff(full, noAttn); d < 1e-9 {
		t.Errorf("zeroing β₁ changed nothing (maxdiff %.3g) — attention branch not contributing", d)
	}
	if d := maxDiff(full, noSSM); d < 1e-9 {
		t.Errorf("zeroing β₂ changed nothing (maxdiff %.3g) — SSM branch not contributing", d)
	}
	if d := maxDiff(noAttn, noSSM); d < 1e-9 {
		t.Errorf("attention-only and SSM-only outputs identical (maxdiff %.3g) — one branch dominating", d)
	}
}

// §V16 (c) GRADIENT FLOW: central finite differences confirm gradients reach
// the SHARED in-proj (fed by both branches through the slice), the SSM branch
// (conv + B projection), β₁/β₂, a fusion-norm gain, and the shared out-proj.
func TestHymbaBlockGradcheck(t *testing.T) {
	const dim, heads, ssmState, seq, h = 4, 2, 2, 3, 1e-6
	b, err := nn.NewHymbaBlock(tensor.F64, dim, heads, ssmState, 3)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, seq, dim)

	tape := autograd.NewTape()
	out, err := b.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := b.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — no gradient reached this parameter", name)
		}
		var nz bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", name)
		}
	}
	check("InProj.W", b.InProj.W) // SHARED projection — must collect grads from BOTH branches
	check("ConvW", b.ConvW)       // SSM branch conv
	check("BProj.W", b.BProj.W)   // SSM branch input-dependent B
	check("Beta1", b.Beta1)       // attention fusion scale
	check("Beta2", b.Beta2)       // SSM fusion scale
	check("NormAttn.Gamma", b.NormAttn.Gamma)
	check("OutProj.W", b.OutProj.W) // shared out-proj
}

// §V16 (d) META TOKENS: WithHymbaMetaTokens(m) prepends m learnable rows —
// the output stays [T,dim] (meta rows sliced off) but its values change (the
// causal mixers see the prefix), the meta tokens are trainable (a non-nil,
// non-zero gradient reaches them), and m=0 is byte-identical to no option.
func TestHymbaMetaTokens(t *testing.T) {
	const dim, heads, ssmState, seq, m, seed = 8, 2, 3, 5, 3, 17
	rng := rand.New(rand.NewPCG(9, 4))
	x := randMat(rng, seq, dim)

	plain, err := nn.NewHymbaBlock(tensor.F64, dim, heads, ssmState, seed)
	if err != nil {
		t.Fatal(err)
	}
	meta, err := nn.NewHymbaBlock(tensor.F64, dim, heads, ssmState, seed, nn.WithHymbaMetaTokens(m))
	if err != nil {
		t.Fatal(err)
	}
	if meta.MetaTokens == nil || !meta.MetaTokens.Shape().Equal(tensor.Shape{m, dim}) {
		t.Fatalf("MetaTokens shape %v, want [%d %d]", meta.MetaTokens, m, dim)
	}
	if got, want := len(meta.Params()), 21; got != want {
		t.Errorf("Params count with meta tokens %d, want %d", got, want)
	}

	outPlain, err := plain.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	tape := autograd.NewTape()
	outMeta, err := meta.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !outMeta.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("meta output shape %v, want [%d %d] (meta rows must be sliced off)", outMeta.Shape(), seq, dim)
	}
	var changed bool
	for i := range outMeta.Numel() {
		c := tensor.Unravel(i, outMeta.Shape())
		if math.Abs(outMeta.AtF64(c...)-outPlain.AtF64(c...)) > 1e-9 {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("meta tokens had no effect on the output — prefix not reaching the mixers")
	}
	// trainable: gradient reaches the meta-token embeddings
	loss, err := sumAll(tape.Context(), outMeta)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(meta.MetaTokens)
	if g == nil {
		t.Fatal("nil grad on MetaTokens — meta tokens not trainable")
	}
	var nz bool
	for i := range g.Numel() {
		if math.Abs(g.AtF64(tensor.Unravel(i, g.Shape())...)) > 1e-12 {
			nz = true
			break
		}
	}
	if !nz {
		t.Error("all-zero grad on MetaTokens — meta tokens not exercised")
	}

	// off-by-default: WithHymbaMetaTokens(0) is byte-identical to no option
	zero, err := nn.NewHymbaBlock(tensor.F64, dim, heads, ssmState, seed, nn.WithHymbaMetaTokens(0))
	if err != nil {
		t.Fatal(err)
	}
	if zero.MetaTokens != nil {
		t.Error("WithHymbaMetaTokens(0) allocated meta tokens; want disabled")
	}
	outZero, err := zero.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range outZero.Numel() {
		c := tensor.Unravel(i, outZero.Shape())
		if outZero.AtF64(c...) != outPlain.AtF64(c...) {
			t.Fatalf("WithHymbaMetaTokens(0) output differs from no-option at %v: %.17g vs %.17g",
				c, outZero.AtF64(c...), outPlain.AtF64(c...))
		}
	}
}

// §V16 (e) E2E TRAINABILITY: a few SGD steps on a fixed regression objective
// reduce the loss, exercising the full forward+backward+update path through
// both parallel branches, the β-norm fusion, and the shared out-proj.
func TestHymbaBlockTrainsE2E(t *testing.T) {
	const dim, heads, ssmState, seq = 6, 2, 3, 4
	b, err := nn.NewHymbaBlock(tensor.F64, dim, heads, ssmState, 5)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(21, 8))
	x := randMat(rng, seq, dim)
	target := randMat(rng, seq, dim)
	opt := nn.NewSGD(b.Params(), 0.02, 0.9)

	step := func() float64 {
		tape := autograd.NewTape()
		out, err := b.Forward(tape.Context(), x)
		if err != nil {
			t.Fatal(err)
		}
		diff, err := backend.Execute(tape.Context(), backend.OpSub, []*tensor.Tensor{out, target}, nil)
		if err != nil {
			t.Fatal(err)
		}
		sq, err := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := sumAll(tape.Context(), sq[0])
		if err != nil {
			t.Fatal(err)
		}
		var l float64
		for i := range loss.Numel() {
			l += loss.AtF64(tensor.Unravel(i, loss.Shape())...)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
		return l
	}

	first := step()
	var last float64
	for range 40 {
		last = step()
	}
	if last >= first {
		t.Errorf("MSE did not decrease over training: first %.6g, last %.6g", first, last)
	}
}

func ExampleHymbaBlock() {
	// A tiny Hymba hybrid-head block: attention and SSM heads run in parallel
	// on the same 4-wide input and are β-norm fused before one shared out-proj.
	b, _ := nn.NewHymbaBlock(tensor.F64, 4, 2, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := b.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(b.Params()))
	// Output: (3, 4) params=20
}
