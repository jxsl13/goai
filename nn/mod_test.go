package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// modRouterLogits computes r_i = x_i · w_router independently.
func modRouterLogits(x [][]float64, router []float64) []float64 {
	r := make([]float64, len(x))
	for i := range x {
		var s float64
		for j := range x[i] {
			s += x[i][j] * router[j]
		}
		r[i] = s
	}
	return r
}

// modTopK returns the ascending positions of the k largest logits (ties → lower pos),
// matching the implementation's selection.
func modTopK(logits []float64, k int) []int {
	pos := make([]int, len(logits))
	for i := range pos {
		pos[i] = i
	}
	sort.SliceStable(pos, func(a, b int) bool { return logits[pos[a]] > logits[pos[b]] })
	sel := append([]int(nil), pos[:k]...)
	sort.Ints(sel)
	return sel
}

func setCol(t *tensor.Tensor, vals []float64) {
	for i, v := range vals {
		t.SetF64(v, i, 0)
	}
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2404.02258 §3.4 Eq.1): Route selects exactly
// the top-k tokens by raw router logit, gathers their rows, and returns the RAW
// logit as the weight (no sigmoid).
func TestMoDRouteSelection(t *testing.T) {
	x := [][]float64{{1, 0}, {0, 1}, {2, 2}, {-1, 0.5}, {0.3, 3}}
	router := []float64{1, 0.5}
	m := nn.NewMixtureOfDepths(tensor.F64, 2, 0.5) // k = round(0.5·5) = 3 (round half away from 0 ⇒ 2.5→3)
	setCol(m.Router, router)
	xt := toT(x)
	gathered, weights, sel, err := m.Route(backend.NewContext(), xt)
	if err != nil {
		t.Fatal(err)
	}
	logits := modRouterLogits(x, router)
	wantK := m.NumProcessed(5)
	wantIdx := modTopK(logits, wantK)
	if len(sel.Idx) != wantK {
		t.Fatalf("selected %d tokens, want k=%d", len(sel.Idx), wantK)
	}
	for i, p := range wantIdx {
		if sel.Idx[i] != p {
			t.Errorf("Idx[%d]=%d want %d", i, sel.Idx[i], p)
		}
		// gathered row == the original token row
		for j := 0; j < 2; j++ {
			if math.Abs(gathered.AtF64(i, j)-x[p][j]) > 1e-12 {
				t.Errorf("gathered[%d,%d]=%g want x[%d][%d]=%g", i, j, gathered.AtF64(i, j), p, j, x[p][j])
			}
		}
		// weight == RAW router logit r_p (not sigmoid)
		if math.Abs(weights.AtF64(i, 0)-logits[p]) > 1e-9 {
			t.Errorf("weight[%d]=%g want raw logit %g", i, weights.AtF64(i, 0), logits[p])
		}
	}
}

// §V2 (defining property, Eq.1): Combine gives selected tokens x_i + r_i·processed_i
// and leaves skipped tokens as pure residual x_i.
func TestMoDCombineResidual(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 7))
	const seq, d = 6, 3
	x := randMat(rng, seq, d)
	m := nn.NewMixtureOfDepths(tensor.F64, d, 0.5) // k=3
	for j := 0; j < d; j++ {
		m.Router.SetF64(rng.NormFloat64(), j, 0)
	}
	gathered, weights, sel, err := m.Route(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// a non-trivial "block": processed = 2·gathered + 1.
	processed := tensor.New(tensor.F64, gathered.Shape())
	for i := range gathered.Numel() {
		c := tensor.Unravel(i, gathered.Shape())
		processed.SetF64(2*gathered.AtF64(c...)+1, c...)
	}
	out, err := m.Combine(backend.NewContext(), x, processed, weights, sel)
	if err != nil {
		t.Fatal(err)
	}
	selected := map[int]int{}
	for i, p := range sel.Idx {
		selected[p] = i
	}
	for p := 0; p < seq; p++ {
		_, isSel := selected[p]
		for j := 0; j < d; j++ {
			want := x.AtF64(p, j)
			if isSel {
				want += weights.AtF64(selected[p], 0) * processed.AtF64(selected[p], j)
			}
			if g := out.AtF64(p, j); math.Abs(g-want) > 1e-9 {
				t.Errorf("out[%d,%d]=%.10g want %.10g (selected=%v)", p, j, g, want, isSel)
			}
		}
	}
}

// §V16 tier-1: capacity k = round(capacity·seq), clamped to [1,seq].
func TestMoDNumProcessed(t *testing.T) {
	cases := []struct {
		cap       float64
		seq, want int
	}{
		{0.125, 2048, 256}, {0.5, 5, 3}, {0.25, 8, 2}, {0.001, 4, 1}, {1, 10, 10},
	}
	for _, c := range cases {
		m := nn.NewMixtureOfDepths(tensor.F64, 4, c.cap)
		if got := m.NumProcessed(c.seq); got != c.want {
			t.Errorf("NumProcessed(cap=%g,seq=%d)=%d want %d", c.cap, c.seq, got, c.want)
		}
	}
}

// §V2: the routed residual is differentiable w.r.t. x and the router (the selection
// is treated as constant, per the paper's straight-through routing).
func TestMoDGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(43, 13))
	const seq, d = 5, 3
	x := randMat(rng, seq, d)
	m := nn.NewMixtureOfDepths(tensor.F64, d, 0.5)
	for j := 0; j < d; j++ {
		m.Router.SetF64(rng.NormFloat64(), j, 0)
	}
	// identity block (processed = gathered) so the pipeline is Route→Combine.
	fwd := func(ctx *backend.Context) (*tensor.Tensor, error) {
		g, w, sel, err := m.Route(ctx, x)
		if err != nil {
			return nil, err
		}
		out, err := m.Combine(ctx, x, g, w, sel)
		if err != nil {
			return nil, err
		}
		return sumAll(ctx, out)
	}
	tape := autograd.NewTape()
	loss, err := fwd(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	scalarFwd := func() float64 {
		l, _ := fwd(backend.NewContext())
		return l.AtF64()
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := scalarFwd()
			p.SetF64(o-h, coord...)
			lm := scalarFwd()
			p.SetF64(o, coord...)
			if num, ana := (lp-lm)/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("x", x)
	check("router", m.Router)
}

func TestMoDErrors(t *testing.T) {
	m := nn.NewMixtureOfDepths(tensor.F64, 3, 0.5)
	if _, _, _, err := m.Route(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 1})); err == nil {
		t.Error("rank-3 input: want error")
	}
	if _, _, _, err := m.Route(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 5})); err == nil {
		t.Error("router dim mismatch: want error")
	}
	x := tensor.New(tensor.F64, tensor.Shape{4, 3})
	_, w, sel, _ := m.Route(backend.NewContext(), x)
	if _, err := m.Combine(backend.NewContext(), x, w, w, nil); err == nil {
		t.Error("nil selection: want error")
	}
	if _, err := m.Combine(backend.NewContext(), x, tensor.New(tensor.F64, tensor.Shape{99, 3}), w, sel); err == nil {
		t.Error("processed wrong shape: want error")
	}
}

func ExampleMixtureOfDepths() {
	// 3 tokens, capacity ½ ⇒ 2 processed; router prefers high first-feature tokens.
	x := toT([][]float64{{1, 0}, {0, 1}, {2, 2}})
	m := nn.NewMixtureOfDepths(tensor.F64, 2, 0.5)
	m.Router.SetF64(1, 0, 0) // logit = x[:,0]
	m.Router.SetF64(0, 1, 0)
	g, w, sel, _ := m.Route(backend.NewContext(), x)
	out, _ := m.Combine(backend.NewContext(), x, g, w, sel) // identity block
	// token 1 (logit 0) is NOT in the top-2 ⇒ passes through unchanged.
	fmt.Println(m.NumProcessed(3), out.AtF64(1, 0), out.AtF64(1, 1))
	// Output:
	// 2 0 1
}
