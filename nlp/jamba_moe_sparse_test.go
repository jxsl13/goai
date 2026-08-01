package nlp_test

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestJambaMoESkippingUnselectedExpertsIsExact pins the claim behind evaluating only the routed
// experts: an unselected expert contributes expert(x) times a weight that is EXACTLY zero, and
// adding an exact zero to a finite accumulator returns that accumulator unchanged. So the sparse
// result is not close to the dense one, it is the same bits, and the assertion is exact.
//
// The reference here is the dense sum this code used to compute — every expert, in ascending
// index order, weighted — built from the public API rather than from a stored golden, so it stays
// honest if the expert or router shapes change.
func TestJambaMoESkippingUnselectedExpertsIsExact(t *testing.T) {
	const dim, ffn, nExp, topK, seq = 24, 48, 6, 2, 2 // 2 tokens x top-2 can route to at most 4 of 6
	rng := rand.New(rand.NewSource(21))
	fill := func(sh tensor.Shape, scale float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, sh)
		s := x.Storage().F64()
		for i := range s {
			s[i] = rng.NormFloat64() * scale
		}
		return x
	}
	m := &nlp.JambaMoE{Router: fill(tensor.Shape{dim, nExp}, 0.9), TopK: topK}
	for range nExp {
		m.Experts = append(m.Experts, &nn.SwiGLU{
			Wgate: fill(tensor.Shape{dim, ffn}, 0.12),
			Wup:   fill(tensor.Shape{dim, ffn}, 0.12),
			Wdown: fill(tensor.Shape{ffn, dim}, 0.12),
		})
	}
	x := fill(tensor.Shape{seq, dim}, 0.7)
	ctx := backend.NewContext()

	got, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}

	// Dense reference: recompute the router's scores, keep the top-k per token, and sum every
	// expert's weighted output in ascending index order — zero-weight terms included.
	logits, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, m.Router}, nil)
	if err != nil {
		t.Fatal(err)
	}
	sc, err := backend.Execute(ctx, backend.OpSoftmax, logits, nil)
	if err != nil {
		t.Fatal(err)
	}
	scores := sc[0]
	weight := tensor.New(tensor.F64, tensor.Shape{seq, nExp})
	ws := weight.Storage().F64()
	for tk := range seq {
		idx := make([]int, nExp)
		for i := range idx {
			idx[i] = i
		}
		// Random continuous scores make ties vanishingly unlikely, so this ordering agrees with
		// the router's own selection without having to reproduce its tie-breaking.
		sort.SliceStable(idx, func(a, b int) bool {
			return scores.AtF64(tk, idx[a]) > scores.AtF64(tk, idx[b])
		})
		for _, i := range idx[:topK] {
			ws[tk*nExp+i] = scores.AtF64(tk, i)
		}
	}
	var want *tensor.Tensor
	for i := range nExp {
		out, err := m.Experts[i].Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		wcol, err := weight.Slice(1, i, i+1)
		if err != nil {
			t.Fatal(err)
		}
		term, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{out, wcol.Contiguous()}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if want == nil {
			want = term[0]
			continue
		}
		sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{want, term[0]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		want = sum[0]
	}

	// A sanity floor on the fixture itself: with topK < nExp some expert must have been skipped,
	// otherwise this test would be comparing the dense path against itself.
	skipped := 0
	for i := range nExp {
		all := true
		for tk := range seq {
			if ws[tk*nExp+i] != 0 {
				all = false
				break
			}
		}
		if all {
			skipped++
		}
	}
	if skipped == 0 {
		t.Fatalf("every expert was routed to by some token, so nothing was skipped — the fixture " +
			"does not exercise the change")
	}
	for r := range seq {
		for c := range dim {
			g, w := got.AtF64(r, c), want.AtF64(r, c)
			if math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("[%d,%d]: sparse %v, dense %v — not bit-identical (%d experts unrouted)",
					r, c, g, w, skipped)
			}
		}
	}
}
