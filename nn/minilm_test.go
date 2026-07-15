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

// mlmRelRef is the pure-Go reference for the per-head MiniLM relation maps:
// softmax(x_h·y_hᵀ/√dh) per relation head h — [heads][seq][seq].
func mlmRelRef(x, y [][]float64, heads int) [][][]float64 {
	seq, d := len(x), len(x[0])
	dh := d / heads
	out := make([][][]float64, heads)
	for h := range heads {
		m := make([][]float64, seq)
		for i := range seq {
			row := make([]float64, seq)
			for j := range seq {
				var dot float64
				for c := h * dh; c < (h+1)*dh; c++ {
					dot += x[i][c] * y[j][c]
				}
				row[j] = dot / math.Sqrt(float64(dh))
			}
			m[i] = softmaxRow(row)
		}
		out[h] = m
	}
	return out
}

// mlmPickRef selects the (x,y) operands of a relation within one model's Q/K/V.
func mlmPickRef(q, k, v [][]float64, r nn.MiniLMRelation) ([][]float64, [][]float64) {
	switch r {
	case nn.MiniLMRelationQK:
		return q, k
	case nn.MiniLMRelationQQ:
		return q, q
	case nn.MiniLMRelationKK:
		return k, k
	default:
		return v, v
	}
}

// mlmLossRef independently computes the MiniLM loss: per relation, the KL
// between teacher and student relation rows, averaged over heads·seq and
// summed over relations. KL = Σ p·(log p − log q) with p = teacher (klRow).
func mlmLossRef(sq, sk, sv, tq, tk, tv [][]float64, heads int, rels []nn.MiniLMRelation) float64 {
	seq := len(sq)
	var total float64
	for _, r := range rels {
		sx, sy := mlmPickRef(sq, sk, sv, r)
		tx, ty := mlmPickRef(tq, tk, tv, r)
		sRel := mlmRelRef(sx, sy, heads)
		tRel := mlmRelRef(tx, ty, heads)
		for h := range heads {
			for i := range seq {
				total += klRow(tRel[h][i], sRel[h][i])
			}
		}
	}
	return total / float64(heads*seq)
}

// mlmV2 is the MiniLMv2 multi-relation setting (arXiv:2012.15828 §3.2).
var mlmV2 = []nn.MiniLMRelation{nn.MiniLMRelationQQ, nn.MiniLMRelationKK, nn.MiniLMRelationVV}

// Fixed student (width 4 → heads=2, head_dim 2) and teacher projections; the
// wide teacher (width 8 → head_dim 4) exercises mismatched widths.
var (
	mlmSQ = [][]float64{{0.5, -1.2, 0.3, 0.8}, {1.1, 0.4, -0.6, 0.2}, {-0.3, 0.9, 1.4, -0.5}}
	mlmSK = [][]float64{{-0.7, 0.2, 1.0, -0.4}, {0.6, -1.1, 0.5, 0.9}, {1.3, 0.1, -0.8, 0.4}}
	mlmSV = [][]float64{{0.9, 0.3, -1.4, 0.6}, {-0.2, 1.2, 0.7, -0.9}, {0.4, -0.6, 0.1, 1.1}}
	mlmTQ = [][]float64{{1.0, -0.5, 0.2, 0.7}, {-0.8, 0.9, 0.4, -0.1}, {0.3, 0.6, -1.2, 0.5}}
	mlmTK = [][]float64{{0.2, 0.8, -0.3, 1.1}, {1.4, -0.6, 0.9, 0.1}, {-0.5, 0.3, 0.7, -1.0}}
	mlmTV = [][]float64{{-1.1, 0.4, 0.8, 0.2}, {0.5, 1.0, -0.7, 0.6}, {0.9, -0.3, 0.2, -0.8}}

	mlmWideTQ = [][]float64{
		{0.4, -0.9, 0.6, 0.1, -0.3, 0.8, 0.2, -0.5},
		{-0.6, 0.3, 1.1, -0.2, 0.7, -0.4, 0.9, 0.5},
		{0.8, 0.2, -0.7, 1.0, -0.1, 0.6, -0.9, 0.3},
	}
	mlmWideTK = [][]float64{
		{1.2, 0.1, -0.4, 0.7, 0.3, -0.8, 0.5, 0.9},
		{-0.2, 0.6, 0.8, -1.1, 0.4, 0.2, -0.6, 0.1},
		{0.5, -0.3, 0.2, 0.9, -0.7, 1.0, 0.3, -0.4},
	}
	mlmWideTV = [][]float64{
		{-0.5, 0.9, 0.3, -0.2, 1.1, 0.4, -0.8, 0.6},
		{0.7, -0.1, -0.9, 0.5, 0.2, -0.6, 1.0, 0.3},
		{0.1, 0.8, 0.4, -0.7, -0.3, 0.9, 0.2, -1.2},
	}
)

// §V16 tier-1: the relation maps — the attention relation softmax(Q·Kᵀ/√dh)
// and the value relation softmax(V·Vᵀ/√dh) — match a pure-Go reference of the
// scaled-Gram-softmax at 1e-10, per head and per entry.
func TestMiniLMRelationMapsReference(t *testing.T) {
	const heads = 2
	cases := []struct {
		name string
		x, y [][]float64
	}{
		{"attention Q-K", mlmSQ, mlmSK},
		{"value V-V", mlmSV, mlmSV},
		{"wide teacher V-V", mlmWideTV, mlmWideTV},
	}
	for _, c := range cases {
		got, err := nn.MiniLMRelationMaps(backend.NewContext(), tt(c.x), tt(c.y), heads)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		want := mlmRelRef(c.x, c.y, heads)
		if len(got) != heads {
			t.Fatalf("%s: got %d maps, want %d", c.name, len(got), heads)
		}
		for h := range heads {
			for i := range c.x {
				for j := range c.x {
					if diff := math.Abs(got[h].AtF64(i, j) - want[h][i][j]); diff > 1e-10 {
						t.Errorf("%s head %d [%d,%d]: got %.12g want %.12g (diff %g)",
							c.name, h, i, j, got[h].AtF64(i, j), want[h][i][j], diff)
					}
				}
			}
		}
	}
}

// §V16 tier-1: MiniLMLoss matches the independent reference — relation maps
// plus KL term Σ p·(log p − log q) — at 1e-10, for the v1 default (Q-K + V-V),
// the v2 multi-relation setting, and the attention-only ablation.
func TestMiniLMLossFormula(t *testing.T) {
	const heads = 2
	sq, sk, sv := tt(mlmSQ), tt(mlmSK), tt(mlmSV)
	tq, tk, tv := tt(mlmTQ), tt(mlmTK), tt(mlmTV)
	cases := []struct {
		name string
		rels []nn.MiniLMRelation
		opts []nn.MiniLMOption
	}{
		{"v1 default QK+VV", []nn.MiniLMRelation{nn.MiniLMRelationQK, nn.MiniLMRelationVV}, nil},
		{"v2 QQ+KK+VV", mlmV2, []nn.MiniLMOption{nn.WithMiniLMRelations(mlmV2...)}},
		{"QK only", []nn.MiniLMRelation{nn.MiniLMRelationQK},
			[]nn.MiniLMOption{nn.WithMiniLMRelations(nn.MiniLMRelationQK)}},
	}
	for _, c := range cases {
		got, err := nn.MiniLMLoss(backend.NewContext(), sq, sk, sv, tq, tk, tv, heads, c.opts...)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		want := mlmLossRef(mlmSQ, mlmSK, mlmSV, mlmTQ, mlmTK, mlmTV, heads, c.rels)
		if math.Abs(got.AtF64()-want) > 1e-10 {
			t.Errorf("%s: MiniLMLoss = %.12g, want %.12g", c.name, got.AtF64(), want)
		}
	}
}

// §V16 tier-3 (the defining MiniLM property): teacher head_dim 4, student
// head_dim 2, same heads and seq — the loss is computable, finite, positive
// and matches the reference, even though the raw Q/K/V tensors cannot be
// compared directly (shapes [3,4] vs [3,8] — a naive "match the projections"
// distiller has no valid elementwise loss here at all).
func TestMiniLMWidthIndependence(t *testing.T) {
	const heads = 2
	sq, sk, sv := tt(mlmSQ), tt(mlmSK), tt(mlmSV)
	tq, tk, tv := tt(mlmWideTQ), tt(mlmWideTK), tt(mlmWideTV)
	if sq.Shape().Equal(tq.Shape()) {
		t.Fatal("test setup: student and teacher widths must differ")
	}
	for _, opts := range [][]nn.MiniLMOption{nil, {nn.WithMiniLMRelations(mlmV2...)}} {
		got, err := nn.MiniLMLoss(backend.NewContext(), sq, sk, sv, tq, tk, tv, heads, opts...)
		if err != nil {
			t.Fatalf("mismatched widths must be distillable (relations are [heads,seq,seq]): %v", err)
		}
		v := got.AtF64()
		if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
			t.Errorf("width-independent loss must be finite and positive, got %g", v)
		}
		rels := []nn.MiniLMRelation{nn.MiniLMRelationQK, nn.MiniLMRelationVV}
		if opts != nil {
			rels = mlmV2
		}
		want := mlmLossRef(mlmSQ, mlmSK, mlmSV, mlmWideTQ, mlmWideTK, mlmWideTV, heads, rels)
		if math.Abs(v-want) > 1e-10 {
			t.Errorf("wide-teacher loss = %.12g, want %.12g", v, want)
		}
	}
}

// §V2: finite-difference gradcheck of the loss w.r.t. the STUDENT Q, K and V
// through slice/matmul/softmax/KL, for both the v1 and v2 relation sets — and
// the detached TEACHER tensors receive zero gradient (stop-gradient).
func TestMiniLMGradcheck(t *testing.T) {
	const heads = 2
	for _, c := range []struct {
		name string
		opts []nn.MiniLMOption
	}{
		{"v1", nil},
		{"v2", []nn.MiniLMOption{nn.WithMiniLMRelations(mlmV2...)}},
	} {
		sq, sk, sv := tt(mlmSQ), tt(mlmSK), tt(mlmSV)
		tq, tk, tv := tt(mlmWideTQ), tt(mlmWideTK), tt(mlmWideTV)

		tape := autograd.NewTape()
		loss, err := nn.MiniLMLoss(tape.Context(), sq, sk, sv, tq, tk, tv, heads, c.opts...)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}

		// teacher gets ZERO grad: nil (never reached) or an all-zero tensor.
		for i, teach := range []*tensor.Tensor{tq, tk, tv} {
			g := tape.Grad(teach)
			if g == nil {
				continue
			}
			for e := range g.Numel() {
				co := tensor.Unravel(e, g.Shape())
				if g.AtF64(co...) != 0 {
					t.Errorf("%s: teacher tensor %d got nonzero grad %g at %v", c.name, i, g.AtF64(co...), co)
				}
			}
		}

		const h = 1e-6
		var maxRel float64
		checked := 0
		for _, s := range []*tensor.Tensor{sq, sk, sv} {
			g := tape.Grad(s)
			if g == nil {
				t.Fatalf("%s: nil student gradient", c.name)
			}
			for e := range s.Numel() {
				co := tensor.Unravel(e, s.Shape())
				orig := s.AtF64(co...)
				s.SetF64(orig+h, co...)
				lp, _ := nn.MiniLMLoss(backend.NewContext(), sq, sk, sv, tq, tk, tv, heads, c.opts...)
				s.SetF64(orig-h, co...)
				lm, _ := nn.MiniLMLoss(backend.NewContext(), sq, sk, sv, tq, tk, tv, heads, c.opts...)
				s.SetF64(orig, co...)
				num := (lp.AtF64() - lm.AtF64()) / (2 * h)
				ana := g.AtF64(co...)
				rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
				maxRel = math.Max(maxRel, rel)
				checked++
				if rel > 1e-5 {
					t.Errorf("%s: ∂/∂student%v: numeric %.8g vs analytic %.8g", c.name, co, num, ana)
				}
			}
		}
		t.Logf("%s: gradcheck %d entries, max rel err %.3g", c.name, checked, maxRel)
	}
}

// §V16 tier-1 collapse/sanity: identical teacher and student → loss ≈ 0 (KL of
// identical distributions); the loss is non-negative for distinct inputs; and
// repeated evaluation is bitwise deterministic.
func TestMiniLMCollapseNonNegativeDeterministic(t *testing.T) {
	q, k, v := tt(mlmSQ), tt(mlmSK), tt(mlmSV)
	zero, err := nn.MiniLMLoss(backend.NewContext(), q, k, v, q, k, v, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(zero.AtF64()) > 1e-12 {
		t.Errorf("teacher==student: loss = %g, want ≈ 0", zero.AtF64())
	}

	tq, tk, tv := tt(mlmTQ), tt(mlmTK), tt(mlmTV)
	for _, opts := range [][]nn.MiniLMOption{nil, {nn.WithMiniLMRelations(mlmV2...)}} {
		a, err := nn.MiniLMLoss(backend.NewContext(), q, k, v, tq, tk, tv, 2, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if a.AtF64() < 0 {
			t.Errorf("KL-based loss must be ≥ 0, got %g", a.AtF64())
		}
		b, err := nn.MiniLMLoss(backend.NewContext(), q, k, v, tq, tk, tv, 2, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if a.AtF64() != b.AtF64() {
			t.Errorf("nondeterministic loss: %.17g vs %.17g", a.AtF64(), b.AtF64())
		}
	}
}

// §V16 tier-5 e2e: a tiny student (width 4) trained by SGD on the MiniLM loss
// alone learns to mimic a FIXED wider teacher's (width 8) attention behavior —
// the relation KL drops by well over an order of magnitude.
func TestMiniLMTrainingReducesKL(t *testing.T) {
	const (
		seq, dIn = 5, 4
		dS, dT   = 4, 8
		heads    = 2
		lr       = 1.0
		steps    = 300
	)
	fill := func(rows, cols int, seed float64) *tensor.Tensor {
		m := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for r := range rows {
			for c := range cols {
				m.SetF64(math.Sin(seed+1.7*float64(r*cols+c))*0.9, r, c)
			}
		}
		return m
	}
	x := fill(seq, dIn, 0.3)                                                    // shared input
	wtq, wtk, wtv := fill(dIn, dT, 1.1), fill(dIn, dT, 2.2), fill(dIn, dT, 3.3) // frozen teacher
	wq, wk, wv := fill(dIn, dS, 4.4), fill(dIn, dS, 5.5), fill(dIn, dS, 6.6)    // trainable student

	mm := func(ctx *backend.Context, a, b *tensor.Tensor) *tensor.Tensor {
		o, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return o[0]
	}
	cctx := backend.NewContext()
	tq, tk, tv := mm(cctx, x, wtq), mm(cctx, x, wtk), mm(cctx, x, wtv)

	var first, last float64
	for step := range steps {
		tape := autograd.NewTape()
		ctx := tape.Context()
		sq, sk, sv := mm(ctx, x, wq), mm(ctx, x, wk), mm(ctx, x, wv)
		loss, err := nn.MiniLMLoss(ctx, sq, sk, sv, tq, tk, tv, heads)
		if err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = loss.AtF64()
		}
		last = loss.AtF64()
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		for _, w := range []*tensor.Tensor{wq, wk, wv} {
			g := tape.Grad(w)
			if g == nil {
				t.Fatal("nil gradient for student projection")
			}
			for e := range w.Numel() {
				co := tensor.Unravel(e, w.Shape())
				w.SetF64(w.AtF64(co...)-lr*g.AtF64(co...), co...)
			}
		}
	}
	t.Logf("MiniLM KL: first %.6g → last %.6g (%.1f×)", first, last, first/last)
	if !(last < first*0.1) {
		t.Errorf("student did not move toward the teacher's relations: KL %.6g → %.6g", first, last)
	}
}

// Invalid inputs are rejected with errors, not panics.
func TestMiniLMErrors(t *testing.T) {
	q, k, v := tt(mlmSQ), tt(mlmSK), tt(mlmSV)
	tq, tk, tv := tt(mlmTQ), tt(mlmTK), tt(mlmTV)
	ctx := backend.NewContext()
	if _, err := nn.MiniLMLoss(ctx, q, k, v, tq, tk, tv, 3); err == nil {
		t.Error("heads not dividing width: want error")
	}
	if _, err := nn.MiniLMLoss(ctx, q, k, v, tq, tk, tv, 0); err == nil {
		t.Error("heads < 1: want error")
	}
	if _, err := nn.MiniLMLoss(ctx, q, k, tt([][]float64{{1, 2, 3, 4}}), tq, tk, tv, 2); err == nil {
		t.Error("student Q/K/V shape mismatch: want error")
	}
	if _, err := nn.MiniLMLoss(ctx, q, k, v, tt(mlmWideTQ[:2]), tt(mlmWideTK[:2]), tt(mlmWideTV[:2]), 2); err == nil {
		t.Error("seq mismatch: want error")
	}
	if _, err := nn.MiniLMLoss(ctx, q, k, v, tq, tk, tv, 2, nn.WithMiniLMRelations()); err == nil {
		t.Error("empty relation list: want error")
	}
	if _, err := nn.MiniLMLoss(ctx, q, k, v, tq, tk, tv, 2, nn.WithMiniLMRelations(nn.MiniLMRelation(99))); err == nil {
		t.Error("unknown relation: want error")
	}
	f32 := tensor.New(tensor.F32, tensor.Shape{3, 4})
	if _, err := nn.MiniLMLoss(ctx, q, k, v, f32, tk, tv, 2); err == nil {
		t.Error("dtype mismatch: want error")
	}
	rank1 := tensor.New(tensor.F64, tensor.Shape{4})
	if _, err := nn.MiniLMLoss(ctx, rank1, rank1, rank1, tq, tk, tv, 2); err == nil {
		t.Error("rank-1 input: want error")
	}
	if _, err := nn.MiniLMRelationMaps(ctx, q, tt(mlmWideTQ), 2); err == nil {
		t.Error("MiniLMRelationMaps shape mismatch: want error")
	}
	if _, err := nn.MiniLMRelationMaps(ctx, q, k, 3); err == nil {
		t.Error("MiniLMRelationMaps heads not dividing dim: want error")
	}
}

func ExampleMiniLMLoss() {
	// A student whose last layer already produces the teacher's Q/K/V relations
	// has nothing left to distill: the KL between identical attention and
	// value-relation maps is zero.
	q := tt([][]float64{{0.5, -1.2, 0.3, 0.8}, {1.1, 0.4, -0.6, 0.2}})
	k := tt([][]float64{{-0.7, 0.2, 1.0, -0.4}, {0.6, -1.1, 0.5, 0.9}})
	v := tt([][]float64{{0.9, 0.3, -1.4, 0.6}, {-0.2, 1.2, 0.7, -0.9}})
	loss, _ := nn.MiniLMLoss(backend.NewContext(), q, k, v, q, k, v, 2)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output:
	// 0.0000
}

func ExampleWithMiniLMRelations() {
	// MiniLMv2 multi-relation transfer (Q-Q, K-K, V-V) from a teacher TWICE the
	// student's width: relation maps are [seq,seq], so mismatched widths distill.
	v2 := []nn.MiniLMRelation{nn.MiniLMRelationQQ, nn.MiniLMRelationKK, nn.MiniLMRelationVV}
	sq, sk, sv := tt(mlmSQ), tt(mlmSK), tt(mlmSV)             // [3,4] student
	tq, tk, tv := tt(mlmWideTQ), tt(mlmWideTK), tt(mlmWideTV) // [3,8] teacher
	loss, _ := nn.MiniLMLoss(backend.NewContext(), sq, sk, sv, tq, tk, tv, 2,
		nn.WithMiniLMRelations(v2...))
	fmt.Printf("positive and finite: %t\n", loss.AtF64() > 0 && !math.IsInf(loss.AtF64(), 0))
	// Output:
	// positive and finite: true
}
