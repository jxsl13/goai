package nn

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// apolloQuadratic builds the fixed test problem: minimize ½‖W−W*‖² over a
// [rows,cols] matrix from a seeded random start. Returns the parameter, the
// loss closure and the grad closure (grad = W − W*, recomputed from the live W).
func apolloQuadratic(rows, cols int) (*tensor.Tensor, func() float64, GradFn) {
	rng := rand.New(rand.NewPCG(3, 4))
	star := make([]float64, rows*cols)
	init := make([]float64, rows*cols)
	for i := range star {
		star[i] = rng.NormFloat64()
		init[i] = rng.NormFloat64()
	}
	w := tensor.FromFloat64(tensor.Shape{rows, cols}, init)
	loss := func() float64 {
		var s float64
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			d := w.AtF64(idx...) - star[i]
			s += d * d
		}
		return 0.5 * s
	}
	grad := func(*tensor.Tensor) *tensor.Tensor {
		g := tensor.New(tensor.F32, w.Shape())
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			g.SetF64(w.AtF64(idx...)-star[i], idx...)
		}
		return g
	}
	return w, loss, grad
}

// §V16 tier-1 convergence: on a convex quadratic ½‖W−W*‖² APOLLO's
// sketch→Adam→channel-scale-the-raw-gradient path drives the loss far below the
// start, and lands within a small factor of plain Adam's floor on the SAME problem
// with the SAME lr and step budget — the "AdamW-level performance" claim in
// miniature. (Same-loss-shape as TestGaLoreConvergesQuadratic for comparability.)
func TestAPOLLOConvergesQuadratic(t *testing.T) {
	const rows, cols, steps, lr = 5, 7, 2000, 0.1

	w, loss, grad := apolloQuadratic(rows, cols)
	opt := NewAPOLLO([]*tensor.Tensor{w}, lr, 1, WithAPOLLORank(rows))
	l0 := loss()
	for range steps {
		if err := opt.Step(grad); err != nil {
			t.Fatal(err)
		}
	}
	lf := loss()
	if lf > 0.01*l0 {
		t.Errorf("APOLLO did not converge: loss %.4f → %.4f (want < 1%% of initial)", l0, lf)
	}

	// Adam baseline on the identical problem and budget.
	wa, lossA, gradA := apolloQuadratic(rows, cols)
	adam := NewAdam([]*tensor.Tensor{wa}, lr)
	for range steps {
		if err := adam.Step(gradA); err != nil {
			t.Fatal(err)
		}
	}
	la := lossA()
	if lf > 10*la+1e-6 {
		t.Errorf("APOLLO floor %.6g not competitive with Adam floor %.6g (want within 10x)", lf, la)
	}
}

// The whole point of APOLLO: the persistent state is only the two-word random
// projection seed plus rank-r Adam moments, so its numeric payload is
// O(r·max(m,n)), NOT O(m·n) — while the parameter and its applied update stay
// full-rank. Checked for both projection sides.
func TestAPOLLOStateLowRank(t *testing.T) {
	for _, dims := range []struct{ rows, cols int }{{64, 16}, {16, 64}} {
		w := tensor.New(tensor.F32, tensor.Shape{dims.rows, dims.cols})
		g := tensor.New(tensor.F32, w.Shape())
		g.SetF64(1, 0, 0)
		opt := NewAPOLLO([]*tensor.Tensor{w}, 1e-3, 7, WithAPOLLORank(4))
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
		st := opt.st[0]
		full := dims.rows * dims.cols // 1024
		want := 4 * max(dims.rows, dims.cols)
		if len(st.m) != want || len(st.v) != want {
			t.Errorf("dims %v: moment state %d/%d floats, want %d = r*max(m,n)", dims, len(st.m), len(st.v), want)
		}
		if len(st.m) >= full {
			t.Errorf("dims %v: moment state %d not smaller than full %d — no memory win", dims, len(st.m), full)
		}
		if st.rank != 4 || st.left != (dims.rows <= dims.cols) {
			t.Errorf("dims %v: projection metadata rank=%d left=%v", dims, st.rank, st.left)
		}
		proj := st.projection(dims.rows, dims.cols)
		if len(proj) != 4 || len(proj[0]) != min(dims.rows, dims.cols) {
			t.Errorf("dims %v: projection %dx%d, want 4x%d (r rows over the SMALLER dim)",
				dims, len(proj), len(proj[0]), min(dims.rows, dims.cols))
		}
	}
}

// The paper's memory claim depends on regenerating P from a stored seed rather
// than retaining P itself. Repeated regeneration must be bit-identical; a Gap
// boundary must replace the seed and therefore the projection while preserving
// the low-rank moment buffers.
func TestAPOLLOProjectionRegeneratesFromSeed(t *testing.T) {
	w := tensor.New(tensor.F32, tensor.Shape{5, 7})
	g := tensor.New(tensor.F32, w.Shape())
	g.SetF64(1, 0, 0)
	opt := NewAPOLLO([]*tensor.Tensor{w}, 1e-3, 17, WithAPOLLORank(3), WithAPOLLOGap(2))
	grad := func(*tensor.Tensor) *tensor.Tensor { return g }
	if err := opt.Step(grad); err != nil {
		t.Fatal(err)
	}
	st := opt.st[0]
	seed1, seed2 := st.seed1, st.seed2
	m, v := st.m, st.v
	p1, p2 := st.projection(5, 7), st.projection(5, 7)
	for i := range p1 {
		for j := range p1[i] {
			if p1[i][j] != p2[i][j] {
				t.Fatalf("regenerated projection differs at [%d,%d]: %v != %v", i, j, p1[i][j], p2[i][j])
			}
		}
	}

	if err := opt.Step(grad); err != nil { // t=2: Gap boundary
		t.Fatal(err)
	}
	if st.seed1 == seed1 && st.seed2 == seed2 {
		t.Fatal("Gap boundary did not replace projection seed")
	}
	if len(st.m) != len(m) || len(st.v) != len(v) || &st.m[0] != &m[0] || &st.v[0] != &v[0] {
		t.Fatal("Gap boundary replaced low-rank moment buffers")
	}
	p3 := st.projection(5, 7)
	same := true
	for i := range p1 {
		for j := range p1[i] {
			same = same && p1[i][j] == p3[i][j]
		}
	}
	if same {
		t.Fatal("new seed regenerated the old projection")
	}
}

// Determinism: the random projection is driven only by the constructor seed, so the
// same seed replays a bit-identical trajectory; a different seed draws a different
// projection (trajectories diverge) yet still converges.
func TestAPOLLODeterminism(t *testing.T) {
	run := func(seed uint64, steps int) (*tensor.Tensor, float64) {
		w, loss, grad := apolloQuadratic(5, 7)
		opt := NewAPOLLO([]*tensor.Tensor{w}, 0.1, seed, WithAPOLLORank(2), WithAPOLLOGap(25))
		for range steps {
			if err := opt.Step(grad); err != nil {
				t.Fatal(err)
			}
		}
		return w, loss()
	}

	a, _ := run(11, 60)
	b, _ := run(11, 60)
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatalf("same seed diverged at %v: %v != %v", idx, a.AtF64(idx...), b.AtF64(idx...))
		}
	}

	c, _ := run(12, 60)
	same := true
	for i := 0; i < a.Numel() && same; i++ {
		idx := tensor.Unravel(i, a.Shape())
		same = a.AtF64(idx...) == c.AtF64(idx...)
	}
	if same {
		t.Error("different seeds produced identical trajectories — projection not seed-driven")
	}
	if _, lf := run(12, 2000); lf > 0.1 {
		t.Errorf("different seed did not converge: final loss %.4f", lf)
	}
}

// APOLLO-Mini (rank-1 sketch, single tensor-wise scaling, α=√128) still converges
// on the quadratic — a looser bar than full APOLLO, matching its extreme
// memory/quality trade (arXiv:2412.05270 §4.2). Its moment state is 2 floats + the
// rank-1 sketch: SGD-territory.
func TestAPOLLOMiniConverges(t *testing.T) {
	w, loss, grad := apolloQuadratic(5, 7)
	// α=√128 multiplies the step, so the lr is scaled down accordingly.
	opt := NewAPOLLO([]*tensor.Tensor{w}, 0.01, 5, WithAPOLLOMini())
	l0 := loss()
	for range 2000 {
		if err := opt.Step(grad); err != nil {
			t.Fatal(err)
		}
	}
	if lf := loss(); lf > 0.1*l0 {
		t.Errorf("APOLLO-Mini did not converge: loss %.4f → %.4f (want < 10%% of initial)", l0, lf)
	}
	if n := len(opt.st[0].m); n != 7 {
		t.Errorf("Mini moment state %d floats, want 7 = 1*max(m,n)", n)
	}
}

// Non-matrix parameters (bias vectors etc.) bypass the projection and get plain
// Adam, exactly like GaLore — the quadratic on a 1-D parameter reaches Adam's floor.
func TestAPOLLONonMatrixAdam(t *testing.T) {
	star := []float64{1.5, -2, 0.5, 3}
	b := tensor.New(tensor.F32, tensor.Shape{4})
	opt := NewAPOLLO([]*tensor.Tensor{b}, 0.05, 9)
	for range 2000 {
		g := tensor.New(tensor.F32, b.Shape())
		for i := range star {
			g.SetF64(b.AtF64(i)-star[i], i)
		}
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
	}
	for i := range star {
		if d := b.AtF64(i) - star[i]; d > 0.05 || d < -0.05 {
			t.Errorf("bias[%d] = %.4f, want ≈ %.4f (plain-Adam path)", i, b.AtF64(i), star[i])
		}
	}
}

// A mismatched gradient shape is rejected, mirroring every other optimizer here.
func TestAPOLLOShapeMismatch(t *testing.T) {
	w := tensor.New(tensor.F32, tensor.Shape{3, 4})
	g := tensor.New(tensor.F32, tensor.Shape{4, 3})
	opt := NewAPOLLO([]*tensor.Tensor{w}, 1e-3, 1)
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err == nil {
		t.Error("expected shape-mismatch error, got nil")
	}
}

// APOLLO keeps its Adam moments in a rank-r RANDOM sketch — no SVD — while the
// weight and its update stay full-rank: for a 1024×1024 layer at rank 64 the moment
// state shrinks from 1024·1024 to 64·1024 floats — 16× smaller — and APOLLO-Mini
// (rank 1) would need just 1024. P is regenerated from a stored seed instead of
// being retained (Zhu, Zhang, Hao et al. 2024, arXiv:2412.05270).
func ExampleAPOLLO() {
	w := tensor.New(tensor.F32, tensor.Shape{1024, 1024})
	opt := NewAPOLLO([]*tensor.Tensor{w}, 1e-3, 42, WithAPOLLORank(64))
	g := tensor.New(tensor.F32, w.Shape()) // zero gradient: a well-defined no-op step
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
		panic(err)
	}
	fmt.Println(1024*1024/(64*1024), "x smaller")
	// Output: 16 x smaller
}
