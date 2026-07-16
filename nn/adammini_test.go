package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 collapse check (THE correctness anchor): with block size 1 every element
// is its own block, mean(g²) over a block is exactly g², and Adam-mini's recurrence is
// Adam's — the trajectories of the REAL nn.Adam and nn.AdamMini must coincide at rtol
// 1e-12 over a multi-step run (2-D + 1-D params, weight decay on). If this drifts, the
// block-v math is wrong.
func TestAdamMiniBlockSizeOneCollapsesToAdam(t *testing.T) {
	const lr, wd, steps = 0.05, 0.01, 20
	shapes := []tensor.Shape{{3, 4}, {5}}
	pval := func(i int) float64 { return math.Sin(float64(i)*1.3) - 0.2 }
	gval := func(i, s int) float64 { return math.Cos(float64(i)*0.7+float64(s)*0.31) * 0.4 }

	mk := func() []*tensor.Tensor {
		ps := make([]*tensor.Tensor, len(shapes))
		for pi, s := range shapes {
			ps[pi] = tensor.New(tensor.F64, s)
			for i := range s.Numel() {
				ps[pi].SetF64(pval(i+13*pi), tensor.Unravel(i, s)...)
			}
		}
		return ps
	}
	pAdam, pMini := mk(), mk()
	optAdam := nn.NewAdamW(pAdam, lr, wd)
	optMini := nn.NewAdamMini(pMini, lr,
		nn.WithAdamMiniWeightDecay(wd),
		nn.WithAdamMiniBlockSizes([]int{1, 1})) // every element its own block == Adam

	if got, want := optMini.SecondMomentSize(), 12+5; got != want {
		t.Fatalf("block size 1 must keep one scalar per element: %d != %d", got, want)
	}

	gradFor := func(ps []*tensor.Tensor, s int) nn.GradFn {
		return func(q *tensor.Tensor) *tensor.Tensor {
			for pi, p := range ps {
				if q != p {
					continue
				}
				g := tensor.New(tensor.F64, p.Shape())
				for i := range p.Numel() {
					g.SetF64(gval(i+13*pi, s), tensor.Unravel(i, p.Shape())...)
				}
				return g
			}
			return nil
		}
	}
	for s := range steps {
		if err := optAdam.Step(gradFor(pAdam, s)); err != nil {
			t.Fatal(err)
		}
		if err := optMini.Step(gradFor(pMini, s)); err != nil {
			t.Fatal(err)
		}
	}
	for pi := range shapes {
		for i := range shapes[pi].Numel() {
			idx := tensor.Unravel(i, shapes[pi])
			a, m := pAdam[pi].AtF64(idx...), pMini[pi].AtF64(idx...)
			if math.Abs(a-m) > 1e-12*math.Max(1, math.Abs(a)) {
				t.Errorf("param %d idx %v: AdamMini %.16g != Adam %.16g", pi, idx, m, a)
			}
		}
	}
}

// §V16 tier-1, THE value test: the second-moment state is O(#blocks), not O(#params).
// For a [512,512] weight Adam stores 262144 v-scalars; Adam-mini stores 512 (one per
// row). Across a representative transformer-shaped parameter list the v-memory shrinks
// by >100x and the TOTAL optimizer state (m + v) lands at ~50% of Adam's — the paper's
// headline saving (arXiv:2406.16793 §1: ~45-50% less optimizer memory).
func TestAdamMiniMemoryReduction(t *testing.T) {
	w := tensor.New(tensor.F64, tensor.Shape{512, 512})
	opt := nn.NewAdamMini([]*tensor.Tensor{w}, 1e-3)
	if got := opt.SecondMomentSize(); got != 512 {
		t.Fatalf("[512,512] must keep one v-scalar per row: got %d want 512", got)
	}
	if adamV := w.Numel(); adamV != 262144 {
		t.Fatalf("premise: Adam keeps %d v-scalars for [512,512]", adamV)
	}

	// Representative small-transformer parameter list: 4 attention projections, an FFN
	// pair, an embedding, and three 1-D norm/bias vectors.
	model := []*tensor.Tensor{
		tensor.New(tensor.F64, tensor.Shape{512, 512}),  // Wq
		tensor.New(tensor.F64, tensor.Shape{512, 512}),  // Wk
		tensor.New(tensor.F64, tensor.Shape{512, 512}),  // Wv
		tensor.New(tensor.F64, tensor.Shape{512, 512}),  // Wo
		tensor.New(tensor.F64, tensor.Shape{2048, 512}), // FFN up
		tensor.New(tensor.F64, tensor.Shape{512, 2048}), // FFN down
		tensor.New(tensor.F64, tensor.Shape{1000, 512}), // embedding
		tensor.New(tensor.F64, tensor.Shape{512}),       // norm gain
		tensor.New(tensor.F64, tensor.Shape{512}),       // norm gain
		tensor.New(tensor.F64, tensor.Shape{2048}),      // FFN bias
	}
	adamV := 0
	for _, p := range model {
		adamV += p.Numel()
	}
	optM := nn.NewAdamMini(model, 1e-3)
	miniV := optM.SecondMomentSize()
	wantMini := 512*4 + 2048 + 512 + 1000 + 3 // per-row blocks + one block per 1-D param
	if miniV != wantMini {
		t.Fatalf("block-v size: got %d want %d", miniV, wantMini)
	}
	vRatio := float64(adamV) / float64(miniV)
	adamState := 2 * adamV     // m + v, both per-element
	miniState := adamV + miniV // m per-element + v per-block
	saving := 1 - float64(miniState)/float64(adamState)
	t.Logf("v-scalars: Adam=%d AdamMini=%d (%.0fx smaller); total state: Adam=%d AdamMini=%d (%.1f%% saved)",
		adamV, miniV, vRatio, adamState, miniState, 100*saving)
	if vRatio < 100 {
		t.Errorf("v-memory must shrink >100x on a transformer-shaped model: got %.1fx", vRatio)
	}
	if saving < 0.45 {
		t.Errorf("total optimizer-state saving must be ~50%% (paper's headline): got %.1f%%", 100*saving)
	}
}

// §V16 tier-1 convergence parity: on the shared noisy-regression fixture, Adam-mini at
// Adam's tuned lr reaches the same noise floor as Adam — here with the HARSHEST possible
// partition (the 1-D heuristic gives the whole weight vector ONE shared v-scalar), so
// comparable loss demonstrates that a block-shared learning rate really is enough.
func TestAdamMiniConvergenceParity(t *testing.T) {
	pr := newRegressionProblem()
	const steps, lr = 2000, 0.01

	pM := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	optM := nn.NewAdamMini([]*tensor.Tensor{pM}, lr) // heuristic: ONE block for the 1-D param
	if got := optM.SecondMomentSize(); got != 1 {
		t.Fatalf("1-D param must get a single block: got %d", got)
	}
	excessM := pr.run(t, optM, pM, steps)

	pA := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	excessA := pr.run(t, nn.NewAdam([]*tensor.Tensor{pA}, lr), pA, steps)

	t.Logf("excess loss after %d steps: AdamMini=%.4g (1 v-scalar)  Adam=%.4g (%d v-scalars)",
		steps, excessM, excessA, pr.dim)
	if excessM > 0.01 {
		t.Errorf("AdamMini must reach the noise floor: excess %.4g > 0.01", excessM)
	}
	if excessM > 5*excessA && excessM > 0.005 {
		t.Errorf("AdamMini (%.4g) not comparable to Adam (%.4g)", excessM, excessA)
	}
}

// §V16 tier-1: the block layout. The shape heuristic gives per-leading-row blocks for
// ≥2-D and one block for 1-D; WithAdamMiniBlockSizes overrides it; a non-dividing block
// size leaves a shorter last block whose v uses the mean over the ACTUAL element count —
// checked against a hand-computed single Adam-mini step.
func TestAdamMiniBlockLayout(t *testing.T) {
	heuristic := []struct {
		shape tensor.Shape
		want  int
	}{
		{tensor.Shape{4, 6}, 4},    // 2-D: one block per row
		{tensor.Shape{2, 3, 4}, 2}, // 3-D: one block per leading index (trailing dims flattened)
		{tensor.Shape{7}, 1},       // 1-D: whole tensor
	}
	for _, c := range heuristic {
		opt := nn.NewAdamMini([]*tensor.Tensor{tensor.New(tensor.F64, c.shape)}, 1e-3)
		if got := opt.SecondMomentSize(); got != c.want {
			t.Errorf("heuristic blocks for %v: got %d want %d", c.shape, got, c.want)
		}
	}

	// Override: [4,6] partitioned at block size 2 → 12 blocks (e.g. per-head with head_dim 2).
	opt := nn.NewAdamMini([]*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{4, 6})}, 1e-3,
		nn.WithAdamMiniBlockSizes([]int{2}))
	if got := opt.SecondMomentSize(); got != 12 {
		t.Errorf("override block size 2 on 24 elements: got %d blocks want 12", got)
	}

	// Non-dividing override: 6 elements at block size 4 → blocks {0..3} and {4,5}. One
	// step from zero state, hand-computed: v_B = (1−β₂)·mean(g² over B), m = (1−β₁)g,
	// p -= lr·(m/(1−β₁))/(√(v_B/(1−β₂)) + ε) = lr·g/(√mean(g²) + ε).
	const lr, eps = 0.1, 1e-8 // β₁/β₂ cancel with bias correction on the first step
	g := []float64{1, -2, 3, -4, 5, -6}
	p := tensor.FromFloat64(tensor.Shape{6}, make([]float64, 6))
	opt = nn.NewAdamMini([]*tensor.Tensor{p}, lr, nn.WithAdamMiniBlockSizes([]int{4}))
	if got := opt.SecondMomentSize(); got != 2 {
		t.Fatalf("6 elements at block size 4: got %d blocks want 2", got)
	}
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(p.Shape(), g)
	}); err != nil {
		t.Fatal(err)
	}
	mean0 := (1.0 + 4 + 9 + 16) / 4 // first block: 4 elements
	mean1 := (25.0 + 36) / 2        // last block: only 2
	for i, gv := range g {
		mean := mean0
		if i >= 4 {
			mean = mean1
		}
		want := -lr * gv / (math.Sqrt(mean) + eps)
		if got := p.AtF64(i); math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Errorf("step 1 p[%d]: got %.16g want %.16g", i, got, want)
		}
	}
}

// §V16 tier-1 fast-path guard (§V22 style): the typed contiguous f64/f32 fast paths
// must produce a BIT-IDENTICAL trajectory to the generic widening-accessor fallback,
// which is forced via non-contiguous transposed views.
func TestAdamMiniFastPathParity(t *testing.T) {
	const steps = 6
	pval := func(i int) float64 { return math.Sin(float64(i)*0.7) + 0.1 }
	gval := func(i, s int) float64 { return math.Cos(float64(i)*0.3+float64(s)) * 0.05 }

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		t.Run(fmt.Sprintf("%v", dt), func(t *testing.T) {
			shapes := []tensor.Shape{{3, 5}, {2, 3, 4}}
			fast := make([]*tensor.Tensor, len(shapes))
			slow := make([]*tensor.Tensor, len(shapes))
			for pi, s := range shapes {
				fast[pi] = tensor.New(dt, s)
				slow[pi] = mustTransposedView(t, dt, s)
				for i := range s.Numel() {
					idx := tensor.Unravel(i, s)
					fast[pi].SetF64(pval(i), idx...)
					slow[pi].SetF64(pval(i), idx...)
				}
			}
			mk := func(ps []*tensor.Tensor) nn.Optimizer {
				return nn.NewAdamMini(ps, 0.05, nn.WithAdamMiniWeightDecay(0.01))
			}
			optFast, optSlow := mk(fast), mk(slow)
			for s := range steps {
				gFast := map[*tensor.Tensor]*tensor.Tensor{}
				gSlow := map[*tensor.Tensor]*tensor.Tensor{}
				for pi, shape := range shapes {
					cg := tensor.New(dt, shape)
					vg := mustTransposedView(t, dt, shape)
					for i := range shape.Numel() {
						idx := tensor.Unravel(i, shape)
						cg.SetF64(gval(i, s), idx...)
						vg.SetF64(gval(i, s), idx...)
					}
					gFast[fast[pi]] = cg
					gSlow[slow[pi]] = vg
				}
				if err := optFast.Step(func(p *tensor.Tensor) *tensor.Tensor { return gFast[p] }); err != nil {
					t.Fatalf("fast step %d: %v", s, err)
				}
				if err := optSlow.Step(func(p *tensor.Tensor) *tensor.Tensor { return gSlow[p] }); err != nil {
					t.Fatalf("slow step %d: %v", s, err)
				}
				for pi, shape := range shapes {
					for i := range shape.Numel() {
						idx := tensor.Unravel(i, shape)
						fv, sv := fast[pi].AtF64(idx...), slow[pi].AtF64(idx...)
						if fv != sv {
							t.Fatalf("step %d param %d idx %v: fast path %.17g != generic path %.17g",
								s, pi, idx, fv, sv)
						}
					}
				}
			}
		})
	}
}

// §V16 sanity: all-zero gradients are a NaN-free no-op, nil gradients are skipped, a
// grad shape mismatch errors like the other optimizers, and an invalid block-size spec
// (wrong count, zero, or larger than the tensor) surfaces as an error from Step.
func TestAdamMiniSanity(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	opt := nn.NewAdamMini([]*tensor.Tensor{p}, 0.1)
	zero := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	for range 3 {
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
			t.Fatal(err)
		}
	}
	if p.AtF64(0) != 1 || p.AtF64(1) != 2 || math.IsNaN(p.AtF64(0)) {
		t.Errorf("zero-grad steps must be a NaN-free no-op: p=[%g %g]", p.AtF64(0), p.AtF64(1))
	}

	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatal(err)
	}
	if p.AtF64(0) != 1 {
		t.Error("nil-grad param must be skipped")
	}

	bad := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("shape mismatch must error")
	}

	for name, mk := range map[string]func() *nn.AdamMini{
		"wrong count": func() *nn.AdamMini {
			return nn.NewAdamMini([]*tensor.Tensor{p}, 0.1, nn.WithAdamMiniBlockSizes([]int{1, 1}))
		},
		"zero size": func() *nn.AdamMini {
			return nn.NewAdamMini([]*tensor.Tensor{p}, 0.1, nn.WithAdamMiniBlockSizes([]int{0}))
		},
		"oversized": func() *nn.AdamMini {
			return nn.NewAdamMini([]*tensor.Tensor{p}, 0.1, nn.WithAdamMiniBlockSizes([]int{3}))
		},
	} {
		if err := mk().Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err == nil {
			t.Errorf("invalid block spec (%s) must error from Step", name)
		}
	}
}

// AdamMini is Adam with one shared second-moment scalar per parameter block instead of
// one per element. Here a [4,3] weight is driven to a target: the heuristic gives one
// block per row, so the optimizer stores 4 v-scalars where Adam would store 12 — and
// still converges.
func ExampleAdamMini() {
	target := []float64{1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12}
	w := tensor.New(tensor.F64, tensor.Shape{4, 3})
	opt := nn.NewAdamMini([]*tensor.Tensor{w}, 0.1)
	for range 500 {
		_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F64, w.Shape())
			for i := range w.Numel() {
				idx := tensor.Unravel(i, w.Shape())
				g.SetF64(w.AtF64(idx...)-target[i], idx...) // ∇½‖w−target‖²
			}
			return g
		})
	}
	maxErr := 0.0
	for i := range w.Numel() {
		idx := tensor.Unravel(i, w.Shape())
		maxErr = math.Max(maxErr, math.Abs(w.AtF64(idx...)-target[i]))
	}
	fmt.Printf("v-scalars: %d (Adam would keep %d)\n", opt.SecondMomentSize(), w.Numel())
	fmt.Printf("converged: %v\n", maxErr < 1e-3)
	// Output:
	// v-scalars: 4 (Adam would keep 12)
	// converged: true
}
