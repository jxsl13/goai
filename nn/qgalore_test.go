package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// qgQuadratic builds the shared test problem: minimize ½‖W−W*‖² over a [rows,cols]
// matrix from a seeded random start. Returns the parameter, the loss closure and the
// grad closure (grad = W − W*, recomputed from the live W). Same shape as
// TestGaLoreConvergesQuadratic for direct comparability.
func qgQuadratic(rows, cols int) (*tensor.Tensor, func() float64, GradFn) {
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

// §V16 anchor 1 — COLLAPSE: with quantization disabled (QuantBits 0), Q-GaLore is
// bit-for-bit nn.GaLore over a full multi-step run at the same rank/scale/gap/betas.
// This pins that the ONLY thing Q-GaLore changes is the precision of the stored
// moments. Both the full-rank and low-rank projection sides are covered.
func TestQGaLoreCollapsesToGaLore(t *testing.T) {
	for _, dims := range []struct{ rows, cols, rank int }{
		{5, 7, 3}, // rows<=cols → left projection, low rank
		{7, 5, 3}, // rows>cols  → right projection, low rank
		{5, 7, 5}, // full-rank projection
	} {
		wq := tensor.FromFloat64(tensor.Shape{dims.rows, dims.cols}, seededInit(dims.rows*dims.cols))
		wg := tensor.FromFloat64(tensor.Shape{dims.rows, dims.cols}, seededInit(dims.rows*dims.cols))

		q := NewQGaLore([]*tensor.Tensor{wq}, 0.05,
			WithQGaLoreRank(dims.rank), WithQGaLoreScale(0.7), WithQGaLoreGap(13),
			WithQGaLoreQuantBits(0)) // quantization OFF → must equal GaLore exactly
		g := NewGaLore([]*tensor.Tensor{wg}, 0.05,
			WithGaLoreRank(dims.rank), WithGaLoreScale(0.7), WithGaLoreGap(13))

		gradFor := func(w *tensor.Tensor) GradFn {
			return func(*tensor.Tensor) *tensor.Tensor {
				gt := tensor.New(tensor.F32, w.Shape())
				for i := range w.Numel() {
					idx := tensor.Unravel(i, w.Shape())
					// a deterministic non-trivial gradient field
					gt.SetF64(math.Sin(w.AtF64(idx...))+0.1*float64(i), idx...)
				}
				return gt
			}
		}

		for step := 0; step < 40; step++ {
			if err := q.Step(gradFor(wq)); err != nil {
				t.Fatal(err)
			}
			if err := g.Step(gradFor(wg)); err != nil {
				t.Fatal(err)
			}
		}
		for i := range wq.Numel() {
			idx := tensor.Unravel(i, wq.Shape())
			if a, b := wq.AtF64(idx...), wg.AtF64(idx...); a != b {
				t.Fatalf("dims %v: quant-off Q-GaLore diverged from GaLore at %v: %v != %v (collapse must be bit-identical)",
					dims, idx, a, b)
			}
		}
	}
}

// seededInit returns n deterministic pseudo-random floats (shared start for the
// collapse test's paired optimizers).
func seededInit(n int) []float64 {
	rng := rand.New(rand.NewPCG(99, 100))
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.NormFloat64()
	}
	return out
}

// §V16 anchor 2 — MEMORY: GaLore keeps its subspace moments as r·max(m,n) fp64 floats
// each for m and v (8 bytes/elem); Q-GaLore's INT8 version is 1 byte/elem plus one
// fp64 scale per BlockSize block. Assert the exact byte counts and the ~8× reduction
// ratio — the defining property of the optimizer.
func TestQGaLoreStateMemory(t *testing.T) {
	for _, dims := range []struct{ rows, cols, rank, bs int }{
		{512, 256, 32, 256},
		{256, 512, 32, 256},
		{128, 128, 16, 64},
	} {
		red := dims.rank * max(dims.rows, dims.cols) // reduced moment length
		nb := numBlocks(red, dims.bs)

		mk := func(bits int) *QGaLore {
			w := tensor.New(tensor.F32, tensor.Shape{dims.rows, dims.cols})
			o := NewQGaLore([]*tensor.Tensor{w}, 1e-3,
				WithQGaLoreRank(dims.rank), WithQGaLoreBlockSize(dims.bs), WithQGaLoreQuantBits(bits))
			g := tensor.New(tensor.F32, w.Shape())
			g.SetF64(1, 0, 0) // non-degenerate gradient so the subspace is well-defined
			if err := o.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
				t.Fatal(err)
			}
			return o
		}

		quant := mk(8)
		full := mk(0)

		wantFull := 2 * red * 8       // m + v, fp64
		wantQuant := 2*red*1 + 2*nb*8 // m + v INT8 codes, plus their block scales
		if got := full.StateBytes(); got != wantFull {
			t.Errorf("dims %v: full-precision state %d bytes, want %d", dims, got, wantFull)
		}
		if got := quant.StateBytes(); got != wantQuant {
			t.Errorf("dims %v: INT8 state %d bytes, want %d", dims, got, wantQuant)
		}
		ratio := float64(wantFull) / float64(wantQuant)
		if ratio < 7 || ratio > 8 {
			t.Errorf("dims %v: reduction ratio %.2f×, want in (7,8] (≈8× minus scale overhead)", dims, ratio)
		}
	}
}

// §V16 anchor 3a — QUANTIZATION accuracy: the block-wise non-linear INT8 round-trip
// reconstructs every element (down to absmax·2^-qgLogRange) within a bounded RELATIVE
// error of the original — the log-grid bound ≈ 2^(qgLogRange/252)−1 ≈ 5.7%. This
// relative bound (not an absolute one) is what keeps a near-zero second moment from
// collapsing to 0. Checked across blocks with mixed magnitudes (so the per-block absmax
// scales actually differ) and an all-zero block (scale 0, exact).
func TestQGaLoreQuantRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	bs := 8
	x := make([]float64, 3*bs)
	for i := range x {
		switch i / bs {
		case 0:
			x[i] = rng.NormFloat64() // ~O(1)
		case 1:
			x[i] = 1000 * rng.NormFloat64() // large-magnitude block
		default:
			x[i] = 0 // all-zero block
		}
	}
	codes := make([]int8, len(x))
	scales := make([]float64, numBlocks(len(x), bs))
	quantizeBlockwise(x, bs, codes, scales)
	out := make([]float64, len(x))
	dequantizeBlockwise(codes, scales, bs, out)

	const relBound = 0.057 // 2^(qgLogRange/252)−1, the log-grid relative-error bound
	for i := range x {
		absmax := scales[i/bs]
		// elements above the grid floor: bounded RELATIVE error.
		if absmax > 0 && math.Abs(x[i]) >= absmax*math.Exp2(-qgLogRange) {
			if err := math.Abs(out[i] - x[i]); err > relBound*math.Abs(x[i])+1e-12 {
				t.Errorf("x[%d]=%.6g dequant=%.6g relerr=%.4g > %.4g", i, x[i], out[i], err/math.Abs(x[i]), relBound)
			}
		}
	}
	// all-zero block: scale 0, exact reconstruction.
	for i := 2 * bs; i < 3*bs; i++ {
		if out[i] != 0 {
			t.Errorf("all-zero block element %d reconstructed as %v, want 0", i, out[i])
		}
	}
}

// §V16 anchor 3b — convergence: quantized Q-GaLore (INT8 moments) converges on the
// quadratic and lands within a small factor of quant-OFF GaLore's floor on the SAME
// problem/budget. Quantizing the states must not wreck the trajectory.
func TestQGaLoreQuantizedConverges(t *testing.T) {
	const rows, cols, steps, lr = 5, 7, 3000, 0.05

	wq, lossQ, gradQ := qgQuadratic(rows, cols)
	q := NewQGaLore([]*tensor.Tensor{wq}, lr, WithQGaLoreRank(rows), WithQGaLoreScale(1.0))
	l0 := lossQ()
	for range steps {
		if err := q.Step(gradQ); err != nil {
			t.Fatal(err)
		}
	}
	lf := lossQ()
	if lf > 0.1*l0 {
		t.Errorf("quantized Q-GaLore did not converge: loss %.4f → %.4f (want < 10%% of initial)", l0, lf)
	}

	// GaLore baseline (== Q-GaLore with quant off) on the identical problem.
	wg, lossG, gradG := qgQuadratic(rows, cols)
	g := NewGaLore([]*tensor.Tensor{wg}, lr, WithGaLoreRank(rows), WithGaLoreScale(1.0))
	for range steps {
		if err := g.Step(gradG); err != nil {
			t.Fatal(err)
		}
	}
	lg := lossG()
	if lf > 100*lg+1e-3 {
		t.Errorf("quantized floor %.6g not close to GaLore floor %.6g (want within 100×)", lf, lg)
	}
}

// Determinism: same params + config replay a bit-identical trajectory (no hidden
// randomness — the projection is SVD-derived, the quantization deterministic).
func TestQGaLoreDeterminism(t *testing.T) {
	run := func() *tensor.Tensor {
		w, _, grad := qgQuadratic(5, 7)
		o := NewQGaLore([]*tensor.Tensor{w}, 0.05, WithQGaLoreRank(3), WithQGaLoreGap(9))
		for range 50 {
			if err := o.Step(grad); err != nil {
				t.Fatal(err)
			}
		}
		return w
	}
	a, b := run(), run()
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatalf("same config diverged at %v: %v != %v", idx, a.AtF64(idx...), b.AtF64(idx...))
		}
	}
}

// Non-matrix parameters (bias vectors) bypass the projection AND the quantization and
// get plain fp Adam, exactly like GaLore — the quadratic on a 1-D parameter reaches
// Adam's floor, and no INT8 state is allocated for it.
func TestQGaLoreNonMatrixAdam(t *testing.T) {
	star := []float64{1.5, -2, 0.5, 3}
	bp := tensor.New(tensor.F32, tensor.Shape{4})
	opt := NewQGaLore([]*tensor.Tensor{bp}, 0.05)
	for range 3000 {
		g := tensor.New(tensor.F32, bp.Shape())
		for i := range star {
			g.SetF64(bp.AtF64(i)-star[i], i)
		}
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
	}
	for i := range star {
		if d := bp.AtF64(i) - star[i]; d > 0.05 || d < -0.05 {
			t.Errorf("bias[%d] = %.4f, want ≈ %.4f (plain-Adam path)", i, bp.AtF64(i), star[i])
		}
	}
	if st := opt.st[0]; st.mq != nil {
		t.Error("non-matrix parameter got INT8 state; should stay fp")
	}
}

// A nil gradient skips the parameter (no state allocated, no update), and a mismatched
// gradient shape is rejected — mirroring every other optimizer here.
func TestQGaLoreNilAndShapeGuards(t *testing.T) {
	w := tensor.New(tensor.F32, tensor.Shape{3, 4})
	opt := NewQGaLore([]*tensor.Tensor{w}, 1e-3)
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatalf("nil grad should be skipped, got %v", err)
	}
	if opt.st[0] != nil {
		t.Error("nil-grad parameter allocated state")
	}
	bad := tensor.New(tensor.F32, tensor.Shape{4, 3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("expected shape-mismatch error, got nil")
	}
}

// f32 and f64 parameters both train through the quantized path (the optimizer works in
// float64 internally and writes back via SetF64 regardless of dtype).
func TestQGaLoreDtypes(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		star := seededInit(30)
		w := tensor.New(dt, tensor.Shape{5, 6})
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			w.SetF64(star[i]+1, idx...) // start offset from the target
		}
		opt := NewQGaLore([]*tensor.Tensor{w}, 0.05, WithQGaLoreRank(5), WithQGaLoreScale(1.0))
		for range 2000 {
			g := tensor.New(dt, w.Shape())
			for i := range star {
				idx := tensor.Unravel(i, w.Shape())
				g.SetF64(w.AtF64(idx...)-star[i], idx...)
			}
			if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
				t.Fatal(err)
			}
		}
		var s float64
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			d := w.AtF64(idx...) - star[i]
			s += d * d
		}
		if s > 0.1 {
			t.Errorf("dtype %v: did not converge, ‖W−W*‖² = %.4f", dt, s)
		}
	}
}

// Q-GaLore stores GaLore's subspace Adam moments in INT8 instead of fp64, cutting the
// optimizer state ~8× further: for a 512×256 layer at rank 32 the moment vectors drop
// from 8 bytes/element to 1 (plus a small per-block scale), while the projection and
// the full-rank weights are unchanged from GaLore (Zhang et al. 2024, arXiv:2407.08296).
func ExampleQGaLore() {
	newOpt := func(opts ...QGaLoreOption) *QGaLore {
		w := tensor.New(tensor.F32, tensor.Shape{512, 256})
		o := NewQGaLore([]*tensor.Tensor{w}, 1e-3,
			append([]QGaLoreOption{WithQGaLoreRank(32)}, opts...)...)
		g := tensor.New(tensor.F32, w.Shape())
		if err := o.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			panic(err)
		}
		return o
	}
	quant := newOpt()                       // INT8 moments (default)
	full := newOpt(WithQGaLoreQuantBits(0)) // full precision ≡ GaLore
	fmt.Printf("%.1fx smaller optimizer state\n", float64(full.StateBytes())/float64(quant.StateBytes()))
	// Output: 7.8x smaller optimizer state
}
