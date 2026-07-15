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

// kurtosis returns the excess-free fourth standardized moment E[(x-μ)^4]/σ^4 over all entries
// (≈3 for a Gaussian, ≫3 for outlier-heavy data).
func kurtosis(x *tensor.Tensor) float64 {
	rows, cols := x.Shape()[0], x.Shape()[1]
	n := float64(rows * cols)
	var mean float64
	for i := range rows {
		for j := range cols {
			mean += x.AtF64(i, j)
		}
	}
	mean /= n
	var m2, m4 float64
	for i := range rows {
		for j := range cols {
			d := x.AtF64(i, j) - mean
			m2 += d * d
			m4 += d * d * d * d
		}
	}
	m2 /= n
	m4 /= n
	return m4 / (m2 * m2)
}

func mse(a, b *tensor.Tensor) float64 {
	rows, cols := a.Shape()[0], a.Shape()[1]
	var s float64
	for i := range rows {
		for j := range cols {
			d := a.AtF64(i, j) - b.AtF64(i, j)
			s += d * d
		}
	}
	return s / float64(rows*cols)
}

// §V16(1) COMPUTATIONAL-INVARIANCE anchor: inserting R then Rᵀ around a linear y = x·W leaves the
// FULL-PRECISION output unchanged — (x·R)·(Rᵀ·W) = x·W to machine precision — for BOTH rotation
// families. The rotation is invisible without quantization.
func TestSpinQuantComputationalInvariance(t *testing.T) {
	const dim, out = 8, 5
	rng := rand.New(rand.NewPCG(1, 2))
	x := tensor.New(tensor.F64, tensor.Shape{4, dim})
	for i := range 4 {
		for j := range dim {
			x.SetF64(rng.NormFloat64(), i, j)
		}
	}
	w := tensor.New(tensor.F64, tensor.Shape{dim, out})
	for i := range dim {
		for j := range out {
			w.SetF64(rng.NormFloat64(), i, j)
		}
	}
	ref := matmul(t, x, w) // x·W, the un-rotated reference output

	build := map[string]func() (*nn.SpinRotation, error){
		"hadamard":  func() (*nn.SpinRotation, error) { return nn.NewHadamardRotation(tensor.F64, dim) },
		"learnable": func() (*nn.SpinRotation, error) { return nn.NewLearnableRotation(tensor.F64, dim, nn.WithSpinSeed(9)) },
	}
	for name, mk := range build {
		s, err := mk()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		ctx := backend.NewContext()
		xr, err := s.Apply(ctx, x) // rotated activations x·R
		if err != nil {
			t.Fatalf("%s Apply: %v", name, err)
		}
		wf, err := s.FoldWeight(ctx, w) // rotation folded into weight Rᵀ·W
		if err != nil {
			t.Fatalf("%s FoldWeight: %v", name, err)
		}
		got := matmul(t, xr, wf) // (x·R)·(Rᵀ·W)
		if d := maxAbsDiff(got, ref); d > 1e-10 {
			t.Errorf("%s: rotation changed the FP output, max|Δ| = %.3g > 1e-10", name, d)
		}
	}
}

// §V16(3) orthogonality: RᵀR = I to machine precision for both rotation families.
func TestSpinQuantOrthogonal(t *testing.T) {
	const dim = 16
	for _, tc := range []struct {
		name string
		mk   func() (*nn.SpinRotation, error)
	}{
		{"hadamard", func() (*nn.SpinRotation, error) { return nn.NewHadamardRotation(tensor.F64, dim, nn.WithSpinSeed(3)) }},
		{"learnable", func() (*nn.SpinRotation, error) { return nn.NewLearnableRotation(tensor.F64, dim, nn.WithSpinSeed(4)) }},
	} {
		s, err := tc.mk()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		r, err := s.Matrix(backend.NewContext())
		if err != nil {
			t.Fatalf("%s Matrix: %v", tc.name, err)
		}
		rt, err := backend.Execute(backend.NewContext(), backend.OpTranspose, []*tensor.Tensor{r}, nil)
		if err != nil {
			t.Fatal(err)
		}
		gram := matmul(t, rt[0], r) // RᵀR
		for i := range dim {
			for j := range dim {
				want := 0.0
				if i == j {
					want = 1
				}
				if d := math.Abs(gram.AtF64(i, j) - want); d > 1e-10 {
					t.Fatalf("%s: (RᵀR)[%d,%d] = %.3g, want %v (not orthogonal)", tc.name, i, j, gram.AtF64(i, j), want)
				}
			}
		}
	}
}

// §V16(2) OUTLIER-REDUCTION value: on injected-outlier activations the rotated version has a
// smaller max-abs and lower kurtosis, and rotate→quant→dequant→unrotate MSE is below directly
// quantizing the un-rotated activations at the same bit-width.
func TestSpinQuantOutlierReduction(t *testing.T) {
	const dim, rows, levels = 32, 8, 16 // 4-bit uniform grid
	rng := rand.New(rand.NewPCG(5, 6))
	x := tensor.New(tensor.F64, tensor.Shape{rows, dim})
	for i := range rows {
		for j := range dim {
			x.SetF64(rng.NormFloat64(), i, j)
		}
	}
	// inject a huge outlier into one column of a few rows (the LLM activation pathology).
	for _, i := range []int{0, 3, 6} {
		x.SetF64(50, i, (i*7)%dim)
	}

	s, err := nn.NewHadamardRotation(tensor.F64, dim, nn.WithSpinSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	rot, err := s.Apply(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	origMax, rotMax := maxAbs(x), maxAbs(rot)
	if rotMax >= origMax {
		t.Errorf("rotation did not shrink the max-abs: rotated %.3g ≥ original %.3g", rotMax, origMax)
	}
	origK, rotK := kurtosis(x), kurtosis(rot)
	if rotK >= origK {
		t.Errorf("rotation did not lower kurtosis: rotated %.3g ≥ original %.3g", rotK, origK)
	}

	// direct fake-quant of the UN-rotated activations, same symmetric per-tensor grid.
	qd := nn.UniformQuantizer(levels, -origMax, origMax)
	direct := tensor.New(tensor.F64, tensor.Shape{rows, dim})
	for i := range rows {
		for j := range dim {
			direct.SetF64(qd(x.AtF64(i, j)), i, j)
		}
	}
	spin, err := s.QuantizeActivations(x, levels)
	if err != nil {
		t.Fatal(err)
	}
	mseDirect, mseSpin := mse(direct, x), mse(spin, x)
	if mseSpin >= mseDirect {
		t.Errorf("rotate-then-quant did not beat direct quant: spin MSE %.4g ≥ direct MSE %.4g", mseSpin, mseDirect)
	}
	t.Logf("max-abs %.3g→%.3g  kurtosis %.3g→%.3g  MSE direct %.4g → spin %.4g",
		origMax, rotMax, origK, rotK, mseDirect, mseSpin)
}

// §V16(3) gradcheck: the learnable rotation R = qr(W).Q is differentiable end-to-end; the
// gradient of a scalar loss over the rotated activations w.r.t. W matches finite differences.
func TestSpinQuantLearnableGradcheck(t *testing.T) {
	const dim, rows = 3, 2
	newRot := func() *nn.SpinRotation {
		s, err := nn.NewLearnableRotation(tensor.F64, dim, nn.WithSpinSeed(7))
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	x := tensor.FromFloat64(tensor.Shape{rows, dim}, []float64{1, -0.5, 2, 0.3, -1.2, 0.8})
	mask := tensor.FromFloat64(tensor.Shape{rows, dim}, []float64{0.7, -0.4, 0.9, 0.2, 1.1, -0.6})

	lossOf := func(s *nn.SpinRotation) float64 {
		out, err := s.Apply(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range rows {
			for j := range dim {
				acc += mask.AtF64(i, j) * out.AtF64(i, j)
			}
		}
		return acc
	}

	s := newRot()
	tape := autograd.NewTape()
	rot, err := s.Apply(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	prod, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{rot, mask}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{prod[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	w := s.Params()[0]
	g := tape.Grad(w)
	if g == nil {
		t.Fatal("nil gradient w.r.t. the learnable rotation matrix")
	}
	const eps = 1e-6
	for i := range dim {
		for j := range dim {
			orig := w.AtF64(i, j)
			w.SetF64(orig+eps, i, j)
			lp := lossOf(s)
			w.SetF64(orig-eps, i, j)
			lm := lossOf(s)
			w.SetF64(orig, i, j)
			fd := (lp - lm) / (2 * eps)
			if an := g.AtF64(i, j); math.Abs(an-fd) > 1e-4*math.Max(1, math.Abs(fd)) {
				t.Errorf("grad[%d,%d]: analytic %.8g vs finite-diff %.8g", i, j, an, fd)
			}
		}
	}
}

// §V16(4) determinism + sanity: the same seed reproduces the same rotation; the Hadamard variant
// carries no parameters while the learnable one does; and the constructors reject bad shapes.
func TestSpinQuantDeterminismAndErrors(t *testing.T) {
	a, err := nn.NewHadamardRotation(tensor.F64, 8, nn.WithSpinSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	b, err := nn.NewHadamardRotation(tensor.F64, 8, nn.WithSpinSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	ra, _ := a.Matrix(backend.NewContext())
	rb, _ := b.Matrix(backend.NewContext())
	if d := maxAbsDiff(ra, rb); d != 0 {
		t.Errorf("same seed produced different Hadamard rotations, max|Δ| = %g", d)
	}
	if a.Dim() != 8 {
		t.Errorf("Dim = %d, want 8", a.Dim())
	}
	if len(a.Params()) != 0 {
		t.Errorf("Hadamard rotation reports %d params, want 0", len(a.Params()))
	}
	l, err := nn.NewLearnableRotation(tensor.F64, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Params()) != 1 {
		t.Errorf("learnable rotation reports %d params, want 1", len(l.Params()))
	}

	if _, err := nn.NewHadamardRotation(tensor.F64, 6); err == nil {
		t.Error("expected an error for a non-power-of-two Hadamard dim")
	}
	if _, err := nn.NewHadamardRotation(tensor.F64, 0); err == nil {
		t.Error("expected an error for dim < 1")
	}
	if _, err := nn.NewLearnableRotation(tensor.F64, 0); err == nil {
		t.Error("expected an error for dim < 1")
	}
	if _, err := l.QuantizeActivations(tensor.New(tensor.F64, tensor.Shape{2, 5}), 1); err == nil {
		t.Error("expected an error for levels < 2")
	}
}

// ExampleSpinRotation demonstrates the SpinQuant flow: the rotation is invisible at full
// precision (computational invariance) yet spreads outliers so low-bit quantization improves.
func ExampleSpinRotation() {
	const dim = 4
	// A tiny linear y = x·W and an activation row carrying one big outlier.
	x := tensor.FromFloat64(tensor.Shape{1, dim}, []float64{0.2, 8, -0.3, 0.1})
	w := tensor.FromFloat64(tensor.Shape{dim, 2}, []float64{
		0.5, -0.2,
		0.1, 0.3,
		-0.4, 0.6,
		0.2, -0.1,
	})
	ctx := backend.NewContext()
	ref, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, w}, nil)

	had, _ := nn.NewHadamardRotation(tensor.F64, dim, nn.WithSpinSeed(1))
	learn, _ := nn.NewLearnableRotation(tensor.F64, dim, nn.WithSpinSeed(1))
	rotations := []*nn.SpinRotation{had, learn}

	// Computational invariance: (x·R)·(Rᵀ·W) == x·W for every rotation.
	invariant := true
	for _, s := range rotations {
		xr, _ := s.Apply(ctx, x)
		wf, _ := s.FoldWeight(ctx, w)
		got, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{xr, wf}, nil)
		if math.Abs(got[0].AtF64(0, 0)-ref[0].AtF64(0, 0)) > 1e-9 {
			invariant = false
		}
	}

	// Outlier reduction: the Hadamard rotation shrinks the activation's max magnitude.
	r, _ := had.Matrix(ctx)
	rot, _ := had.Apply(ctx, x)
	var before, after float64
	for j := range dim {
		before = math.Max(before, math.Abs(x.AtF64(0, j)))
		after = math.Max(after, math.Abs(rot.AtF64(0, j)))
	}
	deq, _ := had.QuantizeActivations(x, 16)

	fmt.Println("dim:", had.Dim())
	fmt.Println("rotation is square:", r.Shape()[0] == dim && r.Shape()[1] == dim)
	fmt.Println("hadamard params:", len(had.Params()))
	fmt.Println("learnable params:", len(learn.Params()))
	fmt.Println("full-precision invariance:", invariant)
	fmt.Println("max-abs shrinks:", after < before)
	fmt.Println("dequant shape:", deq.Shape())

	// Output:
	// dim: 4
	// rotation is square: true
	// hadamard params: 0
	// learnable params: 1
	// full-precision invariance: true
	// max-abs shrinks: true
	// dequant shape: (1, 4)
}
