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

// mixMat fills an [r,c] F64 tensor with a deterministic non-trivial pattern.
func mixMat(r, c int, seed float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			t.SetF64(math.Sin(seed+float64(i*c+j)), i, j)
		}
	}
	return t
}

// logitsMat builds [B,K] logits and [B] class-index targets for CE tests.
func logitsMat(b, k int) (*tensor.Tensor, *tensor.Tensor) {
	z := tensor.New(tensor.F64, tensor.Shape{b, k})
	y := tensor.New(tensor.F64, tensor.Shape{b})
	for i := range b {
		for j := range k {
			z.SetF64(0.3*float64(i)-0.2*float64(j)+math.Cos(float64(i*k+j)), i, j)
		}
		y.SetF64(float64(i%k), i)
	}
	return z, y
}

// §V16 anchor 4: Beta(α,α) via the Gamma sampler — λ∈[0,1] always, mean≈0.5 for
// symmetric α, and large α concentrates near 0.5 (tighter spread).
func TestMixupBetaSamplerMeanRange(t *testing.T) {
	for _, alpha := range []float64{0.2, 1.0, 8.0} {
		m := nn.NewMixup(alpha, 42)
		const n = 20000
		var sum, sumsq float64
		for range n {
			_, _, lam, err := m.Forward(mixMat(2, 1, 0))
			if err != nil {
				t.Fatal(err)
			}
			if lam < 0 || lam > 1 {
				t.Fatalf("alpha=%g: lambda %g out of [0,1]", alpha, lam)
			}
			sum += lam
			sumsq += lam * lam
		}
		mean := sum / n
		variance := sumsq/n - mean*mean
		if math.Abs(mean-0.5) > 0.02 {
			t.Errorf("alpha=%g: mean %g, want ≈0.5 (symmetric Beta)", alpha, mean)
		}
		// Beta(α,α) variance = 1/(4(2α+1)); check the sampler tracks it.
		wantVar := 1 / (4 * (2*alpha + 1))
		if math.Abs(variance-wantVar) > 0.02 {
			t.Errorf("alpha=%g: var %g, want ≈%g", alpha, variance, wantVar)
		}
	}
}

// §V16 anchor 1 (COLLAPSE): Mixup with λ=1 ⇒ x'=x EXACTLY (bit-for-bit).
func TestMixupLambdaOneCollapse(t *testing.T) {
	m := nn.NewMixup(0.2, 7, nn.WithMixupLambda(1))
	x := mixMat(6, 4, 1.5)
	out, _, lam, err := m.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if lam != 1 {
		t.Fatalf("lambda %g, want exactly 1", lam)
	}
	for i := range 6 {
		for j := range 4 {
			if out.AtF64(i, j) != x.AtF64(i, j) {
				t.Fatalf("λ=1 not identity at [%d,%d]: %v vs %v", i, j, out.AtF64(i, j), x.AtF64(i, j))
			}
		}
	}
}

// §V16 anchor 3 (INPUT-MIX): x'[i]=λ·x[i]+(1−λ)·x[π[i]] elementwise to 1e-10.
func TestMixupInputMix(t *testing.T) {
	m := nn.NewMixup(0.5, 3, nn.WithMixupLambda(0.3))
	x := mixMat(8, 5, 2.0)
	out, perm, lam, err := m.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 8 {
		for j := range 5 {
			want := lam*x.AtF64(i, j) + (1-lam)*x.AtF64(perm[i], j)
			if math.Abs(out.AtF64(i, j)-want) > 1e-10 {
				t.Errorf("mix[%d,%d]=%v want %v", i, j, out.AtF64(i, j), want)
			}
		}
	}
}

// §V16 anchor 2 (LABEL-MIX LINEARITY): MixupLoss ≡ λ·CE(a)+(1−λ)·CE(b) to 1e-10.
func TestMixupLossLinearity(t *testing.T) {
	ctx := backend.NewContext()
	z, yA := logitsMat(5, 4)
	perm := []int{2, 0, 4, 1, 3}
	yB := nn.PermuteRows(yA, perm)
	const lam = 0.37
	mix, err := nn.MixupLoss(ctx, z, yA, yB, lam)
	if err != nil {
		t.Fatal(err)
	}
	ceA, _ := nn.CrossEntropy(ctx, z, yA)
	ceB, _ := nn.CrossEntropy(ctx, z, yB)
	want := lam*ceA.AtF64() + (1-lam)*ceB.AtF64()
	if math.Abs(mix.AtF64()-want) > 1e-10 {
		t.Errorf("MixupLoss=%v want %v", mix.AtF64(), want)
	}
}

// §V16 anchor 1 (COLLAPSE, label side): λ=1 ⇒ MixupLoss == CrossEntropy(z, yA)
// bit-exactly, regardless of the partner labels.
func TestMixupLossLambdaOneCollapse(t *testing.T) {
	ctx := backend.NewContext()
	z, yA := logitsMat(6, 3)
	yB := nn.PermuteRows(yA, []int{1, 2, 3, 4, 5, 0})
	mix, err := nn.MixupLoss(ctx, z, yA, yB, 1)
	if err != nil {
		t.Fatal(err)
	}
	ce, _ := nn.CrossEntropy(ctx, z, yA)
	if mix.AtF64() != ce.AtF64() {
		t.Errorf("λ=1 loss %v != CrossEntropy %v (not bit-exact)", mix.AtF64(), ce.AtF64())
	}
}

// PermuteRows works for one-hot / soft targets ([B,K]), not just indices.
func TestPermuteRowsOneHot(t *testing.T) {
	oh := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		oh.SetF64(1, i, i) // row i one-hot at class i
	}
	perm := []int{2, 0, 1}
	out := nn.PermuteRows(oh, perm)
	for i := range 3 {
		for j := range 4 {
			if out.AtF64(i, j) != oh.AtF64(perm[i], j) {
				t.Errorf("permuted[%d,%d]=%v want %v", i, j, out.AtF64(i, j), oh.AtF64(perm[i], j))
			}
		}
	}
}

// §V16 anchor 4 (determinism): same seed ⇒ identical λ and permutation.
func TestMixupDeterminism(t *testing.T) {
	x := mixMat(10, 3, 0.5)
	a, pa, la, _ := nn.NewMixup(0.4, 123).Forward(x)
	b, pb, lb, _ := nn.NewMixup(0.4, 123).Forward(x)
	if la != lb {
		t.Fatalf("lambda differs across equal seeds: %v vs %v", la, lb)
	}
	for i := range pa {
		if pa[i] != pb[i] {
			t.Fatalf("perm differs at %d: %d vs %d", i, pa[i], pb[i])
		}
	}
	for i := range 10 {
		if a.AtF64(i, 0) != b.AtF64(i, 0) {
			t.Fatalf("mixed output differs at row %d", i)
		}
	}
}

// MixupLoss is differentiable w.r.t. logits (gradient flows through both CE VJPs).
func TestMixupLossGradient(t *testing.T) {
	tape := autograd.NewTape()
	z, yA := logitsMat(4, 3)
	yB := nn.PermuteRows(yA, []int{3, 2, 1, 0})
	loss, err := nn.MixupLoss(tape.Context(), z, yA, yB, 0.6)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(z)
	if g == nil {
		t.Fatal("no gradient for logits")
	}
	var sum float64
	for i := range 4 {
		for j := range 3 {
			sum += math.Abs(g.AtF64(i, j))
		}
	}
	if sum == 0 {
		t.Error("zero gradient — label path not differentiable")
	}
}

func TestMixupErrors(t *testing.T) {
	if _, _, _, err := nn.NewMixup(0.2, 1).Forward(tensor.New(tensor.F64, tensor.Shape{})); err == nil {
		t.Error("scalar input: want error (no batch axis)")
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("alpha≤0: want panic")
			}
		}()
		nn.NewMixup(0, 1)
	}()
}

func ExampleMixup() {
	// Pin λ=0.75 for a reproducible blend of a 2-sample batch (seed 5 swaps the pair).
	m := nn.NewMixup(0.2, 5, nn.WithMixupLambda(0.75))
	x := tensor.New(tensor.F64, tensor.Shape{2, 1})
	x.SetF64(4, 0, 0)
	x.SetF64(8, 1, 0)
	mixed, perm, lambda, _ := m.Forward(x)
	// row i = λ·x[i] + (1−λ)·x[perm[i]]
	fmt.Printf("lambda=%.2f perm=%v row0=%.2f row1=%.2f\n",
		lambda, perm, mixed.AtF64(0, 0), mixed.AtF64(1, 0))
	// Output:
	// lambda=0.75 perm=[1 0] row0=5.00 row1=7.00
}
