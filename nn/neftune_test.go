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

// §V16 tier-1 (§R96): NEFTune adds Uniform(−mag, mag) noise with mag = α/√(L·d) per
// entry. On a zero embedding the output IS the noise: every entry is within the
// magnitude bound, the mean is ~0 (symmetric) and the std ≈ mag/√3 (uniform).
func TestNEFTuneNoiseFormula(t *testing.T) {
	const L, d = 16, 16
	const alpha = 10.0
	emb := tensor.New(tensor.F64, tensor.Shape{L, d}) // zeros → output is pure noise
	out, err := nn.NEFTune(backend.NewContext(), emb, alpha, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}
	mag := alpha / math.Sqrt(float64(L*d)) // 10/16 = 0.625
	n := L * d
	var mean, sumsq float64
	for i := range n {
		v := out.AtF64(tensor.Unravel(i, out.Shape())...)
		if math.Abs(v) > mag+1e-12 {
			t.Fatalf("noise entry %v exceeds magnitude %v", v, mag)
		}
		mean += v
		sumsq += v * v
	}
	mean /= float64(n)
	std := math.Sqrt(sumsq/float64(n) - mean*mean)
	if math.Abs(mean) > 0.1*mag {
		t.Errorf("noise mean %v not ~0 (mag %v)", mean, mag)
	}
	if want := mag / math.Sqrt(3); math.Abs(std-want) > 0.2*want {
		t.Errorf("noise std %v, want ≈ mag/√3 = %v (uniform)", std, want)
	}
}

// §V16 tier-1 / §V2: the noise is a constant additive perturbation, so gradients flow
// unchanged into the embedding — ∂mean(emb+noise)/∂embᵢ = 1/n.
func TestNEFTuneGradFlows(t *testing.T) {
	emb := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range emb.Numel() {
		emb.SetF64(0.1*float64(i), tensor.Unravel(i, emb.Shape())...)
	}
	tape := autograd.NewTape()
	out, err := nn.NEFTune(tape.Context(), emb, 5, rand.New(rand.NewPCG(3, 4)))
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(tape.Context(), backend.OpMean, []*tensor.Tensor{out}, backend.ReduceAttrs{})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(emb)
	if g == nil {
		t.Fatal("embedding received no gradient through NEFTune")
	}
	want := 1.0 / float64(emb.Numel())
	for i := range emb.Numel() {
		if math.Abs(g.AtF64(tensor.Unravel(i, emb.Shape())...)-want) > 1e-12 {
			t.Fatalf("grad[%d] = %v, want %v", i, g.AtF64(tensor.Unravel(i, emb.Shape())...), want)
		}
	}
}

// §V16 tier-1: NEFTune is deterministic given the rng — the same seed reproduces the
// same noise.
func TestNEFTuneDeterministic(t *testing.T) {
	emb := tensor.New(tensor.F64, tensor.Shape{2, 8})
	a, _ := nn.NEFTune(backend.NewContext(), emb, 7, rand.New(rand.NewPCG(9, 9)))
	b, _ := nn.NEFTune(backend.NewContext(), emb, 7, rand.New(rand.NewPCG(9, 9)))
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatal("NEFTune not deterministic for a fixed rng seed")
		}
	}
}

// NEFTune perturbs a token embedding with uniform noise scaled to the embedding size,
// bounded by α/√(L·d) per entry.
func ExampleNEFTune() {
	emb := tensor.New(tensor.F64, tensor.Shape{4, 4}) // zeros
	out, _ := nn.NEFTune(backend.NewContext(), emb, 8, rand.New(rand.NewPCG(1, 1)))
	mag := 8.0 / math.Sqrt(16) // = 2
	max := 0.0
	for i := range out.Numel() {
		max = math.Max(max, math.Abs(out.AtF64(tensor.Unravel(i, out.Shape())...)))
	}
	fmt.Println(max <= mag)
	// Output: true
}
