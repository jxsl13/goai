package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// params builds a single-tensor parameter list from flat data (shape inferred as [n]).
func params(d ...float64) []*tensor.Tensor {
	return []*tensor.Tensor{tensor.FromFloat64(tensor.Shape{len(d)}, d)}
}

func flat(t *tensor.Tensor) []float64 {
	out := make([]float64, t.Numel())
	for i := range out {
		out[i] = t.AtF64(tensor.Unravel(i, t.Shape())...)
	}
	return out
}

func approx(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len %d != %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("%s[%d] = %g, want %g", name, i, got[i], want[i])
		}
	}
}

// §V16 tier-1: merging a single fine-tuned model with density=1, λ=1 recovers
// that model exactly (τ trimmed to itself, sign-elects itself, mean of one) (§R123).
func TestTIESMergeSingleModel(t *testing.T) {
	base := params(1, 2, 3, 4)
	model := params(1.5, 1.0, 3.7, -2.0)
	out, err := nn.TIESMerge(base, [][]*tensor.Tensor{model}, 1.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "single", flat(out[0]), flat(model[0]))
}

// §V16 tier-1: the disjoint sign-elect merge — with base 0 so task vectors equal
// the model weights, coordinate 0 has models {+3,+1,−2}: elected sign + (Σ=+2),
// so only {+3,+1} average to 2 and the −2 is discarded. Coordinate 1 {−4,−1,+2}:
// elected − (Σ=−3), average {−4,−1}=−2.5, discard +2.
func TestTIESSignElection(t *testing.T) {
	base := params(0, 0)
	m1 := params(3, -4)
	m2 := params(1, -1)
	m3 := params(-2, 2)
	out, err := nn.TIESMerge(base, [][]*tensor.Tensor{m1, m2, m3}, 1.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "elect", flat(out[0]), []float64{2.0, -2.5})
}

// §V16 tier-1: TRIM keeps only the top-density fraction by magnitude. base 0, one
// model [10, 0.1, 8, 0.2], density 0.5 → keep top 2 (10, 8); the small 0.1/0.2
// are zeroed, so the merged task vector is [10,0,8,0].
func TestTIESTrim(t *testing.T) {
	base := params(0, 0, 0, 0)
	model := params(10, 0.1, 8, 0.2)
	out, err := nn.TIESMerge(base, [][]*tensor.Tensor{model}, 0.5, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "trim", flat(out[0]), []float64{10, 0, 8, 0})
}

// §V16 tier-1: full hand-computed merge with a nonzero base and λ scaling.
// base=[1,1], m1=[4,-1] (τ=[3,-2]), m2=[2,-5] (τ=[1,-6]); density 1.
// coord0: Σ=4>0 → mean{3,1}=2; coord1: Σ=−8<0 → mean{−2,−6}=−4.
// θ=base+λ·τ_merged with λ=0.5 → [1+0.5·2, 1+0.5·(−4)] = [2, -1].
func TestTIESMergeLambda(t *testing.T) {
	base := params(1, 1)
	m1 := params(4, -1)
	m2 := params(2, -5)
	out, err := nn.TIESMerge(base, [][]*tensor.Tensor{m1, m2}, 1.0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "lambda", flat(out[0]), []float64{2.0, -1.0})
}

// §V16 tier-1: a coordinate where the trimmed sign-sum is zero (perfect +/−
// cancellation) elects no sign and merges to the base (τ_merged=0). base 0,
// models {+2},{−2}: Σ=0 → out 0.
func TestTIESZeroSumStaysBase(t *testing.T) {
	out, err := nn.TIESMerge(params(5), [][]*tensor.Tensor{params(7), params(3)}, 1.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	// τ1=+2, τ2=−2 → Σ=0 → merged 0 → out = base = 5
	approx(t, "zerosum", flat(out[0]), []float64{5})
}

func TestTIESMergeErrors(t *testing.T) {
	base := params(1, 2)
	good := params(1, 2)
	cases := []struct {
		name    string
		base    []*tensor.Tensor
		models  [][]*tensor.Tensor
		density float64
	}{
		{"no models", base, nil, 0.5},
		{"density 0", base, [][]*tensor.Tensor{good}, 0},
		{"density >1", base, [][]*tensor.Tensor{good}, 1.5},
		{"param count", base, [][]*tensor.Tensor{{good[0], good[0]}}, 0.5},
		{"shape mismatch", base, [][]*tensor.Tensor{params(1, 2, 3)}, 0.5},
	}
	for _, c := range cases {
		if _, err := nn.TIESMerge(c.base, c.models, c.density, 1.0); err == nil {
			t.Errorf("%s: expected error", c.name)
		}
	}
	// dtype mismatch
	f32 := []*tensor.Tensor{tensor.New(tensor.F32, tensor.Shape{2})}
	if _, err := nn.TIESMerge(base, [][]*tensor.Tensor{f32}, 0.5, 1.0); err == nil {
		t.Error("dtype mismatch: expected error")
	}
}

// TIES-Merging (Yadav et al. 2023) combines models fine-tuned from a shared base
// while resolving sign conflicts. Here two models edit a zero base in agreement on
// coord 0 (both push +) but conflict on coord 1 (+3 vs −5): the elected sign is −
// so only the −5 survives. With density=1 and λ=1 the merge is [mean(2,4), −5].
func ExampleTIESMerge() {
	base := params(0, 0)
	m1 := params(2, 3)
	m2 := params(4, -5)
	out, _ := nn.TIESMerge(base, [][]*tensor.Tensor{m1, m2}, 1.0, 1.0)
	v := out[0]
	fmt.Printf("%.1f %.1f\n", v.AtF64(0), v.AtF64(1))
	// Output: 3.0 -5.0
}
