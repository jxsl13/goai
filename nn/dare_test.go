package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func constParams(n int, v float64) []*tensor.Tensor {
	d := make([]float64, n)
	for i := range d {
		d[i] = v
	}
	return []*tensor.Tensor{tensor.FromFloat64(tensor.Shape{n}, d)}
}

// §V16 tier-1: dropRate 0 keeps every delta and rescales by 1/(1−0)=1, so DARE
// returns the fine-tuned model unchanged (§R124).
func TestDARENoDrop(t *testing.T) {
	base := params(1, 2, 3, 4)
	model := params(1.5, 1.0, 3.7, -2.0)
	out, err := nn.DARE(base, model, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, "nodrop", flat(out[0]), flat(model[0]))
}

// §V16 tier-1: each output is EXACTLY either the base (dropped) or base +
// δ/(1−p) (kept). With base 0, model ones, p=0.75 the rescale is 1/0.25=4, so
// every entry is 0 or 4 — no other value can appear.
func TestDARERescaleValues(t *testing.T) {
	base := constParams(2000, 0)
	model := constParams(2000, 1)
	out, err := nn.DARE(base, model, 0.75, 7)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range flat(out[0]) {
		if math.Abs(v) > 1e-12 && math.Abs(v-4) > 1e-12 {
			t.Fatalf("entry %d = %g, want 0 or 4", i, v)
		}
	}
}

// §V16 tier-1: DARE is unbiased in expectation (E[δ̃]=δ) and drops a fraction ≈ p.
// base 0, model constant c=3, p=0.5 → survivors are 6, dropped 0; the empirical
// drop fraction ≈ 0.5 and the mean ≈ 3 = δ.
func TestDAREExpectationAndDropFraction(t *testing.T) {
	const n, c, p = 20000, 3.0, 0.5
	out, err := nn.DARE(constParams(n, 0), constParams(n, c), p, 20260707)
	if err != nil {
		t.Fatal(err)
	}
	var dropped int
	var sum float64
	for _, v := range flat(out[0]) {
		if v == 0 {
			dropped++
		}
		sum += v
	}
	if frac := float64(dropped) / n; math.Abs(frac-p) > 0.02 {
		t.Errorf("drop fraction %.4f, want ≈ %.2f", frac, p)
	}
	if mean := sum / n; math.Abs(mean-c) > 0.15 {
		t.Errorf("mean survivor contribution %.4f, want ≈ %.1f (unbiased)", mean, c)
	}
}

// §V16 tier-1: the mask is deterministic in the seed — same seed reproduces the
// output exactly, a different seed changes it.
func TestDAREDeterministic(t *testing.T) {
	base, model := constParams(500, 0), constParams(500, 1)
	a, _ := nn.DARE(base, model, 0.6, 42)
	b, _ := nn.DARE(base, model, 0.6, 42)
	approx(t, "same-seed", flat(a[0]), flat(b[0]))
	c, _ := nn.DARE(base, model, 0.6, 43)
	same := true
	fa, fc := flat(a[0]), flat(c[0])
	for i := range fa {
		if fa[i] != fc[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different seed produced identical mask")
	}
}

// §V16 tier-1 (composition): DARE-TIES — DARE each model then TIES-merge. Smoke
// test that the pipeline runs and preserves shapes.
func TestDARETIESComposition(t *testing.T) {
	base := params(0, 0, 0, 0)
	m1, _ := nn.DARE(base, params(1, -2, 3, -4), 0.5, 1)
	m2, _ := nn.DARE(base, params(-1, 2, 3, 4), 0.5, 2)
	merged, err := nn.TIESMerge(base, [][]*tensor.Tensor{m1, m2}, 1.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if !merged[0].Shape().Equal(tensor.Shape{4}) {
		t.Errorf("merged shape %v", merged[0].Shape())
	}
}

func TestDAREErrors(t *testing.T) {
	base := params(1, 2)
	good := params(1, 2)
	if _, err := nn.DARE(base, good, 1.0, 1); err == nil {
		t.Error("dropRate 1 must error (divide by zero)")
	}
	if _, err := nn.DARE(base, good, -0.1, 1); err == nil {
		t.Error("negative dropRate must error")
	}
	if _, err := nn.DARE(base, params(1, 2, 3), 0.5, 1); err == nil {
		t.Error("shape mismatch must error")
	}
	if _, err := nn.DARE(base, []*tensor.Tensor{good[0], good[0]}, 0.5, 1); err == nil {
		t.Error("param-count mismatch must error")
	}
}

// DARE (Yu et al. 2024) randomly drops delta parameters and rescales the rest by
// 1/(1−p), keeping the change unbiased. With p=0.5 a surviving delta of 2
// (model−base) is scaled by 2; here (seed 2) both survive → base + 2·2 = 5.
func ExampleDARE() {
	base := params(1, 1)
	model := params(3, 3) // delta = [2, 2]
	out, _ := nn.DARE(base, model, 0.5, 2)
	fmt.Printf("%.0f %.0f\n", out[0].AtF64(0), out[0].AtF64(1))
	// Output: 5 5
}
