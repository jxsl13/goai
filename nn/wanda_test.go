package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// wandaScoreRef independently computes S[j,o]=|W[j,o]|·‖X_j‖₂. §V16 tier-1 parity oracle.
func wandaScoreRef(w, x [][]float64) [][]float64 {
	cin, cout, tokens := len(w), len(w[0]), len(x)
	norm := make([]float64, cin)
	for t := 0; t < tokens; t++ {
		for j := 0; j < cin; j++ {
			norm[j] += x[t][j] * x[t][j]
		}
	}
	for j := range norm {
		norm[j] = math.Sqrt(norm[j])
	}
	s := make([][]float64, cin)
	for j := 0; j < cin; j++ {
		s[j] = make([]float64, cout)
		for o := 0; o < cout; o++ {
			s[j][o] = math.Abs(w[j][o]) * norm[j]
		}
	}
	return s
}

func countZeros(t *tensor.Tensor) int {
	z := 0
	for i := range t.Numel() {
		if t.AtF64(tensor.Unravel(i, t.Shape())...) == 0 {
			z++
		}
	}
	return z
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2306.11695 Eq. 2): the Wanda score matches the
// independent |W|·‖X‖ reference.
func TestWandaScoreFormula(t *testing.T) {
	w := [][]float64{{0.5, -1}, {2, 0.25}, {-3, 4}}
	x := [][]float64{{1, -2, 0.5}, {2, 1, -1}, {0.5, 3, 2}}
	got, err := nn.WandaScore(toT(w), toT(x))
	if err != nil {
		t.Fatal(err)
	}
	want := wandaScoreRef(w, x)
	for j := range w {
		for o := range w[0] {
			if math.Abs(got.AtF64(j, o)-want[j][o]) > 1e-9 {
				t.Errorf("S[%d,%d]=%g want %g", j, o, got.AtF64(j, o), want[j][o])
			}
		}
	}
}

// §V2 (THE defining property): Wanda combines weight magnitude with the input-activation
// norm, so a SMALL weight on a high-variance feature is kept over a LARGE weight on a
// near-dead feature — the opposite of pure magnitude pruning.
func TestWandaKeepsHighActivationSmallWeight(t *testing.T) {
	// col 0 activations are large (‖X_0‖≈14), col 1 tiny (‖X_1‖≈0.014).
	w := toT([][]float64{{0.1}, {1.0}}) // small weight on the loud feature, big on the quiet one
	x := toT([][]float64{{10, 0.01}, {10, 0.01}})
	pruned, mask, err := nn.WandaPrune(w, x, 0.5) // drop 1 of the 2 inputs
	if err != nil {
		t.Fatal(err)
	}
	if pruned.AtF64(0, 0) != 0.1 || mask.AtF64(0, 0) != 1 {
		t.Errorf("small weight on loud feature was pruned: pruned=%g mask=%g", pruned.AtF64(0, 0), mask.AtF64(0, 0))
	}
	if pruned.AtF64(1, 0) != 0 || mask.AtF64(1, 0) != 0 {
		t.Errorf("large weight on quiet feature survived: pruned=%g mask=%g", pruned.AtF64(1, 0), mask.AtF64(1, 0))
	}
}

// §V2: the target unstructured sparsity is met exactly, per output column.
func TestWandaSparsityLevel(t *testing.T) {
	w := toT([][]float64{{1, 5}, {2, 6}, {3, 7}, {4, 8}}) // C_in=4, C_out=2
	x := toT([][]float64{{1, 1, 1, 1}})                   // unit norms ⇒ score = |W|
	_, mask, err := nn.WandaPrune(w, x, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if z := countZeros(mask); z != 4 { // ⌊0.5·4⌋=2 dropped per column · 2 columns
		t.Errorf("total pruned = %d, want 4", z)
	}
}

// §V16 tier-1 (per-OUTPUT comparison group): each output column is pruned to its own k,
// NOT against a single global threshold. Columns of wildly different scale each keep
// exactly C_in−k weights; a global threshold would wipe the small-scale column entirely.
func TestWandaPerOutputComparisonGroup(t *testing.T) {
	w := toT([][]float64{{1, 10}, {2, 20}, {3, 30}, {4, 40}}) // col1 is 10× col0
	x := toT([][]float64{{1, 1, 1, 1}})
	_, mask, err := nn.WandaPrune(w, x, 0.5) // k=2 per column
	if err != nil {
		t.Fatal(err)
	}
	for o := range 2 {
		kept := 0
		for j := range 4 {
			kept += int(mask.AtF64(j, o))
		}
		if kept != 2 {
			t.Errorf("column %d kept %d, want 2 (per-output grouping)", o, kept)
		}
	}
}

// §V16 tier-1 (N:M structured): 2:4 keeps the 2 highest-score weights in every block of
// 4 consecutive inputs.
func TestWandaNMStructured(t *testing.T) {
	w := toT([][]float64{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}) // C_in=8, C_out=1
	x := toT([][]float64{{1, 1, 1, 1, 1, 1, 1, 1}})               // score = |W|
	_, mask, err := nn.WandaPruneNM(w, x, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 0, 1, 1, 0, 0, 1, 1} // each block of 4 keeps its 2 largest
	for j := range want {
		if mask.AtF64(j, 0) != want[j] {
			t.Errorf("mask[%d]=%g want %g", j, mask.AtF64(j, 0), want[j])
		}
	}
}

// §V2 (edges): sparsity 0 is a no-op; sparsity 1 removes everything.
func TestWandaSparsityEdges(t *testing.T) {
	w := toT([][]float64{{1, 2}, {3, 4}})
	x := toT([][]float64{{1, 1}})
	kept, mask, err := nn.WandaPrune(w, x, 0)
	if err != nil {
		t.Fatal(err)
	}
	if countZeros(mask) != 0 || kept.AtF64(1, 1) != 4 {
		t.Errorf("sparsity 0 should be a no-op, got %d zeros, kept[1,1]=%g", countZeros(mask), kept.AtF64(1, 1))
	}
	gone, _, _ := nn.WandaPrune(w, x, 1)
	if countZeros(gone) != 4 {
		t.Errorf("sparsity 1 should zero all, got %d/4 zeros", countZeros(gone))
	}
}

func TestWandaErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.WandaScore(r(3, 4), r(2, 5)); err == nil {
		t.Error("C_in mismatch: want error")
	}
	if _, err := nn.WandaScore(r(3, 4, 1), r(2, 3)); err == nil {
		t.Error("rank-3 W: want error")
	}
	if _, _, err := nn.WandaPrune(r(3, 2), r(2, 3), 1.5); err == nil {
		t.Error("sparsity>1: want error")
	}
	if _, _, err := nn.WandaPruneNM(r(6, 2), r(2, 6), 2, 4); err == nil {
		t.Error("C_in=6 not divisible by m=4: want error")
	}
	if _, _, err := nn.WandaPruneNM(r(4, 2), r(2, 4), 5, 4); err == nil {
		t.Error("n>m: want error")
	}
}

func ExampleWandaPrune() {
	// Two inputs feed one output. Input 0 is a loud feature (large activation norm) with a
	// small weight; input 1 is quiet with a large weight. Magnitude pruning would drop the
	// 0.1 weight — Wanda keeps it, because 0.1·‖loud‖ > 1.0·‖quiet‖.
	w := toT([][]float64{{0.1}, {1.0}})
	x := toT([][]float64{{10, 0.01}, {10, 0.01}})
	pruned, _, _ := nn.WandaPrune(w, x, 0.5)
	fmt.Printf("kept=[%.1f %.1f]\n", pruned.AtF64(0, 0), pruned.AtF64(1, 0))
	// Output:
	// kept=[0.1 0.0]
}

// §B111: WandaPrune documents ⌊sparsity·C_in⌋ dropped per column (the paper's int()
// truncation), but used math.Round — over-pruning fractional cases. C_in=3, sparsity 0.5:
// ⌊1.5⌋=1 dropped (2 kept), not Round(1.5)=2 dropped (1 kept). Also checks the f64-exact
// target 0.7·10=6.999… floors to 7, not 6.
func TestWandaFloorNotRound(t *testing.T) {
	w3 := toT([][]float64{{1}, {2}, {3}}) // C_in=3, C_out=1
	x3 := toT([][]float64{{1, 1, 1}})
	_, mask, err := nn.WandaPrune(w3, x3, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	kept := 0
	for j := range 3 {
		kept += int(mask.AtF64(j, 0))
	}
	if kept != 2 { // ⌊0.5·3⌋=1 dropped ⇒ 2 kept (Round dropped 2 ⇒ 1 kept)
		t.Errorf("C_in=3 sparsity=0.5 kept %d, want 2 (⌊1.5⌋=1 dropped, not Round=2)", kept)
	}

	w10 := toT(func() [][]float64 {
		m := make([][]float64, 10)
		for i := range m {
			m[i] = []float64{float64(i + 1)}
		}
		return m
	}())
	x10 := toT([][]float64{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}})
	_, mask10, err := nn.WandaPrune(w10, x10, 0.7)
	if err != nil {
		t.Fatal(err)
	}
	kept10 := 0
	for j := range 10 {
		kept10 += int(mask10.AtF64(j, 0))
	}
	if kept10 != 3 { // ⌊0.7·10⌋=7 dropped ⇒ 3 kept (naive f64 floor of 6.999… would drop 6 ⇒ 4 kept)
		t.Errorf("C_in=10 sparsity=0.7 kept %d, want 3 (⌊7⌋ dropped, f64-robust)", kept10)
	}
}
