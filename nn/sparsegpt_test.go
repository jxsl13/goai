package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func mat(t *tensor.Tensor) [][]float64 {
	r, c := t.Shape()[0], t.Shape()[1]
	m := make([][]float64, r)
	for i := range r {
		m[i] = make([]float64, c)
		for j := range c {
			m[i][j] = t.AtF64(i, j)
		}
	}
	return m
}

// sgReconErr = ‖(W − Ŵ)·X‖_F, the layer-output error SparseGPT minimizes. W,Ŵ are
// [out,in]; x is [in,samples].
func sgReconErr(orig, pr, x [][]float64) float64 {
	out, in, samples := len(orig), len(orig[0]), len(x[0])
	var s float64
	for r := 0; r < out; r++ {
		for k := 0; k < samples; k++ {
			var acc float64
			for c := 0; c < in; c++ {
				acc += (orig[r][c] - pr[r][c]) * x[c][k]
			}
			s += acc * acc
		}
	}
	return math.Sqrt(s)
}

// magPrune zeros the k smallest-|w| weights per row (no compensation) — the baseline
// SparseGPT must beat.
func magPrune(w [][]float64, sparsity float64) [][]float64 {
	out, in := len(w), len(w[0])
	k := int(math.Round(sparsity * float64(in)))
	res := make([][]float64, out)
	for r := range w {
		res[r] = make([]float64, in)
		copy(res[r], w[r])
		idx := make([]int, in)
		for c := range idx {
			idx[c] = c
		}
		sort.SliceStable(idx, func(a, b int) bool { return math.Abs(w[r][idx[a]]) < math.Abs(w[r][idx[b]]) })
		for t := range k {
			res[r][idx[t]] = 0
		}
	}
	return res
}

// §V2 (THE defining benefit): SparseGPT's OBS inverse-Hessian compensation reconstructs
// the layer output ‖WX − ŴX‖_F with strictly lower error than magnitude pruning at the
// same sparsity — the whole reason to pay for the Hessian.
func TestSparseGPTReconstructionBeatsMagnitude(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 774))
	const out, in, samples = 6, 8, 40
	w := randMat(rng, out, in)
	x := randMat(rng, in, samples)
	sg, _, err := nn.SparseGPTPrune(w, x, 0.5, nn.WithSparseGPTBlock(in))
	if err != nil {
		t.Fatal(err)
	}
	wA, xA := mat(w), mat(x)
	sgErr := sgReconErr(wA, mat(sg), xA)
	magErr := sgReconErr(wA, magPrune(wA, 0.5), xA)
	if !(sgErr < magErr) {
		t.Errorf("SparseGPT recon err %.6g not < magnitude %.6g", sgErr, magErr)
	}
}

// §V2 (invariant): sparsity 0 prunes nothing and applies no update, so the output equals
// W exactly and the mask is all-ones.
func TestSparseGPTZeroSparsityIsIdentity(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 775))
	w := randMat(rng, 4, 5)
	x := randMat(rng, 5, 20)
	pr, mask, err := nn.SparseGPTPrune(w, x, 0)
	if err != nil {
		t.Fatal(err)
	}
	for r := range 4 {
		for c := range 5 {
			if pr.AtF64(r, c) != w.AtF64(r, c) || mask.AtF64(r, c) != 1 {
				t.Fatalf("sparsity 0 changed [%d,%d]: %g→%g mask=%g", r, c, w.AtF64(r, c), pr.AtF64(r, c), mask.AtF64(r, c))
			}
		}
	}
}

// §V2 (edge): sparsity 1 removes every weight.
func TestSparseGPTFullSparsityIsZero(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 776))
	w := randMat(rng, 4, 6)
	x := randMat(rng, 6, 20)
	pr, _, err := nn.SparseGPTPrune(w, x, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range pr.Numel() {
		if v := pr.AtF64(tensor.Unravel(i, pr.Shape())...); v != 0 {
			t.Fatalf("sparsity 1 left a nonzero weight %g", v)
		}
	}
}

// §V2: with one block over the whole row, each output row is pruned to exactly
// ⌊sparsity·in⌋ zeros (per-row comparison group, Algorithm 1).
func TestSparseGPTSparsityLevel(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 777))
	const out, in = 5, 8
	w := randMat(rng, out, in)
	x := randMat(rng, in, 30)
	_, mask, err := nn.SparseGPTPrune(w, x, 0.5, nn.WithSparseGPTBlock(in))
	if err != nil {
		t.Fatal(err)
	}
	for r := range out {
		zeros := 0
		for c := range in {
			if mask.AtF64(r, c) == 0 {
				zeros++
			}
		}
		if zeros != 4 { // ⌊0.5·8⌋
			t.Errorf("row %d pruned %d, want 4", r, zeros)
		}
	}
}

// §V16 tier-1 (OBS closed form): when the calibration Hessian is diagonal (orthogonal
// inputs) there is NO cross-column compensation, so SparseGPT reduces to weighted-
// magnitude pruning and the kept weight is left EXACTLY unchanged — verifying the metric
// (prune the smaller weight) and that compensation only flows through off-diagonal Hinv.
func TestSparseGPTDiagonalHessianNoCompensation(t *testing.T) {
	w := toT([][]float64{{3, 1}})         // one output, two inputs
	x := toT([][]float64{{1, 0}, {0, 1}}) // XXᵀ = I ⇒ diagonal H, no off-diagonal compensation
	pr, mask, err := nn.SparseGPTPrune(w, x, 0.5, nn.WithSparseGPTBlock(2))
	if err != nil {
		t.Fatal(err)
	}
	if pr.AtF64(0, 0) != 3 || mask.AtF64(0, 0) != 1 {
		t.Errorf("kept weight changed: %g (mask %g), want 3 unchanged", pr.AtF64(0, 0), mask.AtF64(0, 0))
	}
	if pr.AtF64(0, 1) != 0 || mask.AtF64(0, 1) != 0 {
		t.Errorf("smaller weight not pruned: %g (mask %g)", pr.AtF64(0, 1), mask.AtF64(0, 1))
	}
}

func TestSparseGPTErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, _, err := nn.SparseGPTPrune(r(4, 3, 1), r(3, 5), 0.5); err == nil {
		t.Error("rank-3 w: want error")
	}
	if _, _, err := nn.SparseGPTPrune(r(4, 3), r(5, 6), 0.5); err == nil {
		t.Error("in mismatch: want error")
	}
	if _, _, err := nn.SparseGPTPrune(r(4, 3), r(3, 6), 1.5); err == nil {
		t.Error("sparsity>1: want error")
	}
}

// magPruneNM zeros the n smallest-|w| weights in each group of m consecutive inputs per
// row (no compensation) — the structured baseline SparseGPTPruneNM must beat.
func magPruneNM(w [][]float64, n, m int) [][]float64 {
	out, in := len(w), len(w[0])
	res := make([][]float64, out)
	for r := range w {
		res[r] = make([]float64, in)
		copy(res[r], w[r])
		for base := 0; base < in; base += m {
			idx := make([]int, m)
			for t := range idx {
				idx[t] = base + t
			}
			sort.SliceStable(idx, func(a, b int) bool { return math.Abs(w[r][idx[a]]) < math.Abs(w[r][idx[b]]) })
			for t := range n {
				res[r][idx[t]] = 0
			}
		}
	}
	return res
}

// §V16 tier-1 (N:M structured, arXiv:2301.00774 §3.3): 2:4 keeps the 2 highest-saliency
// weights in every block of 4 consecutive inputs. With orthogonal calibration (diagonal
// Hessian) the saliency is ∝ w² and the kept weights are left exactly unchanged.
func TestSparseGPTNMStructure(t *testing.T) {
	xId := make([][]float64, 8)
	for i := range xId {
		xId[i] = make([]float64, 8)
		xId[i][i] = 1 // X = I₈ ⇒ XXᵀ = I ⇒ diagonal H, no cross-column compensation
	}
	w := toT([][]float64{{1, 2, 3, 4, 5, 6, 7, 8}})
	pr, mask, err := nn.SparseGPTPruneNM(w, toT(xId), 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	wantMask := []float64{0, 0, 1, 1, 0, 0, 1, 1} // each block of 4 keeps its 2 largest
	for c := range wantMask {
		if mask.AtF64(0, c) != wantMask[c] {
			t.Errorf("mask[%d]=%g want %g", c, mask.AtF64(0, c), wantMask[c])
		}
	}
	// kept weights unchanged (diagonal Hessian ⇒ no compensation).
	for _, c := range []int{2, 3, 6, 7} {
		if pr.AtF64(0, c) != float64(c+1) {
			t.Errorf("kept weight col %d = %g, want %d unchanged", c, pr.AtF64(0, c), c+1)
		}
	}
}

// §V2 (the OBS benefit, N:M): 2:4 SparseGPT reconstructs ‖(W−Ŵ)X‖_F with strictly lower
// error than magnitude-based 2:4 pruning at the same structured sparsity.
func TestSparseGPTNMReconstructionBeatsMagnitude(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 778))
	const out, in, samples = 6, 8, 40
	w := randMat(rng, out, in)
	x := randMat(rng, in, samples)
	sg, _, err := nn.SparseGPTPruneNM(w, x, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	wA, xA := mat(w), mat(x)
	sgErr := sgReconErr(wA, mat(sg), xA)
	magErr := sgReconErr(wA, magPruneNM(wA, 2, 4), xA)
	if !(sgErr < magErr) {
		t.Errorf("SparseGPT 2:4 recon err %.6g not < magnitude 2:4 %.6g", sgErr, magErr)
	}
}

// §V2: N:M yields exactly n/m sparsity per row.
func TestSparseGPTNMSparsityCount(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 779))
	const out, in = 5, 8
	w := randMat(rng, out, in)
	x := randMat(rng, in, 30)
	_, mask, err := nn.SparseGPTPruneNM(w, x, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	for r := range out {
		zeros := 0
		for c := range in {
			if mask.AtF64(r, c) == 0 {
				zeros++
			}
		}
		if zeros != 4 { // 2 per 4-block · 2 blocks
			t.Errorf("row %d pruned %d, want 4 (2:4)", r, zeros)
		}
	}
}

func TestSparseGPTNMErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, _, err := nn.SparseGPTPruneNM(r(4, 8), r(8, 10), 5, 4); err == nil {
		t.Error("n>m: want error")
	}
	if _, _, err := nn.SparseGPTPruneNM(r(4, 6), r(6, 10), 2, 4); err == nil {
		t.Error("in=6 not divisible by m=4: want error")
	}
	if _, _, err := nn.SparseGPTPruneNM(r(4, 8, 1), r(8, 10), 2, 4); err == nil {
		t.Error("rank-3 w: want error")
	}
}

func ExampleSparseGPTPrune() {
	// One output reads two inputs through orthogonal calibration activations (diagonal
	// Hessian). SparseGPT prunes the smaller weight and — because there is no cross-column
	// coupling here — leaves the surviving weight untouched.
	w := toT([][]float64{{3, 1}})
	x := toT([][]float64{{1, 0}, {0, 1}})
	pr, _, _ := nn.SparseGPTPrune(w, x, 0.5)
	fmt.Printf("[%.0f %.0f]\n", pr.AtF64(0, 0), pr.AtF64(0, 1))
	// Output:
	// [3 0]
}
