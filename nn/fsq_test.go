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

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1 (paper CONFIRMED, arXiv:2309.15505): quantized codes lie on the finite
// grid — code·⌊L/2⌋ is an integer level — and each channel uses ≤ L distinct values.
func TestFSQCodesOnGrid(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 50))
	levels := []int{3, 4, 5, 8}
	fsq, err := nn.NewFSQ(levels)
	if err != nil {
		t.Fatal(err)
	}
	const batch = 200
	z := randMat(rng, batch, len(levels))
	for i := range z.Numel() { // widen the range so all levels are exercised
		c := tensor.Unravel(i, z.Shape())
		z.SetF64(z.AtF64(c...)*4, c...)
	}
	codes, idx, err := fsq.Quantize(backend.NewContext(), z)
	if err != nil {
		t.Fatal(err)
	}
	seen := make([]map[int]bool, len(levels))
	for i := range seen {
		seen[i] = map[int]bool{}
	}
	for b := 0; b < batch; b++ {
		for i, l := range levels {
			hw := float64(l / 2)
			lvl := codes.AtF64(b, i) * hw
			if math.Abs(lvl-math.Round(lvl)) > 1e-9 {
				t.Errorf("code[%d,%d]·⌊L/2⌋=%g not integer", b, i, lvl)
			}
			seen[i][int(math.Round(lvl))] = true
		}
		if idx[b] < 0 || idx[b] >= fsq.CodebookSize() {
			t.Errorf("index %d out of [0,%d)", idx[b], fsq.CodebookSize())
		}
	}
	for i, l := range levels {
		if len(seen[i]) > l {
			t.Errorf("channel %d used %d distinct levels, want ≤ %d", i, len(seen[i]), l)
		}
	}
}

// §V16 tier-1: the flat index is the mixed-radix encoding of the per-channel levels.
func TestFSQIndexEncoding(t *testing.T) {
	fsq, _ := nn.NewFSQ([]int{3, 3}) // odd L: bounded=tanh(z), levels {−1,0,1}→{0,1,2}
	cases := []struct {
		z    [][]float64
		want int
	}{
		{[][]float64{{-9, -9}}, 0}, // (0,0) ⇒ 0
		{[][]float64{{0, 0}}, 4},   // (1,1) ⇒ 1 + 1·3 = 4 (center)
		{[][]float64{{9, 9}}, 8},   // (2,2) ⇒ 2 + 2·3 = 8 (last)
		{[][]float64{{9, -9}}, 2},  // (2,0) ⇒ 2 + 0·3 = 2
	}
	for _, c := range cases {
		_, idx, _ := fsq.Quantize(backend.NewContext(), toT(c.z))
		if idx[0] != c.want {
			t.Errorf("z=%v ⇒ index %d, want %d", c.z[0], idx[0], c.want)
		}
	}
	if fsq.CodebookSize() != 9 {
		t.Errorf("CodebookSize=%d want 9", fsq.CodebookSize())
	}
}

// §V2 (straight-through, §3.1): the round STE makes the gradient pass through the
// tanh bound — verified against the EXACT closed form (finite-diff is invalid through
// stop_gradient). For odd L the bounding half = normalization ⌊L/2⌋, so
// d(code)/dz = 1 − tanh²(z).
func TestFSQStraightThroughGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 51))
	fsq, _ := nn.NewFSQ([]int{5, 5, 5}) // odd levels ⇒ shift 0, half == ⌊L/2⌋
	const batch, d = 4, 3
	z := randMat(rng, batch, d)
	coef := randMat(rng, batch, d)
	tape := autograd.NewTape()
	codes, _, err := fsq.Quantize(tape.Context(), z)
	if err != nil {
		t.Fatal(err)
	}
	prod, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{codes, coef}, nil)
	loss, _ := sumAll(tape.Context(), prod[0])
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(z)
	for b := 0; b < batch; b++ {
		for i := 0; i < d; i++ {
			th := math.Tanh(z.AtF64(b, i))
			want := coef.AtF64(b, i) * (1 - th*th) // d(code)/dz = 1 − tanh²(z)
			if math.Abs(g.AtF64(b, i)-want) > 1e-9 {
				t.Errorf("grad_z[%d,%d]=%g want STE closed-form %g", b, i, g.AtF64(b, i), want)
			}
		}
	}
}

// §V16 tier-1: FSQ has no learned codebook (unlike VQ-VAE) and no auxiliary loss.
func TestFSQNoParams(t *testing.T) {
	fsq, _ := nn.NewFSQ([]int{4, 4})
	if p := fsq.Params(); len(p) != 0 {
		t.Errorf("FSQ.Params()=%v, want none (implicit codebook)", p)
	}
}

func TestFSQErrors(t *testing.T) {
	if _, err := nn.NewFSQ(nil); err == nil {
		t.Error("empty levels: want error")
	}
	if _, err := nn.NewFSQ([]int{3, 1}); err == nil {
		t.Error("level < 2: want error")
	}
	fsq, _ := nn.NewFSQ([]int{3, 4})
	if _, _, err := fsq.Quantize(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Error("channel/levels mismatch: want error")
	}
	if _, _, err := fsq.Quantize(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 2, 1})); err == nil {
		t.Error("rank-3: want error")
	}
}

func ExampleFSQ() {
	// 2 channels, 3 levels each ⇒ an implicit 9-entry codebook, no parameters.
	fsq, _ := nn.NewFSQ([]int{3, 3})
	z := toT([][]float64{{2.0, -2.0}}) // ch0 → top level (+1), ch1 → bottom (−1)
	codes, idx, _ := fsq.Quantize(backend.NewContext(), z)
	fmt.Printf("codes=[%.0f %.0f] index=%d size=%d\n", codes.AtF64(0, 0), codes.AtF64(0, 1), idx[0], fsq.CodebookSize())
	// Output:
	// codes=[1 -1] index=2 size=9
}
