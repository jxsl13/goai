package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// wkvRef computes the WKV operator directly from the definition (no running-max
// stabilization) — the §V16 tier-1 parity oracle for the stable kernel.
func wkvRef(k, v [][]float64, w, u []float64) [][]float64 {
	seq, d := len(k), len(k[0])
	out := make([][]float64, seq)
	for t := range out {
		out[t] = make([]float64, d)
		for c := 0; c < d; c++ {
			var num, den float64
			for i := 0; i < t; i++ {
				wt := math.Exp(-float64(t-1-i)*w[c] + k[i][c])
				num += wt * v[i][c]
				den += wt
			}
			bonus := math.Exp(u[c] + k[t][c])
			num += bonus * v[t][c]
			den += bonus
			out[t][c] = num / den
		}
	}
	return out
}

func dvec(vals ...float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{len(vals)})
	for i, x := range vals {
		t.SetF64(x, i)
	}
	return t
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2305.13048): the stable running-max kernel
// matches the direct WKV formula.
func TestWKVParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 120))
	const seq, d = 6, 3
	k, v := randMat(rng, seq, d), randMat(rng, seq, d)
	w := dvec(0.3, 1.0, 2.5) // positive per-channel decay
	u := dvec(0.1, -0.2, 0.5)
	got, err := nn.WKV(k, v, w, u)
	if err != nil {
		t.Fatal(err)
	}
	want := wkvRef(toRows(k), toRows(v), []float64{0.3, 1.0, 2.5}, []float64{0.1, -0.2, 0.5})
	for tt := 0; tt < seq; tt++ {
		for c := 0; c < d; c++ {
			if math.Abs(got.AtF64(tt, c)-want[tt][c]) > 1e-9 {
				t.Errorf("wkv[%d,%d]=%.10g want %.10g", tt, c, got.AtF64(tt, c), want[tt][c])
			}
		}
	}
}

// §V16 tier-1: at t=1 the output is exactly v_1 (the past-sum is empty).
func TestWKVFirstTokenIsValue(t *testing.T) {
	k := toT([][]float64{{0.5, -1, 2}})
	v := toT([][]float64{{3, 7, -2}})
	got, err := nn.WKV(k, v, dvec(1, 1, 1), dvec(0.5, 0.5, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	for c := 0; c < 3; c++ {
		if math.Abs(got.AtF64(0, c)-v.AtF64(0, c)) > 1e-12 {
			t.Errorf("wkv_1[%d]=%g want v_1=%g", c, got.AtF64(0, c), v.AtF64(0, c))
		}
	}
}

// §V2 (defining property): with w=0, k=0, u=0 every weight is 1, so wkv_t is the
// running MEAN of the values seen so far.
func TestWKVUniformIsRunningMean(t *testing.T) {
	v := toT([][]float64{{2}, {4}, {6}, {8}})
	got, err := nn.WKV(toT([][]float64{{0}, {0}, {0}, {0}}), v, dvec(0), dvec(0))
	if err != nil {
		t.Fatal(err)
	}
	for t2, want := range []float64{2, 3, 4, 5} { // running means of [2],[2,4],[2,4,6],...
		if math.Abs(got.AtF64(t2, 0)-want) > 1e-9 {
			t.Errorf("wkv[%d]=%g want running mean %g", t2, got.AtF64(t2, 0), want)
		}
	}
}

// §V2 (bonus): a large current-token bonus u makes e^{u+k_t} dominate every past
// weight, so the output collapses onto the current token. (Note: large decay w alone
// does NOT do this — the immediately-previous token has zero decay e^{−0·w}; only the
// bonus outweighs it.)
func TestWKVLargeBonusIsCurrentToken(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 121))
	const seq, d = 5, 2
	k, v := randMat(rng, seq, d), randMat(rng, seq, d)
	got, err := nn.WKV(k, v, dvec(1, 1), dvec(30, 30)) // huge bonus
	if err != nil {
		t.Fatal(err)
	}
	for tt := 0; tt < seq; tt++ {
		for c := 0; c < d; c++ {
			if math.Abs(got.AtF64(tt, c)-v.AtF64(tt, c)) > 1e-6 {
				t.Errorf("large-bonus wkv[%d,%d]=%g != current v %g", tt, c, got.AtF64(tt, c), v.AtF64(tt, c))
			}
		}
	}
}

func TestWKVErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.WKV(r(4, 3), r(4, 2), dvec(0, 0, 0), dvec(0, 0, 0)); err == nil {
		t.Error("k/v shape mismatch: want error")
	}
	if _, err := nn.WKV(r(4, 3), r(4, 3), dvec(0, 0), dvec(0, 0, 0)); err == nil {
		t.Error("w dim mismatch: want error")
	}
	if _, err := nn.WKV(r(4, 3, 1), r(4, 3, 1), dvec(0, 0, 0), dvec(0, 0, 0)); err == nil {
		t.Error("rank-3: want error")
	}
}

func ExampleWKV() {
	// single channel, w=0/u=0: the WKV output is the running mean of the values.
	k := toT([][]float64{{0}, {0}, {0}})
	v := toT([][]float64{{1}, {2}, {9}})
	out, _ := nn.WKV(k, v, dvec(0), dvec(0))
	fmt.Printf("%.1f %.1f %.1f\n", out.AtF64(0, 0), out.AtF64(1, 0), out.AtF64(2, 0))
	// Output:
	// 1.0 1.5 4.0
}
