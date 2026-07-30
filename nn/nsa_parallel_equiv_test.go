package nn_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// NSABranches fans its per-head computation over GOMAXPROCS. Each head writes disjoint output
// columns and runs its exact serial code, so all three branches must be BYTE-FOR-BYTE identical
// to the single-worker result. Locked by computing at GOMAXPROCS=1 and N.
func TestNSABranchesParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	const seq, dm, heads = 96, 128, 4
	mk := func(f func(i int) float64) *tensor.Tensor {
		tt := tensor.New(tensor.F64, tensor.Shape{seq, dm})
		s := tt.Storage().F64()
		for i := range s {
			s[i] = f(i)
		}
		return tt
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.021) })
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.017) })
	v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.013) })

	runtime.GOMAXPROCS(1)
	c1, s1, w1, err := nn.NSABranches(q, k, v, heads, 16, 4, 32, 0)
	if err != nil {
		t.Fatal(err)
	}
	runtime.GOMAXPROCS(prev)
	c2, s2, w2, err := nn.NSABranches(q, k, v, heads, 16, 4, 32, 0)
	if err != nil {
		t.Fatal(err)
	}
	for name, pair := range map[string][2]*tensor.Tensor{"cmp": {c1, c2}, "slc": {s1, s2}, "win": {w1, w2}} {
		a, b := pair[0].Storage().F64(), pair[1].Storage().F64()
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s[%d]: serial %v != parallel %v", name, i, a[i], b[i])
			}
		}
	}
}
