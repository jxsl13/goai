package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func ssdInputs(seed, T, d, n int) (x, a, b, c *tensor.Tensor) {
	mk2 := func(s, rows, cols int, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for i := range t.Numel() {
			t.SetF64(math.Sin(float64(s*100+i)*1.7)*scale, tensor.Unravel(i, t.Shape())...)
		}
		return t
	}
	x = mk2(seed, T, d, 0.9)
	b = mk2(seed+1, T, n, 0.8)
	c = mk2(seed+2, T, n, 0.8)
	a = tensor.New(tensor.F64, tensor.Shape{T})
	for t := range T {
		a.SetF64(0.3+0.65*math.Abs(math.Sin(float64(seed*7+t))), t) // decays in (0,1)
	}
	return
}

// §T553, the paper's central theorem (§V16 tier-2): the linear-time recurrence and
// the quadratic 1-semiseparable "attention" form compute the SAME sequence map.
func TestSSDDuality(t *testing.T) {
	x, a, b, c := ssdInputs(3, 12, 5, 4)
	rec, err := nn.SSDRecurrent(x, a, b, c)
	if err != nil {
		t.Fatal(err)
	}
	quad, err := nn.SSDQuadratic(x, a, b, c)
	if err != nil {
		t.Fatal(err)
	}
	for i := range rec.Numel() {
		idx := tensor.Unravel(i, rec.Shape())
		if d := math.Abs(rec.AtF64(idx...) - quad.AtF64(idx...)); d > 1e-12 {
			t.Fatalf("duality broken at %v: recurrent %.17g vs quadratic %.17g", idx, rec.AtF64(idx...), quad.AtF64(idx...))
		}
	}
}

// a_t = 1: the mask is plain C_i·B_j causal linear attention (running sum);
// a_t = 0: the mask is diagonal — y_t = (C_t·B_t)·x_t exactly.
func TestSSDDegenerateDecays(t *testing.T) {
	x, a, b, c := ssdInputs(5, 8, 3, 4)
	T, n := 8, 4
	for i := range T {
		a.SetF64(0, i)
	}
	y, err := nn.SSDRecurrent(x, a, b, c)
	if err != nil {
		t.Fatal(err)
	}
	for i := range T {
		var cb float64
		for k := range n {
			cb += c.AtF64(i, k) * b.AtF64(i, k)
		}
		for d := range 3 {
			want := cb * x.AtF64(i, d)
			if math.Abs(y.AtF64(i, d)-want) > 1e-12 {
				t.Fatalf("a=0 not diagonal at (%d,%d)", i, d)
			}
		}
	}
	for i := range T {
		a.SetF64(1, i)
	}
	y1, err := nn.SSDRecurrent(x, a, b, c)
	if err != nil {
		t.Fatal(err)
	}
	// a=1 running-sum check at the last row: y_T = Σ_j (C_T·B_j)·x_j.
	last := T - 1
	for d := range 3 {
		var want float64
		for j := 0; j <= last; j++ {
			var cb float64
			for k := range n {
				cb += c.AtF64(last, k) * b.AtF64(j, k)
			}
			want += cb * x.AtF64(j, d)
		}
		if math.Abs(y1.AtF64(last, d)-want) > 1e-12 {
			t.Fatalf("a=1 not causal linear attention at d=%d", d)
		}
	}
}

// The duality in action: both forms produce the same map — pick recurrent for
// long sequences (O(T)), quadratic for analysis (O(T²), materialized mask).
func ExampleSSDRecurrent() {
	x := tensor.FromFloat64(tensor.Shape{3, 1}, []float64{1, 1, 1})
	a := tensor.FromFloat64(tensor.Shape{3}, []float64{0.5, 0.5, 0.5})
	b := tensor.FromFloat64(tensor.Shape{3, 1}, []float64{1, 1, 1})
	c := tensor.FromFloat64(tensor.Shape{3, 1}, []float64{1, 1, 1})
	y, _ := nn.SSDRecurrent(x, a, b, c)
	fmt.Println(y.AtF64(0, 0), y.AtF64(1, 0), y.AtF64(2, 0)) // 1, 1+0.5, 1+0.5+0.25
	// Output: 1 1.5 1.75
}
