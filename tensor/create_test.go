package tensor_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func TestZerosOnesFull(t *testing.T) {
	z := tensor.Zeros(tensor.F64, tensor.Shape{2, 3})
	if !z.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("Zeros shape %v", z.Shape())
	}
	for i := range z.Numel() {
		if z.AtF64(tensor.Unravel(i, z.Shape())...) != 0 {
			t.Fatalf("Zeros[%d] != 0", i)
		}
	}
	o := tensor.Ones(tensor.F32, tensor.Shape{4})
	if o.Dtype() != tensor.F32 {
		t.Errorf("Ones dtype %v", o.Dtype())
	}
	for i := range 4 {
		if o.AtF64(i) != 1 {
			t.Errorf("Ones[%d] = %v", i, o.AtF64(i))
		}
	}
	f := tensor.Full(tensor.F64, tensor.Shape{3}, -2.5)
	for i := range 3 {
		if f.AtF64(i) != -2.5 {
			t.Errorf("Full[%d] = %v", i, f.AtF64(i))
		}
	}
}

func TestLike(t *testing.T) {
	src := tensor.Ones(tensor.F32, tensor.Shape{2, 5})
	z := tensor.ZerosLike(src)
	o := tensor.OnesLike(src)
	if !z.Shape().Equal(src.Shape()) || z.Dtype() != tensor.F32 || z.AtF64(1, 1) != 0 {
		t.Errorf("ZerosLike mismatch: shape %v dtype %v v %v", z.Shape(), z.Dtype(), z.AtF64(1, 1))
	}
	if o.AtF64(0, 0) != 1 {
		t.Errorf("OnesLike value %v", o.AtF64(0, 0))
	}
}

func TestArange(t *testing.T) {
	a := tensor.Arange(tensor.F64, 0, 5, 1)
	want := []float64{0, 1, 2, 3, 4}
	if a.Numel() != len(want) {
		t.Fatalf("Arange len %d, want %d", a.Numel(), len(want))
	}
	for i, w := range want {
		if a.AtF64(i) != w {
			t.Errorf("Arange[%d] = %v, want %v", i, a.AtF64(i), w)
		}
	}
	// fractional step
	b := tensor.Arange(tensor.F64, 1, 2, 0.25) // [1,1.25,1.5,1.75]
	if b.Numel() != 4 || b.AtF64(2) != 1.5 {
		t.Errorf("Arange step 0.25: len %d, [2]=%v", b.Numel(), b.AtF64(2))
	}
	// negative step
	c := tensor.Arange(tensor.F64, 3, 0, -1) // [3,2,1]
	if c.Numel() != 3 || c.AtF64(0) != 3 || c.AtF64(2) != 1 {
		t.Errorf("Arange negative step: len %d", c.Numel())
	}
	// empty (start >= stop, positive step)
	if e := tensor.Arange(tensor.F64, 5, 5, 1); e.Numel() != 0 {
		t.Errorf("empty Arange len %d, want 0", e.Numel())
	}
}

func TestLinspace(t *testing.T) {
	a := tensor.Linspace(tensor.F64, 0, 1, 5) // [0,0.25,0.5,0.75,1]
	want := []float64{0, 0.25, 0.5, 0.75, 1}
	for i, w := range want {
		if math.Abs(a.AtF64(i)-w) > 1e-15 {
			t.Errorf("Linspace[%d] = %v, want %v", i, a.AtF64(i), w)
		}
	}
	// endpoint is exact
	if a.AtF64(4) != 1 {
		t.Errorf("Linspace endpoint = %v, want exactly 1", a.AtF64(4))
	}
	// n=1 → [start]
	if s := tensor.Linspace(tensor.F64, 3, 9, 1); s.Numel() != 1 || s.AtF64(0) != 3 {
		t.Errorf("Linspace n=1 = %v", s.AtF64(0))
	}
}

func TestEyeIdentity(t *testing.T) {
	e := tensor.Eye(tensor.F64, 3)
	if !e.Shape().Equal(tensor.Shape{3, 3}) {
		t.Fatalf("Eye shape %v", e.Shape())
	}
	for i := range 3 {
		for j := range 3 {
			want := 0.0
			if i == j {
				want = 1
			}
			if e.AtF64(i, j) != want {
				t.Errorf("Eye[%d,%d] = %v, want %v", i, j, e.AtF64(i, j), want)
			}
		}
	}
	if id := tensor.Identity(tensor.F32, 2); id.AtF64(0, 0) != 1 || id.AtF64(0, 1) != 0 {
		t.Error("Identity is not the identity matrix")
	}
}

func TestRandRandn(t *testing.T) {
	// deterministic by seed
	a := tensor.Rand(tensor.F64, 42, tensor.Shape{100})
	b := tensor.Rand(tensor.F64, 42, tensor.Shape{100})
	c := tensor.Rand(tensor.F64, 43, tensor.Shape{100})
	for i := range 100 {
		if a.AtF64(i) != b.AtF64(i) {
			t.Fatalf("Rand not deterministic at %d", i)
		}
		if a.AtF64(i) < 0 || a.AtF64(i) >= 1 {
			t.Fatalf("Rand[%d] = %v out of [0,1)", i, a.AtF64(i))
		}
	}
	if a.AtF64(0) == c.AtF64(0) {
		t.Error("different seeds should differ")
	}
	// Randn: mean ≈ 0 over many samples
	r := tensor.Randn(tensor.F64, 7, tensor.Shape{10000})
	var sum float64
	for i := range 10000 {
		sum += r.AtF64(i)
	}
	if mean := sum / 10000; math.Abs(mean) > 0.05 {
		t.Errorf("Randn mean %v, want ≈ 0", mean)
	}
}

func ExampleArange() {
	a := tensor.Arange(tensor.F64, 0, 4, 1)
	fmt.Println(a.AtF64(0), a.AtF64(1), a.AtF64(2), a.AtF64(3))
	// Output: 0 1 2 3
}

func ExampleLinspace() {
	a := tensor.Linspace(tensor.F64, 0, 1, 3)
	fmt.Println(a.AtF64(0), a.AtF64(1), a.AtF64(2))
	// Output: 0 0.5 1
}

func ExampleEye() {
	e := tensor.Eye(tensor.F64, 2)
	fmt.Println(e.AtF64(0, 0), e.AtF64(0, 1), e.AtF64(1, 0), e.AtF64(1, 1))
	// Output: 1 0 0 1
}

func ExampleZeros() {
	z := tensor.Zeros(tensor.F64, tensor.Shape{2, 2})
	fmt.Println(z.Shape(), z.AtF64(0, 0))
	// Output: (2, 2) 0
}
