package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Xavier: all values within ±√(6/(fanIn+fanOut)), mean ≈ 0, deterministic.
func TestXavierUniform(t *testing.T) {
	const fanIn, fanOut = 30, 20
	bound := math.Sqrt(6.0 / float64(fanIn+fanOut))
	w := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.XavierUniform(w, fanIn, fanOut, 1)

	var sum float64
	n := w.Numel()
	for i := range n {
		v := w.AtF64(tensor.Unravel(i, w.Shape())...)
		if v < -bound || v > bound {
			t.Fatalf("value %v outside ±%v", v, bound)
		}
		sum += v
	}
	if mean := sum / float64(n); math.Abs(mean) > bound/5 {
		t.Errorf("mean %v too far from 0 (bound %v)", mean, bound)
	}

	// determinism (§V13): same seed → identical tensor
	w2 := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.XavierUniform(w2, fanIn, fanOut, 1)
	for i := range n {
		idx := tensor.Unravel(i, w.Shape())
		if w.AtF64(idx...) != w2.AtF64(idx...) {
			t.Fatal("same seed must reproduce identical init")
		}
	}
	// different seed → different tensor
	w3 := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.XavierUniform(w3, fanIn, fanOut, 2)
	same := true
	for i := range n {
		idx := tensor.Unravel(i, w.Shape())
		if w.AtF64(idx...) != w3.AtF64(idx...) {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds must differ")
	}
}

// Kaiming-normal: sample std ≈ √(2/fanIn) within 15% at n=6000.
func TestKaimingNormal(t *testing.T) {
	const fanIn, fanOut = 100, 60
	want := math.Sqrt(2.0 / float64(fanIn))
	w := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.KaimingNormal(w, fanIn, 7)

	var sum, sq float64
	n := w.Numel()
	for i := range n {
		v := w.AtF64(tensor.Unravel(i, w.Shape())...)
		sum += v
		sq += v * v
	}
	mean := sum / float64(n)
	std := math.Sqrt(sq/float64(n) - mean*mean)
	if math.Abs(std-want)/want > 0.15 {
		t.Errorf("std %v, want ~%v", std, want)
	}
	if math.Abs(mean) > want/5 {
		t.Errorf("mean %v too far from 0", mean)
	}
}

func TestZerosAndNewLinear(t *testing.T) {
	l := nn.NewLinear(tensor.F64, 4, 3, 99)
	for j := range 3 {
		if l.B.AtF64(j) != 0 {
			t.Error("bias must init to zero")
		}
	}
	if !l.W.Shape().Equal(tensor.Shape{4, 3}) {
		t.Errorf("W shape %v, want (4,3)", l.W.Shape())
	}
}
