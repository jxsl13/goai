package ops_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: LogDet matches the log-det branch of numpy.linalg.slogdet (sign=+1
// for SPD) on fixed matrices whose golden value came from numpy 2.5.1 in .venv.
func TestLogDetVsNumpy(t *testing.T) {
	cases := []struct {
		name   string
		n      int
		a      []float64
		logdet float64
	}{
		{"3x3", 3,
			[]float64{4, 12, -16, 12, 37, -43, -16, -43, 98},
			3.583518938456111},
		{"4x4", 4,
			[]float64{
				4.957555796462167, -1.506881695982532, -0.6380945028884456, -0.8890558190293714,
				-1.506881695982532, 6.98988257027522, 1.346849996002904, 1.8048632832193072,
				-0.6380945028884456, 1.346849996002904, 4.994569928934063, 0.7592623904683085,
				-0.8890558190293714, 1.8048632832193072, 0.7592623904683085, 5.361185147456223},
			6.598076671067326},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := tensor.FromFloat64(tensor.Shape{c.n, c.n}, c.a)
			ld, err := ops.LogDet(a)
			if err != nil {
				t.Fatalf("LogDet: %v", err)
			}
			if got := ld.AtF64(); math.Abs(got-c.logdet) > 1e-12 {
				t.Errorf("logdet = %.15g, want %.15g", got, c.logdet)
			}
		})
	}
}

// logdet(A) = ln(det(A)): a diagonal matrix has det = ∏ d_ii, so logdet = Σ ln d_ii.
func TestLogDetDiagonal(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{2, 0, 0, 0, 3, 0, 0, 0, 5})
	ld, err := ops.LogDet(a)
	if err != nil {
		t.Fatalf("LogDet: %v", err)
	}
	want := math.Log(2) + math.Log(3) + math.Log(5) // = ln(30)
	if got := ld.AtF64(); math.Abs(got-want) > 1e-12 {
		t.Errorf("logdet = %.15g, want %.15g (ln 30)", got, want)
	}
}

func TestLogDetErrors(t *testing.T) {
	if _, err := ops.LogDet(tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 0, 0, 0, 1, 0})); err == nil {
		t.Error("non-square: want error")
	}
	if _, err := ops.LogDet(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 1})); err == nil {
		t.Error("indefinite: want error")
	}
}

func ExampleLogDet() {
	// A = 2·I₃ has det = 8, so logdet = ln 8.
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{2, 0, 0, 0, 2, 0, 0, 0, 2})
	ld, _ := ops.LogDet(a)
	fmt.Printf("%.6f\n", ld.AtF64())
	// Output:
	// 2.079442
}
