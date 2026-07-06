package ref

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type gemmFile struct {
	Gemm struct {
		A      []float64 `json:"a"`
		AShape []int     `json:"a_shape"`
		B      []float64 `json:"b"`
		BShape []int     `json:"b_shape"`
		F64    []float64 `json:"f64"`
		F32    []float64 `json:"f32"`
		Shape  []int     `json:"shape"`
	} `json:"gemm"`
}

func loadGemm(t *testing.T) gemmFile {
	t.Helper()
	raw, err := os.ReadFile("testdata/elementwise.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g gemmFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func matmul(t *testing.T, a, b *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	return out[0]
}

// §V1 PARITY: GEMM matches the NumPy-equivalent golden (f64 rtol 1e-12, f32 1e-5).
func TestGemmParity(t *testing.T) {
	g := loadGemm(t).Gemm

	a64 := tensor.FromFloat64(tensor.Shape(g.AShape), g.A)
	b64 := tensor.FromFloat64(tensor.Shape(g.BShape), g.B)
	c := matmul(t, a64, b64)
	if !c.Shape().Equal(tensor.Shape(g.Shape)) {
		t.Fatalf("shape %v, want %v", c.Shape(), g.Shape)
	}
	for i := range g.F64 {
		idx := tensor.Unravel(i, c.Shape())
		if !almostEqual(c.AtF64(idx...), g.F64[i], 1e-12, 1e-12) {
			t.Errorf("gemm f64[%d]: got %.17g want %.17g", i, c.AtF64(idx...), g.F64[i])
		}
	}

	af := make([]float32, len(g.A))
	for i, v := range g.A {
		af[i] = float32(v)
	}
	bf := make([]float32, len(g.B))
	for i, v := range g.B {
		bf[i] = float32(v)
	}
	c32 := matmul(t, tensor.FromFloat32(tensor.Shape(g.AShape), af), tensor.FromFloat32(tensor.Shape(g.BShape), bf))
	for i := range g.F32 {
		idx := tensor.Unravel(i, c32.Shape())
		if !almostEqual(c32.AtF64(idx...), g.F32[i], 1e-5, 1e-6) {
			t.Errorf("gemm f32[%d]: got %.9g want %.9g", i, c32.AtF64(idx...), g.F32[i])
		}
	}
}

// §V6: A·I == A (identity), a basic algebraic property.
func TestGemmIdentity(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	id := tensor.New(tensor.F64, tensor.Shape{3, 3})
	for i := range 3 {
		id.SetF64(1, i, i)
	}
	c := matmul(t, a, id)
	for i := range 2 {
		for j := range 3 {
			if c.AtF64(i, j) != a.AtF64(i, j) {
				t.Errorf("A·I mismatch at (%d,%d)", i, j)
			}
		}
	}
}

// Transposed operands work via views (no trans flags in the kernel).
func TestGemmTransposedView(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	// B stored as (2,3) then transposed to (3,2) view → A(2,3)·Bᵀ(3,2) = (2,2)
	bt := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 0, 1, 0, 1, 0})
	btv, _ := bt.Transpose(0, 1) // (3,2), non-contiguous
	c := matmul(t, a, btv)
	if !c.Shape().Equal(tensor.Shape{2, 2}) {
		t.Fatalf("shape %v, want (2,2)", c.Shape())
	}
	// row0 [1,2,3]·colB0[1,0,1]=1+0+3=4 ; ·colB1[0,1,0]=2
	if c.AtF64(0, 0) != 4 || c.AtF64(0, 1) != 2 {
		t.Errorf("transposed-view matmul wrong: %v %v", c.AtF64(0, 0), c.AtF64(0, 1))
	}
}

// §V10: f32 GEMM accumulates in f64 — large K stays accurate.
func TestGemmF64Accumulation(t *testing.T) {
	const k = 100_000
	a := make([]float32, k)
	b := make([]float32, k)
	for i := range a {
		a[i], b[i] = 0.1, 1.0
	}
	c := matmul(t, tensor.FromFloat32(tensor.Shape{1, k}, a), tensor.FromFloat32(tensor.Shape{k, 1}, b))
	exact := float64(k) * float64(float32(0.1))
	if math.Abs(c.AtF64(0, 0)-exact) > 1e-2 {
		t.Errorf("gemm f64-accum: got %.6f want ~%.6f", c.AtF64(0, 0), exact)
	}
}

func TestGemmErrors(t *testing.T) {
	// inner dim mismatch
	if _, err := backend.Execute(backend.NewContext(), backend.OpMatMul,
		[]*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{2, 3}), tensor.New(tensor.F64, tensor.Shape{4, 2})}, nil); err == nil {
		t.Error("inner-dim mismatch must error")
	}
	// rank error
	if _, err := backend.Execute(backend.NewContext(), backend.OpMatMul,
		[]*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{2, 3, 4}), tensor.New(tensor.F64, tensor.Shape{4, 2})}, nil); err == nil {
		t.Error("rank!=2 must error")
	}
	// K=0 → zero matrix, no panic
	z := matmul(t, tensor.New(tensor.F64, tensor.Shape{2, 0}), tensor.New(tensor.F64, tensor.Shape{0, 3}))
	if !z.Shape().Equal(tensor.Shape{2, 3}) || z.AtF64(1, 2) != 0 {
		t.Errorf("K=0 matmul: shape %v val %v", z.Shape(), z.AtF64(1, 2))
	}
}
