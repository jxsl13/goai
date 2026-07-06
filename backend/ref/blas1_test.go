package ref

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type blasFile struct {
	Blas struct {
		A     []float64 `json:"a"`
		B     []float64 `json:"b"`
		Alpha float64   `json:"alpha"`
		Dot   struct {
			F64 float64 `json:"f64"`
			F32 float64 `json:"f32"`
		} `json:"dot"`
		Nrm2 struct {
			F64 float64 `json:"f64"`
			F32 float64 `json:"f32"`
		} `json:"nrm2"`
		Axpy struct {
			F64 []float64 `json:"f64"`
			F32 []float64 `json:"f32"`
		} `json:"axpy"`
	} `json:"blas"`
}

func loadBlas(t *testing.T) blasFile {
	t.Helper()
	raw, err := os.ReadFile("testdata/elementwise.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g blasFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func vecF64(d []float64) *tensor.Tensor { return tensor.FromFloat64(tensor.Shape{len(d)}, d) }
func vecF32(d []float64) *tensor.Tensor {
	f := make([]float32, len(d))
	for i, v := range d {
		f[i] = float32(v)
	}
	return tensor.FromFloat32(tensor.Shape{len(d)}, f)
}

func scalarOut(t *testing.T, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) float64 {
	t.Helper()
	out, err := backend.Execute(backend.NewContext(), op, ins, attrs)
	if err != nil {
		t.Fatalf("%v: %v", op, err)
	}
	return out[0].AtF64()
}

// §V1 PARITY + §V10: dot/nrm2/axpy match golden within tolerance.
func TestBlas1Parity(t *testing.T) {
	g := loadBlas(t).Blas

	if got := scalarOut(t, backend.OpDot, nil, vecF64(g.A), vecF64(g.B)); !almostEqual(got, g.Dot.F64, 1e-12, 1e-12) {
		t.Errorf("dot f64: got %.17g want %.17g", got, g.Dot.F64)
	}
	if got := scalarOut(t, backend.OpDot, nil, vecF32(g.A), vecF32(g.B)); !almostEqual(got, g.Dot.F32, 1e-5, 1e-6) {
		t.Errorf("dot f32: got %.9g want %.9g", got, g.Dot.F32)
	}
	if got := scalarOut(t, backend.OpNrm2, nil, vecF64(g.A)); !almostEqual(got, g.Nrm2.F64, 1e-12, 1e-12) {
		t.Errorf("nrm2 f64: got %.17g want %.17g", got, g.Nrm2.F64)
	}
	if got := scalarOut(t, backend.OpNrm2, nil, vecF32(g.A)); !almostEqual(got, g.Nrm2.F32, 1e-5, 1e-6) {
		t.Errorf("nrm2 f32: got %.9g want %.9g", got, g.Nrm2.F32)
	}

	attrs := backend.Attrs{"alpha": g.Alpha}
	out, err := backend.Execute(backend.NewContext(), backend.OpAXPY, []*tensor.Tensor{vecF64(g.A), vecF64(g.B)}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Axpy.F64 {
		if !almostEqual(out[0].AtF64(i), g.Axpy.F64[i], 1e-12, 1e-12) {
			t.Errorf("axpy f64[%d]: got %.17g want %.17g", i, out[0].AtF64(i), g.Axpy.F64[i])
		}
	}
	out32, _ := backend.Execute(backend.NewContext(), backend.OpAXPY, []*tensor.Tensor{vecF32(g.A), vecF32(g.B)}, attrs)
	for i := range g.Axpy.F32 {
		if !almostEqual(out32[0].AtF64(i), g.Axpy.F32[i], 1e-5, 1e-6) {
			t.Errorf("axpy f32[%d]: got %.9g want %.9g", i, out32[0].AtF64(i), g.Axpy.F32[i])
		}
	}
}

// §G4/§V9: nrm2 must not overflow for large magnitudes where naive sum-of-squares
// would become +Inf.
func TestNrm2OverflowSafe(t *testing.T) {
	x := vecF64([]float64{1e200, 1e200}) // naive: (1e200)^2 = 1e400 → +Inf
	got := scalarOut(t, backend.OpNrm2, nil, x)
	want := 1e200 * math.Sqrt2
	if math.IsInf(got, 0) || !almostEqual(got, want, 1e-12, 0) {
		t.Errorf("nrm2 overflow-safe: got %g, want %g", got, want)
	}
	// and underflow-safe for tiny magnitudes
	tiny := vecF64([]float64{1e-200, 1e-200})
	gt := scalarOut(t, backend.OpNrm2, nil, tiny)
	if wt := 1e-200 * math.Sqrt2; !almostEqual(gt, wt, 1e-12, 0) {
		t.Errorf("nrm2 underflow-safe: got %g, want %g", gt, wt)
	}
}

// §V10: f32 dot accumulates in f64 — large-K stays accurate.
func TestDotF64Accumulation(t *testing.T) {
	const n = 100_000
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i], b[i] = 0.1, 1.0
	}
	got := scalarOut(t, backend.OpDot, nil,
		tensor.FromFloat32(tensor.Shape{n}, a), tensor.FromFloat32(tensor.Shape{n}, b))
	// exact: n * f32(0.1) * 1.0, accumulated in f64
	exact := float64(n) * float64(float32(0.1))
	if math.Abs(got-exact) > 1e-2 {
		t.Errorf("dot f64-accum: got %.6f want ~%.6f", got, exact)
	}
}

func TestBlas1Edge(t *testing.T) {
	// empty vector: dot → 0, nrm2 → 0
	e := tensor.New(tensor.F64, tensor.Shape{0})
	if got := scalarOut(t, backend.OpDot, nil, e, e); got != 0 {
		t.Errorf("empty dot = %v, want 0", got)
	}
	if got := scalarOut(t, backend.OpNrm2, nil, e); got != 0 {
		t.Errorf("empty nrm2 = %v, want 0", got)
	}
	// shape mismatch → error
	if _, err := backend.Execute(backend.NewContext(), backend.OpDot,
		[]*tensor.Tensor{vecF64([]float64{1, 2}), vecF64([]float64{1, 2, 3})}, nil); err == nil {
		t.Error("dot shape mismatch must error")
	}
	// alpha default 1
	out, _ := backend.Execute(backend.NewContext(), backend.OpAXPY,
		[]*tensor.Tensor{vecF64([]float64{1, 2}), vecF64([]float64{10, 20})}, nil)
	if out[0].AtF64(0) != 11 || out[0].AtF64(1) != 22 {
		t.Errorf("axpy default alpha=1: got %v,%v", out[0].AtF64(0), out[0].AtF64(1))
	}
}
