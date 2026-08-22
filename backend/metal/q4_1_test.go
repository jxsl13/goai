//go:build darwin && cgo

package metal_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

func q41Inputs(m, n, k int) (*tensor.Tensor, []byte) {
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i, value := range qmRand(m*k, 71) {
		x.Storage().F32()[i] = float32(value)
	}
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 72))
	wq, err := gguf.Quantize(w, gguf.Q4_1)
	if err != nil {
		panic(err)
	}
	return x, wq
}

func assertQ41Close(t *testing.T, got, want *tensor.Tensor, tol float64) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
	}
	var maxRel float64
	for i, g := range got.Storage().F32() {
		w := want.Storage().F32()[i]
		den := math.Abs(float64(w))
		if den < 1 {
			den = 1
		}
		rel := math.Abs(float64(g-w)) / den
		if rel > maxRel {
			maxRel = rel
		}
		if rel > tol {
			t.Fatalf("element %d: got %g want %g relative=%g > %g", i, g, w, rel, tol)
		}
	}
	t.Logf("Q4_1 max relative difference %.3e", maxRel)
}

// Exact GGUF type-3 bytes are decoded by Metal; only f32 versus f64 accumulation may differ.
func TestMetalQMatMulQ4_1CrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, tc := range []struct{ m, n, k int }{
		{1, 1, 32}, {1, 7, 256}, {4, 5, 512}, {1, 64, 2048}, {2, 3, 4096},
	} {
		x, wq := q41Inputs(tc.m, tc.n, tc.k)
		xBefore := slices.Clone(x.Storage().F32())
		wBefore := slices.Clone(wq)
		want, err := gguf.QMatMul(x, wq, gguf.Q4_1, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ4_1(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatalf("M=%d N=%d K=%d: %v", tc.m, tc.n, tc.k, err)
		}
		assertQ41Close(t, got, want, 2e-5)
		for i, value := range xBefore {
			if x.Storage().F32()[i] != value {
				t.Fatalf("M=%d N=%d K=%d: activation %d mutated", tc.m, tc.n, tc.k, i)
			}
		}
		for i, value := range wBefore {
			if wq[i] != value {
				t.Fatalf("M=%d N=%d K=%d: weight byte %d mutated", tc.m, tc.n, tc.k, i)
			}
		}
	}
}

func TestMetalQMatMulQ4_1CooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	previous := metal.SetQ4_1Cooperative(false)
	defer metal.SetQ4_1Cooperative(previous)
	for _, tc := range []struct{ n, k int }{{1, 32}, {7, 256}, {8, 512}, {3, 4096}, {64, 2048}} {
		x, wq := q41Inputs(1, tc.n, tc.k)
		metal.SetQ4_1Cooperative(false)
		scalar, err := metal.QMatMulQ4_1(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		metal.SetQ4_1Cooperative(true)
		cooperative, err := metal.QMatMulQ4_1(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		assertQ41Close(t, cooperative, scalar, 2e-5)
	}
}

func TestMetalQMatMulQ4_1FloatingPointClass(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 3, 32
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = 1
	}
	x.Storage().F32()[0] = float32(math.Inf(1))
	x.Storage().F32()[k] = float32(math.Inf(-1))
	x.Storage().F32()[2*k] = float32(math.NaN())
	// d=0, m=1 makes every Q4_1 weight exactly one regardless of the nibble plane.
	wq := make([]byte, n*20)
	for row := range n {
		wq[row*20+2], wq[row*20+3] = 0, 0x3c
	}
	got, err := metal.QMatMulQ4_1(x, wq, n, k)
	if err != nil {
		t.Fatal(err)
	}
	for col := range n {
		if !math.IsInf(got.AtF64(0, col), 1) {
			t.Fatalf("positive infinity row col %d = %g", col, got.AtF64(0, col))
		}
		if !math.IsInf(got.AtF64(1, col), -1) {
			t.Fatalf("negative infinity row col %d = %g", col, got.AtF64(1, col))
		}
		if !math.IsNaN(got.AtF64(2, col)) {
			t.Fatalf("NaN row col %d = %g", col, got.AtF64(2, col))
		}
	}
}

func TestMetalQMatMulQ4_1DispatchAndValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x, wq := q41Inputs(1, 3, 64)
	got, err := metal.QMatMulQ4_1(x, wq, 3, 64)
	if err != nil {
		t.Fatal(err)
	}
	want, err := gguf.QMatMul(x, wq, gguf.Q4_1, 3, 64)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (metal.Backend{}).QMatMul(x, wq, uint32(gguf.Q4_1), 3, 64); !errors.Is(err, backend.ErrQuantUnsupported) {
		t.Fatalf("generic host route error = %v, want ErrQuantUnsupported", err)
	}
	assertQ41Close(t, got, want, 2e-5)
	if _, err := metal.QMatMulQ4_1(tensor.New(tensor.F32, tensor.Shape{1, 33}), make([]byte, 20), 1, 33); err == nil {
		t.Fatal("accepted K not divisible by 32")
	}
	if _, err := metal.QMatMulQ4_1(x, wq[:len(wq)-1], 3, 64); err == nil {
		t.Fatal("accepted truncated Q4_1 matrix")
	}
	if _, err := metal.QMatMulQ4_1(tensor.New(tensor.F64, tensor.Shape{1, 64}), wq, 3, 64); err == nil {
		t.Fatal("accepted F64 activation")
	}
}

func TestMetalQMatMulQ4_1RecorderResident(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 7, 256
	x, wq := q41Inputs(m, n, k)
	want, err := gguf.QMatMul(x, wq, gguf.Q4_1, n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw, err := metal.UploadQWeightQ4_1(wq, n, k)
	if err != nil {
		t.Fatal(err)
	}
	defer rw.Close()
	xb, err := metal.NewDeviceBufferF32(x.Storage().F32())
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Release()
	ob, err := metal.NewDeviceBufferF32(make([]float32, m*n))
	if err != nil {
		t.Fatal(err)
	}
	defer ob.Release()
	recorder, err := metal.NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer recorder.Free()
	if err := recorder.QMatMulResident(xb, rw, ob, m); err != nil {
		t.Fatal(err)
	}
	if err := recorder.Finish(); err != nil {
		t.Fatal(err)
	}
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if err := ob.DownloadF32(out.Storage().F32()); err != nil {
		t.Fatal(err)
	}
	assertQ41Close(t, out, want, 2e-5)
}
