//go:build darwin && cgo

package metal_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

func syntheticIQ4NL(n, k, seed int) []byte {
	blocks := n * (k / 32)
	raw := make([]byte, blocks*18)
	scales := [...]uint16{0x3c00, 0x3800, 0xbc00, 0x3400} // 1, 0.5, -1, 0.25
	for block := range blocks {
		base := block * 18
		binary.LittleEndian.PutUint16(raw[base:], scales[(block+seed)%len(scales)])
		for i := 2; i < 18; i++ {
			raw[base+i] = byte((block*17 + i*29 + seed*11) & 0xff)
		}
	}
	return raw
}

func iq4NLInputs(m, n, k int) (*tensor.Tensor, []byte) {
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i, value := range qmRand(m*k, 81) {
		x.Storage().F32()[i] = float32(value)
	}
	return x, syntheticIQ4NL(n, k, 82)
}

func assertIQ4NLClose(t *testing.T, got, want *tensor.Tensor, tol float64) {
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
	t.Logf("IQ4_NL max relative difference %.3e", maxRel)
}

func TestMetalQMatMulIQ4NLCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, tc := range []struct{ m, n, k int }{
		{1, 1, 32}, {1, 7, 256}, {4, 5, 512}, {1, 64, 2048}, {2, 3, 4096},
	} {
		x, wq := iq4NLInputs(tc.m, tc.n, tc.k)
		xBefore := slices.Clone(x.Storage().F32())
		wBefore := slices.Clone(wq)
		want, err := gguf.QMatMul(x, wq, gguf.IQ4_NL, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulIQ4_NL(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatalf("M=%d N=%d K=%d: %v", tc.m, tc.n, tc.k, err)
		}
		assertIQ4NLClose(t, got, want, 2e-5)
		if !slices.Equal(xBefore, x.Storage().F32()) {
			t.Fatalf("M=%d N=%d K=%d: activation mutated", tc.m, tc.n, tc.k)
		}
		if !bytes.Equal(wBefore, wq) {
			t.Fatalf("M=%d N=%d K=%d: weight bytes mutated", tc.m, tc.n, tc.k)
		}
	}
}

func TestMetalQMatMulIQ4NLCooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	previous := metal.SetIQ4NLCooperative(false)
	defer metal.SetIQ4NLCooperative(previous)
	for _, tc := range []struct{ n, k int }{{1, 32}, {7, 256}, {8, 512}, {3, 4096}, {64, 2048}} {
		x, wq := iq4NLInputs(1, tc.n, tc.k)
		metal.SetIQ4NLCooperative(false)
		scalar, err := metal.QMatMulIQ4_NL(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		metal.SetIQ4NLCooperative(true)
		cooperative, err := metal.QMatMulIQ4_NL(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		assertIQ4NLClose(t, cooperative, scalar, 2e-5)
	}
}

func TestMetalQMatMulIQ4NLFloatingPointClass(t *testing.T) {
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
	// d=1 and code 8 make every IQ4_NL weight exactly one.
	wq := make([]byte, n*18)
	for row := range n {
		binary.LittleEndian.PutUint16(wq[row*18:], 0x3c00)
		for i := 2; i < 18; i++ {
			wq[row*18+i] = 0x88
		}
	}
	got, err := metal.QMatMulIQ4_NL(x, wq, n, k)
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

func TestMetalQMatMulIQ4NLDispatchAndValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x, wq := iq4NLInputs(1, 3, 64)
	got, err := metal.QMatMulIQ4_NL(x, wq, 3, 64)
	if err != nil {
		t.Fatal(err)
	}
	want, err := gguf.QMatMul(x, wq, gguf.IQ4_NL, 3, 64)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (metal.Backend{}).QMatMul(x, wq, uint32(gguf.IQ4_NL), 3, 64); !errors.Is(err, backend.ErrQuantUnsupported) {
		t.Fatalf("generic host route error = %v, want ErrQuantUnsupported", err)
	}
	if _, err := (metal.Backend{}).UploadQuant(wq, uint32(gguf.IQ4_NL), 3, 64); !errors.Is(err, backend.ErrQuantUnsupported) {
		t.Fatalf("generic resident host route error = %v, want ErrQuantUnsupported", err)
	}
	assertIQ4NLClose(t, got, want, 2e-5)
	if _, err := metal.QMatMulIQ4_NL(tensor.New(tensor.F32, tensor.Shape{1, 33}), make([]byte, 18), 1, 33); err == nil {
		t.Fatal("accepted K not divisible by 32")
	}
	if _, err := metal.QMatMulIQ4_NL(x, wq[:len(wq)-1], 3, 64); err == nil {
		t.Fatal("accepted truncated IQ4_NL matrix")
	}
	if _, err := metal.QMatMulIQ4_NL(tensor.New(tensor.F64, tensor.Shape{1, 64}), wq, 3, 64); err == nil {
		t.Fatal("accepted F64 activation")
	}
}

func TestMetalQMatMulIQ4NLRecorderResident(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 7, 256
	x, wq := iq4NLInputs(m, n, k)
	want, err := gguf.QMatMul(x, wq, gguf.IQ4_NL, n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw, err := metal.UploadQWeightIQ4_NL(wq, n, k)
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
	assertIQ4NLClose(t, out, want, 2e-5)
}
