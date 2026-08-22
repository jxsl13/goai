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

func syntheticIQ4XS(n, k, seed int) []byte {
	blocks := n * (k / 256)
	raw := make([]byte, blocks*136)
	scales := [...]uint16{0x2800, 0x2400, 0xa800, 0x2000} // 0.03125, 0.015625, -0.03125, 0.0078125
	for block := range blocks {
		base := block * 136
		//perfscan:ignore PS4001 strided f16 fields in heterogeneous IQ4_XS blocks cannot use a same-layout bulk copy
		binary.LittleEndian.PutUint16(raw[base:], scales[(block+seed)%len(scales)])
		var scalesH uint16
		for sb := range 8 {
			encoded := uint16((block*7 + sb*11 + seed*13) & 63)
			scalesH |= (encoded >> 4) << (2 * sb)
			raw[base+4+sb/2] |= byte(encoded&15) << (4 * (sb % 2))
		}
		binary.LittleEndian.PutUint16(raw[base+2:], scalesH)
		for i := 8; i < 136; i++ {
			raw[base+i] = byte((block*17 + i*29 + seed*11) & 0xff)
		}
	}
	return raw
}

func iq4XSInputs(m, n, k int) (*tensor.Tensor, []byte) {
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i, value := range qmRand(m*k, 91) {
		x.Storage().F32()[i] = float32(value)
	}
	return x, syntheticIQ4XS(n, k, 92)
}

func assertIQ4XSClose(t *testing.T, got, want *tensor.Tensor, tol float64) {
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
	t.Logf("IQ4_XS max relative difference %.3e", maxRel)
}

func TestMetalQMatMulIQ4XSCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, tc := range []struct{ m, n, k int }{
		{1, 1, 256}, {1, 7, 512}, {4, 5, 1024}, {1, 64, 2048}, {2, 3, 5632},
	} {
		x, wq := iq4XSInputs(tc.m, tc.n, tc.k)
		xBefore := slices.Clone(x.Storage().F32())
		wBefore := slices.Clone(wq)
		want, err := gguf.QMatMul(x, wq, gguf.IQ4_XS, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulIQ4_XS(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatalf("M=%d N=%d K=%d: %v", tc.m, tc.n, tc.k, err)
		}
		assertIQ4XSClose(t, got, want, 1e-4)
		if !slices.Equal(xBefore, x.Storage().F32()) {
			t.Fatalf("M=%d N=%d K=%d: activation mutated", tc.m, tc.n, tc.k)
		}
		if !bytes.Equal(wBefore, wq) {
			t.Fatalf("M=%d N=%d K=%d: weight bytes mutated", tc.m, tc.n, tc.k)
		}
	}
}

func TestMetalQMatMulIQ4XSCooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	previous := metal.SetIQ4XSCooperative(false)
	defer metal.SetIQ4XSCooperative(previous)
	for _, tc := range []struct{ n, k int }{{1, 256}, {7, 512}, {8, 1024}, {3, 4096}, {64, 2048}} {
		x, wq := iq4XSInputs(1, tc.n, tc.k)
		metal.SetIQ4XSCooperative(false)
		scalar, err := metal.QMatMulIQ4_XS(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		metal.SetIQ4XSCooperative(true)
		cooperative, err := metal.QMatMulIQ4_XS(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		assertIQ4XSClose(t, cooperative, scalar, 1e-4)
	}
}

func TestMetalQMatMulIQ4XSFloatingPointClass(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 3, 256
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = 1
	}
	x.Storage().F32()[0] = float32(math.Inf(1))
	x.Storage().F32()[k] = float32(math.Inf(-1))
	x.Storage().F32()[2*k] = float32(math.NaN())
	// d=1, encoded sub-scale 33 means signed scale 1, and code 8 is exactly one.
	wq := make([]byte, n*136)
	for row := range n {
		base := row * 136
		//perfscan:ignore PS4001 strided f16 fields in heterogeneous IQ4_XS blocks cannot use a same-layout bulk copy
		binary.LittleEndian.PutUint16(wq[base:], 0x3c00)
		binary.LittleEndian.PutUint16(wq[base+2:], 0xaaaa)
		for i := 4; i < 8; i++ {
			wq[base+i] = 0x11
		}
		for i := 8; i < 136; i++ {
			wq[base+i] = 0x88
		}
	}
	got, err := metal.QMatMulIQ4_XS(x, wq, n, k)
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

func TestMetalQMatMulIQ4XSDispatchAndValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x, wq := iq4XSInputs(1, 3, 512)
	got, err := metal.QMatMulIQ4_XS(x, wq, 3, 512)
	if err != nil {
		t.Fatal(err)
	}
	want, err := gguf.QMatMul(x, wq, gguf.IQ4_XS, 3, 512)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (metal.Backend{}).QMatMul(x, wq, uint32(gguf.IQ4_XS), 3, 512); !errors.Is(err, backend.ErrQuantUnsupported) {
		t.Fatalf("generic host route error = %v, want ErrQuantUnsupported", err)
	}
	if _, err := (metal.Backend{}).UploadQuant(wq, uint32(gguf.IQ4_XS), 3, 512); !errors.Is(err, backend.ErrQuantUnsupported) {
		t.Fatalf("generic resident host route error = %v, want ErrQuantUnsupported", err)
	}
	assertIQ4XSClose(t, got, want, 1e-4)
	if _, err := metal.QMatMulIQ4_XS(tensor.New(tensor.F32, tensor.Shape{1, 257}), make([]byte, 136), 1, 257); err == nil {
		t.Fatal("accepted K not divisible by 256")
	}
	if _, err := metal.QMatMulIQ4_XS(x, wq[:len(wq)-1], 3, 512); err == nil {
		t.Fatal("accepted truncated IQ4_XS matrix")
	}
	if _, err := metal.QMatMulIQ4_XS(tensor.New(tensor.F64, tensor.Shape{1, 512}), wq, 3, 512); err == nil {
		t.Fatal("accepted F64 activation")
	}
}

func TestMetalQMatMulIQ4XSRecorderResident(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 7, 512
	x, wq := iq4XSInputs(m, n, k)
	want, err := gguf.QMatMul(x, wq, gguf.IQ4_XS, n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw, err := metal.UploadQWeightIQ4_XS(wq, n, k)
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
	assertIQ4XSClose(t, out, want, 1e-4)
}
