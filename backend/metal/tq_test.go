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

type metalTQCase struct {
	name       string
	qt         gguf.QuantType
	blockBytes int
	direct     func(*tensor.Tensor, []byte, int, int) (*tensor.Tensor, error)
	upload     func([]byte, int, int) (*metal.ResidentQWeight, error)
	toggle     func(bool) bool
}

func metalTQCases() []metalTQCase {
	return []metalTQCase{
		{"TQ1_0", gguf.TQ1_0, 54, metal.QMatMulTQ1_0, metal.UploadQWeightTQ1_0, metal.SetTQ1Cooperative},
		{"TQ2_0", gguf.TQ2_0, 66, metal.QMatMulTQ2_0, metal.UploadQWeightTQ2_0, metal.SetTQ2Cooperative},
	}
}

func syntheticTQ(format metalTQCase, n, k, seed int) []byte {
	blocks := n * (k / 256)
	raw := make([]byte, blocks*format.blockBytes)
	scales := [...]uint16{0x2800, 0xa800, 0x3000, 0x3400}
	for block := range blocks {
		base := block * format.blockBytes
		for i := range format.blockBytes - 2 {
			raw[base+i] = byte((block*139 + i*67 + seed*23) & 0xff)
		}
		//perfscan:ignore PS4001 strided trailing f16 field in a heterogeneous TQ fixture
		binary.LittleEndian.PutUint16(raw[base+format.blockBytes-2:], scales[(block+seed)%len(scales)])
	}
	return raw
}

func metalTQInputs(format metalTQCase, m, n, k int) (*tensor.Tensor, []byte) {
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i, value := range qmRand(m*k, 20260822+int(format.qt)) {
		x.Storage().F32()[i] = float32(value)
	}
	return x, syntheticTQ(format, n, k, 73)
}

func assertMetalTQClose(t *testing.T, got, want *tensor.Tensor, tolerance float64) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
	}
	maxRelative := 0.0
	for i, actual := range got.Storage().F32() {
		reference := want.Storage().F32()[i]
		denominator := max(1, math.Abs(float64(reference)))
		relative := math.Abs(float64(actual-reference)) / denominator
		maxRelative = max(maxRelative, relative)
		if relative > tolerance {
			t.Fatalf("element %d: got %g want %g relative=%g > %g", i, actual, reference, relative, tolerance)
		}
	}
	t.Logf("maximum relative difference %.3e", maxRelative)
}

func TestMetalQMatMulTQCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalTQCases() {
		t.Run(format.name, func(t *testing.T) {
			for _, shape := range []struct{ m, n, k int }{
				{1, 1, 256}, {1, 7, 512}, {4, 5, 1024}, {1, 64, 2048}, {2, 3, 5632},
			} {
				x, weight := metalTQInputs(format, shape.m, shape.n, shape.k)
				xBefore, weightBefore := slices.Clone(x.Storage().F32()), slices.Clone(weight)
				want, err := gguf.QMatMul(x, weight, format.qt, shape.n, shape.k)
				if err != nil {
					t.Fatal(err)
				}
				got, err := format.direct(x, weight, shape.n, shape.k)
				if err != nil {
					t.Fatalf("M=%d N=%d K=%d: %v", shape.m, shape.n, shape.k, err)
				}
				assertMetalTQClose(t, got, want, 1e-4)
				if !slices.Equal(xBefore, x.Storage().F32()) || !bytes.Equal(weightBefore, weight) {
					t.Fatalf("M=%d N=%d K=%d: kernel mutated an input", shape.m, shape.n, shape.k)
				}
			}
		})
	}
}

func TestMetalQMatMulTQCooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalTQCases() {
		t.Run(format.name, func(t *testing.T) {
			previous := format.toggle(false)
			defer format.toggle(previous)
			for _, shape := range []struct{ n, k int }{{1, 256}, {7, 512}, {8, 1024}, {3, 4096}, {64, 2048}} {
				x, weight := metalTQInputs(format, 1, shape.n, shape.k)
				format.toggle(false)
				scalar, err := format.direct(x, weight, shape.n, shape.k)
				if err != nil {
					t.Fatal(err)
				}
				format.toggle(true)
				cooperative, err := format.direct(x, weight, shape.n, shape.k)
				if err != nil {
					t.Fatal(err)
				}
				assertMetalTQClose(t, cooperative, scalar, 1e-4)
			}
		})
	}
}

func TestMetalQMatMulTQFloatingPointClass(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 3, 256
	for _, format := range metalTQCases() {
		t.Run(format.name, func(t *testing.T) {
			x := tensor.New(tensor.F32, tensor.Shape{m, k})
			for i := range x.Storage().F32() {
				x.Storage().F32()[i] = 1
			}
			x.Storage().F32()[0] = float32(math.Inf(1))
			x.Storage().F32()[k] = float32(math.Inf(-1))
			x.Storage().F32()[2*k] = float32(math.NaN())
			weight := make([]byte, n*format.blockBytes)
			fill := byte(0xff)
			if format.qt == gguf.TQ2_0 {
				fill = 0xaa
			}
			for row := range n {
				base := row * format.blockBytes
				for i := range format.blockBytes - 2 {
					weight[base+i] = fill
				}
				//perfscan:ignore PS4001 exact trailing-scale floating-point-class fixture
				binary.LittleEndian.PutUint16(weight[base+format.blockBytes-2:], 0x3c00)
			}
			got, err := format.direct(x, weight, n, k)
			if err != nil {
				t.Fatal(err)
			}
			for column := range n {
				if !math.IsInf(got.AtF64(0, column), 1) || !math.IsInf(got.AtF64(1, column), -1) || !math.IsNaN(got.AtF64(2, column)) {
					t.Fatalf("column %d classes = %g %g %g", column, got.AtF64(0, column), got.AtF64(1, column), got.AtF64(2, column))
				}
			}
		})
	}
}

func TestMetalQMatMulTQ2RawCodeThreeIsPlusTwo(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 1, 256
	x := tensor.New(tensor.F32, tensor.Shape{1, k})
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = 1
	}
	weight := make([]byte, 66)
	for i := range 64 {
		weight[i] = 0xff
	}
	binary.LittleEndian.PutUint16(weight[64:], 0x3c00)
	got, err := metal.QMatMulTQ2_0(x, weight, n, k)
	if err != nil {
		t.Fatal(err)
	}
	if value := got.AtF64(0, 0); value != 512 {
		t.Fatalf("all raw-code-3 dot = %g, want 512 (+2 per weight)", value)
	}
}

func TestMetalQMatMulTQDispatchAndValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalTQCases() {
		t.Run(format.name, func(t *testing.T) {
			x, weight := metalTQInputs(format, 1, 3, 512)
			got, err := format.direct(x, weight, 3, 512)
			if err != nil {
				t.Fatal(err)
			}
			want, err := gguf.QMatMul(x, weight, format.qt, 3, 512)
			if err != nil {
				t.Fatal(err)
			}
			assertMetalTQClose(t, got, want, 1e-4)
			if _, err := (metal.Backend{}).QMatMul(x, weight, uint32(format.qt), 3, 512); !errors.Is(err, backend.ErrQuantUnsupported) {
				t.Fatalf("generic host route error = %v, want ErrQuantUnsupported", err)
			}
			if _, err := (metal.Backend{}).UploadQuant(weight, uint32(format.qt), 3, 512); !errors.Is(err, backend.ErrQuantUnsupported) {
				t.Fatalf("generic resident route error = %v, want ErrQuantUnsupported", err)
			}
			if _, err := format.direct(tensor.New(tensor.F32, tensor.Shape{1, 257}), make([]byte, format.blockBytes), 1, 257); err == nil {
				t.Fatal("accepted K not divisible by 256")
			}
			if _, err := format.direct(x, weight[:len(weight)-1], 3, 512); err == nil {
				t.Fatal("accepted a truncated matrix")
			}
			if _, err := format.direct(tensor.New(tensor.F64, tensor.Shape{1, 512}), weight, 3, 512); err == nil {
				t.Fatal("accepted F64 activation")
			}
		})
	}
}

func TestMetalQMatMulTQRecorderResident(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 7, 512
	for _, format := range metalTQCases() {
		t.Run(format.name, func(t *testing.T) {
			x, weight := metalTQInputs(format, m, n, k)
			want, err := gguf.QMatMul(x, weight, format.qt, n, k)
			if err != nil {
				t.Fatal(err)
			}
			resident, err := format.upload(weight, n, k)
			if err != nil {
				t.Fatal(err)
			}
			defer resident.Close()
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
			if err := recorder.QMatMulResident(xb, resident, ob, m); err != nil {
				t.Fatal(err)
			}
			if err := recorder.Finish(); err != nil {
				t.Fatal(err)
			}
			out := tensor.New(tensor.F32, tensor.Shape{m, n})
			if err := ob.DownloadF32(out.Storage().F32()); err != nil {
				t.Fatal(err)
			}
			assertMetalTQClose(t, out, want, 1e-4)
		})
	}
}
