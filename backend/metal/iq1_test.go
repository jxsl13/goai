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

type metalIQ1Case struct {
	name       string
	qt         gguf.QuantType
	blockBytes int
	synthetic  func(n, k, seed int) []byte
	direct     func(*tensor.Tensor, []byte, int, int) (*tensor.Tensor, error)
	upload     func([]byte, int, int) (*metal.ResidentQWeight, error)
	toggle     func(bool) bool
}

func metalIQ1Cases() []metalIQ1Case {
	return []metalIQ1Case{
		{"IQ1_S", gguf.IQ1_S, 50, syntheticIQ1S, metal.QMatMulIQ1_S, metal.UploadQWeightIQ1_S, metal.SetIQ1SCooperative},
		{"IQ1_M", gguf.IQ1_M, 56, syntheticIQ1M, metal.QMatMulIQ1_M, metal.UploadQWeightIQ1_M, metal.SetIQ1MCooperative},
	}
}

func syntheticIQ1S(n, k, seed int) []byte {
	blocks := n * (k / 256)
	raw := make([]byte, blocks*50)
	scales := [...]uint16{0x2800, 0xa800, 0x3000, 0x3400}
	for block := range blocks {
		base := block * 50
		//perfscan:ignore PS4001 strided f16 field in a heterogeneous IQ1_S fixture
		binary.LittleEndian.PutUint16(raw[base:], scales[(block+seed)%len(scales)])
		for i := 2; i < 50; i++ {
			raw[base+i] = byte((block*139 + i*67 + seed*23) & 0xff)
		}
	}
	return raw
}

func syntheticIQ1M(n, k, seed int) []byte {
	blocks := n * (k / 256)
	raw := make([]byte, blocks*56)
	scales := [...]uint16{0x2800, 0xa800, 0x3000, 0x3400}
	for block := range blocks {
		base := block * 56
		for i := range 48 {
			raw[base+i] = byte((block*131 + i*61 + seed*17) & 0xff)
		}
		d := scales[(block+seed)%len(scales)]
		for word := range 4 {
			low := uint16((block*977 + word*313 + seed*83) & 0x0fff)
			top := (d >> (4 * word) & 0x0f) << 12
			//perfscan:ignore PS4001 split-f16 fixture words are intentionally strided
			binary.LittleEndian.PutUint16(raw[base+48+word*2:], low|top)
		}
	}
	return raw
}

func metalIQ1Inputs(tc metalIQ1Case, m, n, k int) (*tensor.Tensor, []byte) {
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i, value := range qmRand(m*k, 20260822+int(tc.qt)) {
		x.Storage().F32()[i] = float32(value)
	}
	return x, tc.synthetic(n, k, 73)
}

func assertMetalIQ1Close(t *testing.T, got, want *tensor.Tensor, tolerance float64) {
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

func TestMetalQMatMulIQ1CrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalIQ1Cases() {
		t.Run(format.name, func(t *testing.T) {
			for _, shape := range []struct{ m, n, k int }{
				{1, 1, 256}, {1, 7, 512}, {4, 5, 1024}, {1, 64, 2048}, {2, 3, 5632},
			} {
				x, weight := metalIQ1Inputs(format, shape.m, shape.n, shape.k)
				xBefore, weightBefore := slices.Clone(x.Storage().F32()), slices.Clone(weight)
				want, err := gguf.QMatMul(x, weight, format.qt, shape.n, shape.k)
				if err != nil {
					t.Fatal(err)
				}
				got, err := format.direct(x, weight, shape.n, shape.k)
				if err != nil {
					t.Fatalf("M=%d N=%d K=%d: %v", shape.m, shape.n, shape.k, err)
				}
				assertMetalIQ1Close(t, got, want, 1e-4)
				if !slices.Equal(xBefore, x.Storage().F32()) || !bytes.Equal(weightBefore, weight) {
					t.Fatalf("M=%d N=%d K=%d: kernel mutated an input", shape.m, shape.n, shape.k)
				}
			}
		})
	}
}

func TestMetalQMatMulIQ1CooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalIQ1Cases() {
		t.Run(format.name, func(t *testing.T) {
			previous := format.toggle(false)
			defer format.toggle(previous)
			for _, shape := range []struct{ n, k int }{{1, 256}, {7, 512}, {8, 1024}, {3, 4096}, {64, 2048}} {
				x, weight := metalIQ1Inputs(format, 1, shape.n, shape.k)
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
				assertMetalIQ1Close(t, cooperative, scalar, 1e-4)
			}
		})
	}
}

func TestMetalQMatMulIQ1FloatingPointClass(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 3, 256
	for _, format := range metalIQ1Cases() {
		t.Run(format.name, func(t *testing.T) {
			x, weight := metalIQ1Inputs(format, m, n, k)
			for i := range x.Storage().F32() {
				x.Storage().F32()[i] = 1
			}
			x.Storage().F32()[0] = float32(math.Inf(1))
			x.Storage().F32()[k] = float32(math.Inf(-1))
			x.Storage().F32()[2*k] = float32(math.NaN())
			want, err := gguf.QMatMul(x, weight, format.qt, n, k)
			if err != nil {
				t.Fatal(err)
			}
			got, err := format.direct(x, weight, n, k)
			if err != nil {
				t.Fatal(err)
			}
			for i, value := range got.Storage().F32() {
				reference := want.Storage().F32()[i]
				if math.IsNaN(float64(value)) != math.IsNaN(float64(reference)) ||
					math.IsInf(float64(value), 1) != math.IsInf(float64(reference), 1) ||
					math.IsInf(float64(value), -1) != math.IsInf(float64(reference), -1) {
					t.Fatalf("element %d class = %g, want %g", i, value, reference)
				}
			}
		})
	}
}

func TestMetalQMatMulIQ1DispatchAndValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalIQ1Cases() {
		t.Run(format.name, func(t *testing.T) {
			x, weight := metalIQ1Inputs(format, 1, 3, 512)
			if _, err := format.direct(x, weight, 3, 512); err != nil {
				t.Fatal(err)
			}
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

func TestMetalQMatMulIQ1RecorderResident(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 7, 512
	for _, format := range metalIQ1Cases() {
		t.Run(format.name, func(t *testing.T) {
			x, weight := metalIQ1Inputs(format, m, n, k)
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
			assertMetalIQ1Close(t, out, want, 1e-4)
		})
	}
}
