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

type metalCompactQuantCase struct {
	name       string
	qt         gguf.QuantType
	blockElems int
	blockBytes int
	direct     func(*tensor.Tensor, []byte, int, int) (*tensor.Tensor, error)
	upload     func([]byte, int, int) (*metal.ResidentQWeight, error)
	toggle     func(bool) bool
}

func metalCompactQuantCases() []metalCompactQuantCase {
	return []metalCompactQuantCase{
		{"Q1_0", gguf.Q1_0, 128, 18, metal.QMatMulQ1_0, metal.UploadQWeightQ1_0, metal.SetQ1Cooperative},
		{"MXFP4", gguf.MXFP4, 32, 17, metal.QMatMulMXFP4, metal.UploadQWeightMXFP4, metal.SetMXFP4Cooperative},
	}
}

func syntheticCompactQuant(format metalCompactQuantCase, n, k, seed int) []byte {
	blocks := n * (k / format.blockElems)
	raw := make([]byte, blocks*format.blockBytes)
	q1Scales := [...]uint16{0x2800, 0xa800, 0x3000, 0x3400}
	mxScales := [...]byte{119, 122, 125, 127}
	for block := range blocks {
		base := block * format.blockBytes
		if format.qt == gguf.Q1_0 {
			//perfscan:ignore PS4001 intentionally strided f16 scales in a heterogeneous Q1 fixture
			binary.LittleEndian.PutUint16(raw[base:], q1Scales[(block+seed)%len(q1Scales)])
			for i := 2; i < format.blockBytes; i++ {
				raw[base+i] = byte((block*139 + i*67 + seed*23) & 0xff)
			}
			continue
		}
		raw[base] = mxScales[(block+seed)%len(mxScales)]
		for i := 1; i < format.blockBytes; i++ {
			raw[base+i] = byte((block*139 + i*67 + seed*23) & 0xff)
		}
	}
	return raw
}

func metalCompactQuantInputs(format metalCompactQuantCase, m, n, k int) (*tensor.Tensor, []byte) {
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i, value := range qmRand(m*k, 20260822+int(format.qt)) {
		x.Storage().F32()[i] = float32(value)
	}
	return x, syntheticCompactQuant(format, n, k, 73)
}

func assertMetalCompactQuantClose(t *testing.T, got, want *tensor.Tensor, tolerance float64) {
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

func TestMetalCompactQuantCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalCompactQuantCases() {
		t.Run(format.name, func(t *testing.T) {
			for _, shape := range []struct{ m, n, k int }{
				{1, 1, 128}, {1, 7, 256}, {4, 5, 1024}, {1, 64, 2048}, {2, 3, 5632},
			} {
				x, weight := metalCompactQuantInputs(format, shape.m, shape.n, shape.k)
				xBefore, weightBefore := slices.Clone(x.Storage().F32()), slices.Clone(weight)
				want, err := gguf.QMatMul(x, weight, format.qt, shape.n, shape.k)
				if err != nil {
					t.Fatal(err)
				}
				got, err := format.direct(x, weight, shape.n, shape.k)
				if err != nil {
					t.Fatalf("M=%d N=%d K=%d: %v", shape.m, shape.n, shape.k, err)
				}
				assertMetalCompactQuantClose(t, got, want, 1e-4)
				if !slices.Equal(xBefore, x.Storage().F32()) || !bytes.Equal(weightBefore, weight) {
					t.Fatalf("M=%d N=%d K=%d: kernel mutated an input", shape.m, shape.n, shape.k)
				}
			}
		})
	}
}

func TestMetalCompactQuantCooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalCompactQuantCases() {
		t.Run(format.name, func(t *testing.T) {
			previous := format.toggle(false)
			defer format.toggle(previous)
			for _, shape := range []struct{ n, k int }{{1, 128}, {7, 256}, {8, 1024}, {3, 4096}, {64, 2048}} {
				x, weight := metalCompactQuantInputs(format, 1, shape.n, shape.k)
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
				assertMetalCompactQuantClose(t, cooperative, scalar, 1e-4)
			}
		})
	}
}

func TestMetalCompactQuantFloatingPointClass(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 3, 128
	for _, format := range metalCompactQuantCases() {
		t.Run(format.name, func(t *testing.T) {
			x := tensor.New(tensor.F32, tensor.Shape{m, k})
			for i := range x.Storage().F32() {
				x.Storage().F32()[i] = 1
			}
			x.Storage().F32()[0] = float32(math.Inf(1))
			x.Storage().F32()[k] = float32(math.Inf(-1))
			x.Storage().F32()[2*k] = float32(math.NaN())
			weight := make([]byte, n*(k/format.blockElems)*format.blockBytes)
			for block := range len(weight) / format.blockBytes {
				base := block * format.blockBytes
				if format.qt == gguf.Q1_0 {
					//perfscan:ignore PS4001 intentionally strided f16 scales in a floating-class fixture
					binary.LittleEndian.PutUint16(weight[base:], 0x3c00)
					for i := 2; i < format.blockBytes; i++ {
						weight[base+i] = 0xff
					}
				} else {
					weight[base] = 127
					for i := 1; i < format.blockBytes; i++ {
						weight[base+i] = 0x77
					}
				}
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

func TestMetalMXFP4ExactNibbleCodebook(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 1, 32
	x := tensor.New(tensor.F32, tensor.Shape{1, k})
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = float32(i + 1)
	}
	weight := make([]byte, 17)
	weight[0] = 127 // exact half scale
	values := [...]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
	var want float32
	for code := range 16 {
		weight[1+code] = byte(code | code<<4)
		want += float32(code+1) * 0.5 * values[code]
		want += float32(code+17) * 0.5 * values[code]
	}
	got, err := metal.QMatMulMXFP4(x, weight, n, k)
	if err != nil {
		t.Fatal(err)
	}
	if diff := math.Abs(float64(got.AtF64(0, 0) - float64(want))); diff > 1e-5 {
		t.Fatalf("all-code dot = %g, want %g (difference %g)", got.AtF64(0, 0), want, diff)
	}
}

func TestMetalCompactQuantDispatchAndValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, format := range metalCompactQuantCases() {
		t.Run(format.name, func(t *testing.T) {
			x, weight := metalCompactQuantInputs(format, 1, 3, 512)
			got, err := format.direct(x, weight, 3, 512)
			if err != nil {
				t.Fatal(err)
			}
			want, err := gguf.QMatMul(x, weight, format.qt, 3, 512)
			if err != nil {
				t.Fatal(err)
			}
			assertMetalCompactQuantClose(t, got, want, 1e-4)
			if _, err := (metal.Backend{}).QMatMul(x, weight, uint32(format.qt), 3, 512); !errors.Is(err, backend.ErrQuantUnsupported) {
				t.Fatalf("generic host route error = %v, want ErrQuantUnsupported", err)
			}
			if _, err := (metal.Backend{}).UploadQuant(weight, uint32(format.qt), 3, 512); !errors.Is(err, backend.ErrQuantUnsupported) {
				t.Fatalf("generic resident route error = %v, want ErrQuantUnsupported", err)
			}
			badK := format.blockElems + 1
			if _, err := format.direct(tensor.New(tensor.F32, tensor.Shape{1, badK}), make([]byte, format.blockBytes), 1, badK); err == nil {
				t.Fatalf("accepted K not divisible by %d", format.blockElems)
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

func TestMetalCompactQuantRecorderResident(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, n, k = 3, 7, 512
	for _, format := range metalCompactQuantCases() {
		t.Run(format.name, func(t *testing.T) {
			x, weight := metalCompactQuantInputs(format, m, n, k)
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
			assertMetalCompactQuantClose(t, out, want, 1e-4)
		})
	}
}
