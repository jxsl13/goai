//go:build arm64

package cpu

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func TestAbsF32Arm64ExactAllLengths(t *testing.T) {
	edges := []uint32{
		0x00000000, 0x80000000,
		0x00000001, 0x80000001,
		0x007fffff, 0x807fffff,
		0x00800000, 0x80800000,
		0x3f800000, 0xbf800000,
		0x7f7fffff, 0xff7fffff,
		0x7f800000, 0xff800000,
		0x7f800001, 0xff800001,
		0x7fa00001, 0xffa00001,
		0x7fc00000, 0xffc00000,
		0x7fffffff, 0xffffffff,
	}
	for n := 0; n <= 257; n++ {
		srcBacking := make([]float32, n+3)
		dstBacking := make([]float32, n+7)
		src := srcBacking[1 : 1+n]
		dst := dstBacking[3 : 3+n]
		state := uint32(0x9e3779b9)
		for i := range src {
			bits := edges[i%len(edges)]
			if i >= len(edges) {
				state ^= state << 13
				state ^= state >> 17
				state ^= state << 5
				bits = state
			}
			src[i] = math.Float32frombits(bits)
			dst[i] = math.Float32frombits(0x7fc0dead)
		}

		absF32(dst, src)
		for i, value := range src {
			want := math.Float32bits(float32(math.Abs(float64(value))))
			if got := math.Float32bits(dst[i]); got != want {
				t.Fatalf("n=%d i=%d input=%08x: got %08x, want %08x", n, i, math.Float32bits(value), got, want)
			}
		}
	}
}

func TestAbsF32Arm64ExactInPlace(t *testing.T) {
	values := make([]float32, 257)
	want := make([]uint32, len(values))
	state := uint32(0x9e3779b9)
	for i := range values {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		values[i] = math.Float32frombits(state)
		want[i] = math.Float32bits(float32(math.Abs(float64(values[i]))))
	}

	absF32(values, values)
	for i, value := range values {
		if got := math.Float32bits(value); got != want[i] {
			t.Fatalf("i=%d: got %08x, want %08x", i, got, want[i])
		}
	}
}

func TestAbsF32ParallelThresholdGeometry(t *testing.T) {
	previous := runtime.GOMAXPROCS(2)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })

	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend unavailable")
	}
	ctx := backend.NewContext().WithBackend(be)
	for _, n := range []int{absF32ParallelThreshold - 1, absF32ParallelThreshold} {
		values := make([]float32, n)
		state := uint32(0x9e3779b9)
		for i := range values {
			state ^= state << 13
			state ^= state >> 17
			state ^= state << 5
			values[i] = math.Float32frombits(state)
		}
		x := tensor.FromFloat32(tensor.Shape{n}, values)
		got, err := backend.Execute(ctx, backend.OpAbs, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		out := got[0].Storage().F32()
		for i, value := range values {
			want := math.Float32bits(float32(math.Abs(float64(value))))
			if bits := math.Float32bits(out[i]); bits != want {
				t.Fatalf("n=%d i=%d input=%08x: got %08x, want %08x", n, i, math.Float32bits(value), bits, want)
			}
		}
	}
}
