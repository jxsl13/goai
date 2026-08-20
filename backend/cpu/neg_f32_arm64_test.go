//go:build arm64

package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func TestNegKernelCPUF32ExactParallel(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend unavailable")
	}
	n := negF32ParallelThreshold + 257
	x := tensor.New(tensor.F32, tensor.Shape{n})
	state := uint32(0x243f6a88)
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = math.Float32frombits(nextNegF32TestBits(&state, i))
	}

	out, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpNeg, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range x.Storage().F32() {
		want := math.Float32bits(value) ^ uint32(1<<31)
		if got := math.Float32bits(out[0].Storage().F32()[i]); got != want {
			t.Fatalf("i=%d input=%08x: got %08x, want %08x", i, math.Float32bits(value), got, want)
		}
	}
}
