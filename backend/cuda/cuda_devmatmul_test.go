//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Device×device matmul (both operands resident activations, e.g. attention
// scores·V) must match the reference within the §V11 GPU-f32 tolerance.
func TestCUDADeviceMatMulMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ m, k, n int }{{2, 3, 4}, {8, 64, 8}, {33, 65, 17}, {8, 4096, 8}} {
		a := bench.RandF32(tensor.Shape{s.m, s.k}, 1)
		b := bench.RandF32(tensor.Shape{s.k, s.n}, 2)
		da, _ := cuda.UploadF32(a)
		db, _ := cuda.UploadF32(b)
		dc, err := da.MatMul(db)
		if err != nil {
			t.Fatalf("%v: %v", s, err)
		}
		got, _ := dc.ToHost()
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		assertClose(t, got, gr[0], 1e-6*math.Sqrt(float64(s.k)), s)
		da.Free()
		db.Free()
		dc.Free()
	}
}

// Transpose-B device matmul (attention QKᵀ: Q[seq,hd]·K[seq,hd]ᵀ → [seq,seq])
// must match a materialized-transpose reference.
func TestCUDADeviceMatMulBTMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ m, k, n int }{{2, 3, 4}, {8, 64, 8}, {17, 65, 33}, {8, 128, 8}} {
		a := bench.RandF32(tensor.Shape{s.m, s.k}, 1) // [M,K]
		b := bench.RandF32(tensor.Shape{s.n, s.k}, 2) // [N,K]
		da, _ := cuda.UploadF32(a)
		db, _ := cuda.UploadF32(b)
		dc, err := da.MatMulBT(db)
		if err != nil {
			t.Fatalf("%v: %v", s, err)
		}
		got, _ := dc.ToHost() // [M,N]
		bt, _ := b.Transpose(0, 1)
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, bt}, nil)
		assertClose(t, got, gr[0], 1e-6*math.Sqrt(float64(s.k)), s)
		da.Free()
		db.Free()
		dc.Free()
	}
}

func assertClose(t *testing.T, got, want *tensor.Tensor, rtol float64, s any) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%v: shape %v vs %v", s, got.Shape(), want.Shape())
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if math.Abs(g-w) > rtol*math.Max(1, math.Abs(w)) {
			t.Fatalf("%v [%d]: %v vs %v (rtol %g)", s, i, g, w, rtol)
		}
	}
}
