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

// A resident-B matmul must equal the per-call cuda matmul EXACTLY (same cuBLAS
// Sgemm, same operands — only B's provenance differs) and the f64 reference
// within the §V11 GPU-f32 tolerance. Reuse across several activations A must
// stay correct (the whole point is one upload, many matmuls).
func TestCUDAResidentBMatchesRefAndCuda(t *testing.T) {
	skipNoGPU(t)
	cb, ok := backend.Get(backend.CUDA)
	if !ok {
		t.Fatal("cuda available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)

	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {33, 65, 17}, {128, 128, 128}, {256, 512, 128},
	}
	for _, s := range shapes {
		b := bench.RandF32(tensor.Shape{s.k, s.n}, 2)
		rb, err := cuda.NewResidentB(b)
		if err != nil {
			t.Fatalf("shape %v: NewResidentB: %v", s, err)
		}
		// Two different activations against the same resident weight.
		for seed := 1; seed <= 2; seed++ {
			a := bench.RandF32(tensor.Shape{s.m, s.k}, uint64(seed))
			got, err := rb.MatMul(a)
			if err != nil {
				t.Fatalf("shape %v seed %d: %v", s, seed, err)
			}
			// exact vs the per-call cuda matmul
			gc, _ := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
			for i := range got.Numel() {
				idx := tensor.Unravel(i, got.Shape())
				if g, c := got.AtF64(idx...), gc[0].AtF64(idx...); g != c {
					t.Fatalf("shape %v [%d]: resident %v != cuda %v", s, i, g, c)
				}
			}
			// tolerance vs ref
			gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
			rtol := crossTol(s.k)
			for i := range got.Numel() {
				idx := tensor.Unravel(i, got.Shape())
				g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
					t.Fatalf("shape %v [%d]: resident %v vs ref %v (rtol %g)", s, i, g, r, rtol)
				}
			}
		}
		rb.Free()
		rb.Free() // idempotent
	}
}

// After Free, MatMul must error rather than dereference a freed device buffer.
func TestCUDAResidentBUseAfterFree(t *testing.T) {
	skipNoGPU(t)
	b := bench.RandF32(tensor.Shape{4, 4}, 2)
	rb, err := cuda.NewResidentB(b)
	if err != nil {
		t.Fatal(err)
	}
	rb.Free()
	if _, err := rb.MatMul(bench.RandF32(tensor.Shape{4, 4}, 1)); err == nil {
		t.Fatal("expected error on MatMul after Free")
	}
}

// The transfer win: a weight reused across many activations. Resident uploads B
// once; the per-call path re-uploads B every matmul. Same cuBLAS kernel, so any
// delta is the saved B H2D.
func benchResident(b *testing.B, resident bool, sz int) {
	be, ok := backend.Get(backend.CUDA)
	if !ok {
		b.Skip("cuda not available")
	}
	w := bench.RandF32(tensor.Shape{sz, sz}, 2)
	a := bench.RandF32(tensor.Shape{sz, sz}, 1)
	if resident {
		rb, err := cuda.NewResidentB(w)
		if err != nil {
			b.Fatal(err)
		}
		defer rb.Free()
		b.ResetTimer()
		for range b.N {
			if _, err := rb.MatMul(a); err != nil {
				b.Fatal(err)
			}
		}
		return
	}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, w}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkResidentB_512(b *testing.B)  { benchResident(b, true, 512) }
func BenchmarkPerCallB_512(b *testing.B)   { benchResident(b, false, 512) }
func BenchmarkResidentB_1024(b *testing.B) { benchResident(b, true, 1024) }
func BenchmarkPerCallB_1024(b *testing.B)  { benchResident(b, false, 1024) }
