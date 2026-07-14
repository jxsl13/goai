package tensor

import "testing"

// Benchmarks that guard the §base-perf L0 hot paths against regression (§V5): the
// strided-view materialization (Contiguous) and the F64↔F32 GPU-prep cast (Cast),
// both of which the whole stack builds on. Historic wins vs the pre-devirt path:
// Contiguous ≈3×, Cast ≈17× (a per-element Unravel heap alloc + dtype dispatch → a
// direct typed loop with an incremental offset).

func BenchmarkContiguousStrided(b *testing.B) {
	x := New(F32, Shape{512, 512})
	xt, err := x.Transpose(0, 1) // non-contiguous view
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		_ = xt.Contiguous()
	}
}

func BenchmarkCastF64toF32(b *testing.B) {
	x := New(F64, Shape{512, 512}) // contiguous, the hot GPU-prep case
	b.ResetTimer()
	for range b.N {
		_ = x.Cast(F32)
	}
}

func BenchmarkCastStrided(b *testing.B) {
	x := New(F64, Shape{512, 512})
	xt, err := x.Transpose(0, 1)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		_ = xt.Cast(F32) // strided fallback path
	}
}
