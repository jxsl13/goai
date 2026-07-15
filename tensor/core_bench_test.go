package tensor

import "testing"

// Benchmarks for the remaining L0 hot paths (§base-perf): element access,
// unravel/offset math, stride computation, tensor creation, allocator get/put,
// half conversions, and the fill-based constructors. Together with
// perf_bench_test.go these cover every op the upper layers lean on.

func BenchmarkAtF64Contig2D(b *testing.B) {
	x := New(F32, Shape{64, 64})
	b.ResetTimer()
	var s float64
	for range b.N {
		for i := 0; i < 64; i++ {
			for j := 0; j < 64; j++ {
				s += x.AtF64(i, j)
			}
		}
	}
	_ = s
}

func BenchmarkSetF64Contig2D(b *testing.B) {
	x := New(F32, Shape{64, 64})
	b.ResetTimer()
	for range b.N {
		for i := 0; i < 64; i++ {
			for j := 0; j < 64; j++ {
				x.SetF64(1.0, i, j)
			}
		}
	}
}

func BenchmarkAtF64Strided2D(b *testing.B) {
	x := New(F32, Shape{64, 64})
	xt, err := x.Transpose(0, 1)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	var s float64
	for range b.N {
		for i := 0; i < 64; i++ {
			for j := 0; j < 64; j++ {
				s += xt.AtF64(i, j)
			}
		}
	}
	_ = s
}

func BenchmarkUnravel4D(b *testing.B) {
	sh := Shape{8, 16, 32, 4}
	b.ResetTimer()
	for i := range b.N {
		_ = Unravel(i&4095, sh)
	}
}

func BenchmarkOffset4D(b *testing.B) {
	st := RowMajorStrides(Shape{8, 16, 32, 4})
	idx := []int{3, 7, 15, 2}
	b.ResetTimer()
	var s int
	for range b.N {
		s += Offset(0, st, idx)
	}
	_ = s
}

func BenchmarkRowMajorStrides4D(b *testing.B) {
	sh := Shape{8, 16, 32, 4}
	b.ResetTimer()
	for range b.N {
		_ = RowMajorStrides(sh)
	}
}

func BenchmarkIsContiguous4D(b *testing.B) {
	sh := Shape{8, 16, 32, 4}
	st := RowMajorStrides(sh)
	b.ResetTimer()
	var ok bool
	for range b.N {
		ok = st.IsContiguous(sh)
	}
	_ = ok
}

func BenchmarkNewSmall(b *testing.B) {
	sh := Shape{8, 8}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = New(F32, sh)
	}
}

func BenchmarkNewScalar(b *testing.B) {
	sh := Shape{}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = New(F32, sh)
	}
}

func BenchmarkPoolAllocFree(b *testing.B) {
	p := NewPool()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		buf := p.Alloc(F32, 4096)
		p.Free(buf)
	}
}

func BenchmarkPoolAllocFreeSmall(b *testing.B) {
	p := NewPool()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		buf := p.Alloc(F64, 64)
		p.Free(buf)
	}
}

func BenchmarkCastF32toF64(b *testing.B) {
	x := New(F32, Shape{512, 512})
	b.ResetTimer()
	for range b.N {
		_ = x.Cast(F64)
	}
}

func BenchmarkCastF16toF32(b *testing.B) {
	x := New(F16, Shape{512, 512})
	b.ResetTimer()
	for range b.N {
		_ = x.Cast(F32)
	}
}

func BenchmarkCastF32toF16(b *testing.B) {
	x := New(F32, Shape{512, 512})
	b.ResetTimer()
	for range b.N {
		_ = x.Cast(F16)
	}
}

func BenchmarkCastBF16toF32(b *testing.B) {
	x := New(BF16, Shape{512, 512})
	b.ResetTimer()
	for range b.N {
		_ = x.Cast(F32)
	}
}

// Contiguous on a view that is already densely laid out but carries an offset
// (a dim-0 slice) — should be a flat copy, not a strided gather.
func BenchmarkContiguousOffsetView(b *testing.B) {
	x := New(F32, Shape{512, 512})
	xs, err := x.Slice(0, 128, 384)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		_ = xs.Contiguous()
	}
}

// Contiguous on a last-dim slice: innermost stride is 1, rows are contiguous
// runs — should be row copies, not per-element gathers.
func BenchmarkContiguousInnerRows(b *testing.B) {
	x := New(F32, Shape{512, 512})
	xs, err := x.Slice(1, 0, 256)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		_ = xs.Contiguous()
	}
}

func BenchmarkF16ToF32(b *testing.B) {
	var s float32
	for i := range b.N {
		s += f16ToF32(uint16(i))
	}
	_ = s
}

func BenchmarkF32ToF16(b *testing.B) {
	var s uint16
	for i := range b.N {
		s ^= f32ToF16(float32(i) * 0.25)
	}
	_ = s
}

func BenchmarkOnesF16(b *testing.B) {
	sh := Shape{128, 128}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = Ones(F16, sh)
	}
}

func BenchmarkEyeF32(b *testing.B) {
	b.ResetTimer()
	for range b.N {
		_ = Eye(F32, 256)
	}
}

func BenchmarkSliceView(b *testing.B) {
	x := New(F32, Shape{64, 64})
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = x.Slice(0, 8, 56)
	}
}

func BenchmarkTransposeView(b *testing.B) {
	x := New(F32, Shape{64, 64})
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = x.Transpose(0, 1)
	}
}

func BenchmarkPermuteView(b *testing.B) {
	x := New(F32, Shape{8, 16, 32, 4})
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = x.Permute(2, 0, 3, 1)
	}
}

func BenchmarkRandF32(b *testing.B) {
	sh := Shape{128, 128}
	b.ResetTimer()
	for range b.N {
		_ = Rand(F32, 42, sh)
	}
}
