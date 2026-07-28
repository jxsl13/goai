package tensor

import "testing"

// benchStridedHalfCast builds the shape that actually reaches gatherGeneric: a
// NON-CONTIGUOUS f16/bf16 view cast to a wider float type. Every same-dtype view goes
// to gatherCast/gatherRows/gatherBlocked2D, and F32<->F64 is handled explicitly, so the
// generic widen-through-f64 walk is the residual for half casts only.
//
// The shape mimics a sliced attention/KV tensor: [B,H,T,D] in f16, sliced on H so the
// view is strided, then materialized as f32.
func benchStridedHalfCast(b *testing.B, src, dst Dtype) {
	const B, H, T, D = 4, 8, 64, 64
	t := New(src, Shape{B, H, T, D})
	s := t.Storage().U16()
	for i := range s {
		s[i] = uint16(i % 4096)
	}
	view, err := t.Slice(1, 1, H-1) // strided on a non-innermost axis
	if err != nil {
		b.Fatalf("Slice: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = view.Cast(dst)
	}
}

func BenchmarkStridedCastF16toF32(b *testing.B)  { benchStridedHalfCast(b, F16, F32) }
func BenchmarkStridedCastBF16toF64(b *testing.B) { benchStridedHalfCast(b, BF16, F64) }
