package nlp

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// f32CloneSlow is the verbatim pre-fast-path f32Clone, the bit-identity oracle:
// the contiguous typed-slice path must reproduce it exactly for every source
// dtype, and the non-contiguous fallback IS this loop.
func f32CloneSlow(t *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(tensor.F32, t.Shape())
	dst := out.Storage().F32()
	for i := range t.Numel() {
		dst[i] = float32(t.AtF64(tensor.Unravel(i, t.Shape())...))
	}
	return out
}

func fillSeqNLP(t *tensor.Tensor, start, step float64) {
	sh := t.Shape()
	for i := range t.Numel() {
		t.SetF64(start+float64(i)*step, tensor.Unravel(i, sh)...)
	}
}

// f32Clone must be bit-identical to the per-element oracle for every source dtype
// (F32/F64 take the fast path, F16/BF16 fall through), across shapes.
func TestF32CloneBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64, tensor.F16, tensor.BF16} {
		for _, sh := range []tensor.Shape{{1}, {2048}, {4, 5}, {7, 3, 2}, {256, 128}} {
			src := tensor.New(dt, sh)
			// 0.5-spaced values: exact in F32/F64, and their round-trip through the
			// half floats is deterministic (both paths narrow identically).
			fillSeqNLP(src, -3.5, 0.5)
			fast, slow := f32Clone(src), f32CloneSlow(src)
			fs, ss := fast.Storage().F32(), slow.Storage().F32()
			for i := range fast.Numel() {
				if math.Float32bits(fs[i]) != math.Float32bits(ss[i]) {
					t.Fatalf("dt=%v sh=%v elem %d: fast %v != slow %v (not bit-identical)", dt, sh, i, fs[i], ss[i])
				}
			}
		}
	}
}

// A non-contiguous source must route to the fallback and stay bit-identical.
func TestF32CloneNonContiguous(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		base := tensor.New(dt, tensor.Shape{6, 4})
		fillSeqNLP(base, 1.0, 0.25)
		view, err := base.Transpose(0, 1) // [4,6], non-contiguous
		if err != nil {
			t.Fatal(err)
		}
		if view.IsContiguous() {
			t.Fatal("expected a non-contiguous source")
		}
		fast, slow := f32Clone(view), f32CloneSlow(view)
		fs, ss := fast.Storage().F32(), slow.Storage().F32()
		for i := range fast.Numel() {
			if math.Float32bits(fs[i]) != math.Float32bits(ss[i]) {
				t.Fatalf("dt=%v elem %d: fast %v != slow %v (non-contiguous fallback not bit-identical)", dt, i, fs[i], ss[i])
			}
		}
		if fast.Shape().String() != view.Shape().String() {
			t.Fatalf("dt=%v: shape %v, want %v", dt, fast.Shape(), view.Shape())
		}
	}
}

func benchF32Clone(b *testing.B, dt tensor.Dtype, fn func(*tensor.Tensor) *tensor.Tensor) {
	src := tensor.New(dt, tensor.Shape{8000, 2048}) // token-embedding scale
	for i := range src.Numel() {
		src.SetF64(float64(i%97)*0.5, tensor.Unravel(i, src.Shape())...)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fn(src)
	}
}

func BenchmarkF32CloneFastF32(b *testing.B) { benchF32Clone(b, tensor.F32, f32Clone) }
func BenchmarkF32CloneSlowF32(b *testing.B) { benchF32Clone(b, tensor.F32, f32CloneSlow) }
func BenchmarkF32CloneFastF64(b *testing.B) { benchF32Clone(b, tensor.F64, f32Clone) }
func BenchmarkF32CloneSlowF64(b *testing.B) { benchF32Clone(b, tensor.F64, f32CloneSlow) }
