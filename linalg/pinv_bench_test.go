package linalg_test

import (
	"testing"

	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

func benchPinv(b *testing.B, m, n int) {
	a := bench.RandF64(tensor.Shape{m, n}, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := linalg.Pinv(a); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPinv_64x32(b *testing.B) { benchPinv(b, 64, 32) }
func BenchmarkPinv_16x8(b *testing.B)  { benchPinv(b, 16, 8) }

// Control: NormFro touches the same tensors but not Pinv's accumulation loop.
func BenchmarkPinvControlNormFro(b *testing.B) {
	a := bench.RandF64(tensor.Shape{64, 32}, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := linalg.NormFro(a); err != nil {
			b.Fatal(err)
		}
	}
}
