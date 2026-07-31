package cpu

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// benchWKV measures OpWKV (RWKV-4 time-mixing recurrence) at a realistic
// [seq, d] shape. The F64 CPU kernel (wkvKernelCPU) now runs the channel-
// vectorized simd.WKVScanF64 scan (the d channels are independent → 4-wide over d
// with expF64x4v) — 5.6–6.7× over the scalar ref path this benchmark first
// measured (28.8ms→4.3ms at 512×1024, 116ms→20.8ms at 1024×2048). The SIMD exp is
// ~1 ulp vs libm; RWKV's decode step stays scalar and forward/decode still agree
// to ~1e-15 (TestRWKVDecodeMatchesForward, gate 1e-9). The non-AVX build falls back
// to the byte-for-byte scalar scan.
func benchWKV(b *testing.B, seq, d int) {
	k := bench.RandF64(tensor.Shape{seq, d}, 1)
	v := bench.RandF64(tensor.Shape{seq, d}, 2)
	w := bench.RandF64(tensor.Shape{d}, 3)
	u := bench.RandF64(tensor.Shape{d}, 4)
	in := []*tensor.Tensor{k, v, w, u}
	ctx := backend.NewContext()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpWKV, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWKV_512x1024(b *testing.B)  { benchWKV(b, 512, 1024) }
func BenchmarkWKV_1024x2048(b *testing.B) { benchWKV(b, 1024, 2048) }

func benchWKVF32(b *testing.B, seq, d int) {
	k := bench.RandF32(tensor.Shape{seq, d}, 1)
	v := bench.RandF32(tensor.Shape{seq, d}, 2)
	w := bench.RandF32(tensor.Shape{d}, 3)
	u := bench.RandF32(tensor.Shape{d}, 4)
	in := []*tensor.Tensor{k, v, w, u}
	ctx := backend.NewContext()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpWKV, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWKVF32_512x1024(b *testing.B)  { benchWKVF32(b, 512, 1024) }
func BenchmarkWKVF32_1024x2048(b *testing.B) { benchWKVF32(b, 1024, 2048) }

// benchWKVF32On measures the F32 WKV on a chosen backend — PRE (Ref, serial) vs POST
// (CPU, channel-parallel) A/B for the dtype-gap win.
func benchWKVF32On(b *testing.B, name backend.Name, seq, d int) {
	be, _ := backend.Get(name)
	k := bench.RandF32(tensor.Shape{seq, d}, 1)
	v := bench.RandF32(tensor.Shape{seq, d}, 2)
	w := bench.RandF32(tensor.Shape{d}, 3)
	u := bench.RandF32(tensor.Shape{d}, 4)
	in := []*tensor.Tensor{k, v, w, u}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpWKV, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWKVF32Ref_512x1024(b *testing.B)  { benchWKVF32On(b, backend.Ref, 512, 1024) }
func BenchmarkWKVF32CPU_512x1024(b *testing.B)  { benchWKVF32On(b, backend.CPU, 512, 1024) }
func BenchmarkWKVF32Ref_1024x2048(b *testing.B) { benchWKVF32On(b, backend.Ref, 1024, 2048) }
func BenchmarkWKVF32CPU_1024x2048(b *testing.B) { benchWKVF32On(b, backend.CPU, 1024, 2048) }
