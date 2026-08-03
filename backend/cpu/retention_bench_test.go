package cpu_test

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
	"testing"
)

func BenchmarkRetentionFwdF32_512x64(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	q := bench.RandF32(tensor.Shape{512, 64}, 1)
	k := bench.RandF32(tensor.Shape{512, 64}, 2)
	v := bench.RandF32(tensor.Shape{512, 64}, 3)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v}
	attrs := backend.RetentionAttrs{Gamma: 0.968}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpRetention, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRetentionBwdF32_512x64(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	q := bench.RandF32(tensor.Shape{512, 64}, 1)
	k := bench.RandF32(tensor.Shape{512, 64}, 2)
	v := bench.RandF32(tensor.Shape{512, 64}, 3)
	dO := bench.RandF32(tensor.Shape{512, 64}, 4)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v, dO}
	attrs := backend.RetentionAttrs{Gamma: 0.968}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpRetentionBackward, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFlashAttnFwdF32_512x8h(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	q := bench.RandF32(tensor.Shape{512, 512}, 1)
	k := bench.RandF32(tensor.Shape{512, 512}, 2)
	v := bench.RandF32(tensor.Shape{512, 512}, 3)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v}
	attrs := backend.AttnAttrs{Heads: 8, Causal: true, Block: 64}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpFlashAttn, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFlashAttnFwdF64_512x8h covers the F64 arm of the flash kernel, which had no cell.
// Its F32 twin above rides a separate branch — the f32 path was unroll-and-jammed over four keys
// by an earlier round and the f64 path was not — so a change to the f64 scores is invisible to
// every existing benchmark, and measuring one against the other proves nothing.
func BenchmarkFlashAttnFwdF64_512x8h(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	q := bench.RandF64(tensor.Shape{512, 512}, 1)
	k := bench.RandF64(tensor.Shape{512, 512}, 2)
	v := bench.RandF64(tensor.Shape{512, 512}, 3)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v}
	attrs := backend.AttnAttrs{Heads: 8, Causal: true, Block: 64}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpFlashAttn, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// The F64 twins. Both retention kernels branch on dtype — the f32 arm reduces with dot4T and the
// f64 arm keeps a serial dot — and every cell above is f32, so the f64 arms were unmeasured. That
// is the same gap the flash kernel had: a change to the f64 scores reads as noise against an f32
// benchmark because it never enters that branch.
func BenchmarkRetentionFwdF64_512x64(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	q := bench.RandF64(tensor.Shape{512, 64}, 1)
	k := bench.RandF64(tensor.Shape{512, 64}, 2)
	v := bench.RandF64(tensor.Shape{512, 64}, 3)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v}
	attrs := backend.RetentionAttrs{Gamma: 0.968}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpRetention, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRetentionBwdF64_512x64(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	q := bench.RandF64(tensor.Shape{512, 64}, 1)
	k := bench.RandF64(tensor.Shape{512, 64}, 2)
	v := bench.RandF64(tensor.Shape{512, 64}, 3)
	dO := bench.RandF64(tensor.Shape{512, 64}, 4)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v, dO}
	attrs := backend.RetentionAttrs{Gamma: 0.968}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpRetentionBackward, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
