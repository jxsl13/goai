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
