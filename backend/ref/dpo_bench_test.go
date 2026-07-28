package ref_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func benchDPO(b *testing.B, batch int) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	in := []*tensor.Tensor{
		bench.RandF64(tensor.Shape{batch}, 1), bench.RandF64(tensor.Shape{batch}, 2),
		bench.RandF64(tensor.Shape{batch}, 3), bench.RandF64(tensor.Shape{batch}, 4),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpDPO, in, backend.DPOAttrs{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDPOLoss_64(b *testing.B)   { benchDPO(b, 64) }
func BenchmarkDPOLoss_4096(b *testing.B) { benchDPO(b, 4096) }

func benchPPO(b *testing.B, batch int) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	in := []*tensor.Tensor{
		bench.RandF64(tensor.Shape{batch}, 1), bench.RandF64(tensor.Shape{batch}, 2),
		bench.RandF64(tensor.Shape{batch}, 3),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpPPOClip, in, backend.PPOClipAttrs{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPPOClip_4096(b *testing.B) { benchPPO(b, 4096) }

func benchPref(b *testing.B, op backend.Op, attrs backend.Attrs, nIn, batch int) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	in := make([]*tensor.Tensor, nIn)
	for i := range in {
		in[i] = bench.RandF64(tensor.Shape{batch}, uint64(i+1))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGRPO_4096(b *testing.B) { benchPref(b, backend.OpGRPO, backend.GRPOAttrs{}, 4, 4096) }
func BenchmarkCPO_4096(b *testing.B)  { benchPref(b, backend.OpCPO, backend.CPOAttrs{}, 2, 4096) }
func BenchmarkIPO_4096(b *testing.B)  { benchPref(b, backend.OpIPO, backend.IPOAttrs{}, 4, 4096) }
