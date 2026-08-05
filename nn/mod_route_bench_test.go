package nn

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchMoDRoute(b *testing.B, seq, d int) {
	ctx := backend.NewContext()
	m := NewMixtureOfDepths(tensor.F32, d, 0.25)
	x := tensor.New(tensor.F32, tensor.Shape{seq, d})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32(i%97)*0.01 - 0.5
	}
	rs := m.Router.Storage().F32()
	for i := range rs {
		rs[i] = float32(i%13) * 0.02
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, _, err := m.Route(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoDRoute_2048x512(b *testing.B) { benchMoDRoute(b, 2048, 512) }
func BenchmarkMoDRoute_4096x768(b *testing.B) { benchMoDRoute(b, 4096, 768) }
