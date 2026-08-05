package nn

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchMoDCombine(b *testing.B, seq, d int) {
	ctx := backend.NewContext()
	m := NewMixtureOfDepths(tensor.F32, d, 0.25)
	x := tensor.New(tensor.F32, tensor.Shape{seq, d})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32(i%97)*0.01 - 0.5
	}
	for i := range m.Router.Storage().F32() {
		m.Router.Storage().F32()[i] = float32(i%13) * 0.02
	}
	_, weights, sel, err := m.Route(ctx, x)
	if err != nil {
		b.Fatal(err)
	}
	processed := tensor.New(tensor.F32, tensor.Shape{len(sel.Idx), d})
	for i := range processed.Storage().F32() {
		processed.Storage().F32()[i] = float32(i%53) * 0.03
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := m.Combine(ctx, x, processed, weights, sel); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoDCombine_2048x512(b *testing.B) { benchMoDCombine(b, 2048, 512) }
func BenchmarkMoDCombine_4096x768(b *testing.B) { benchMoDCombine(b, 4096, 768) }
