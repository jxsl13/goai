package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMHAMaskedBackward covers the masked multi-head attention VJP (dQ/dK/dV/dMask).
// seq=128, dm=256, heads=4, 2-D mask.
func BenchmarkMHAMaskedBackward(b *testing.B) {
	const seq, dm, heads = 128, 256, 4
	mk := func(r, c int, fn func(i int) float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{r, c})
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	q := mk(seq, dm, func(i int) float64 { return math.Sin(float64(i) * 0.01) })
	k := mk(seq, dm, func(i int) float64 { return math.Cos(float64(i) * 0.013) })
	v := mk(seq, dm, func(i int) float64 { return math.Sin(float64(i) * 0.017) })
	g := mk(seq, dm, func(i int) float64 { return math.Cos(float64(i) * 0.019) })
	// causal mask [seq,seq]: 0 for j<=i, -Inf above
	mask := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	ms := mask.Storage().F64()
	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			if j > i {
				ms[i*seq+j] = math.Inf(-1)
			}
		}
	}
	ctx := backend.NewContext()
	attrs := backend.AttnAttrs{Heads: heads}
	in := []*tensor.Tensor{q, k, v, mask, g}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHAMaskedBackward, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
