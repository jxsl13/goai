package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkGSPOVJP covers the GSPO sequence-ratio gradient over a realistic rollout:
// G=256 sequences × length 512 (131072 tokens), F64.
func BenchmarkGSPOVJP(b *testing.B) {
	const G, L = 256, 512
	total := G * L
	lengths := make([]int, G)
	for i := range lengths {
		lengths[i] = L
	}
	mk := func(seed int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{total})
		s := t.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64((i+seed)*7)*0.0001) * 0.1
		}
		return t
	}
	lpNew, lpOld := mk(0), mk(9)
	adv := tensor.New(tensor.F64, tensor.Shape{G})
	as := adv.Storage().F64()
	for i := range as {
		as[i] = math.Cos(float64(i) * 0.05)
	}
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.SetF64(1.0)
	ctx := backend.NewContext()
	fn := vjps[backend.OpGSPO]
	in := []*tensor.Tensor{lpNew, lpOld, adv}
	attrs := backend.GSPOAttrs{Epsilon: 3e-4, Lengths: lengths}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := fn(ctx, in, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}
