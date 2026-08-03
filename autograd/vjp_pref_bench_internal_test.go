package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// prefInputs builds n-element F64 operands in a range every rule's branches actually reach.
func prefInputs(n, k int) []*tensor.Tensor {
	out := make([]*tensor.Tensor, k)
	for j := range k {
		t := tensor.New(tensor.F64, tensor.Shape{n})
		s := t.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64(i)*0.31+float64(j)*1.3) * 0.4
		}
		out[j] = t
	}
	return out
}

func benchPrefVJP(b *testing.B, op backend.Op, k int, attrs backend.Attrs, n int) {
	vjp := vjps[op]
	if vjp == nil {
		b.Skipf("%v VJP not registered", op)
	}
	in := prefInputs(n, k)
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 1.5
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(nil, in, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCPOVJP_65536(b *testing.B) {
	benchPrefVJP(b, backend.OpCPO, 2, backend.CPOAttrs{Beta: 0.1, Alpha: 1}, 65536)
}

func BenchmarkDPOVJP_65536(b *testing.B) {
	benchPrefVJP(b, backend.OpDPO, 4, backend.DPOAttrs{Beta: 0.1}, 65536)
}

func BenchmarkIPOVJP_65536(b *testing.B) {
	benchPrefVJP(b, backend.OpIPO, 4, backend.IPOAttrs{Beta: 0.1}, 65536)
}

func BenchmarkKTOVJP_65536(b *testing.B) {
	benchPrefVJP(b, backend.OpKTO, 3, backend.KTOAttrs{Beta: 0.1}, 65536)
}

func BenchmarkSimPOVJP_65536(b *testing.B) {
	benchPrefVJP(b, backend.OpSimPO, 2, backend.SimPOAttrs{Beta: 2, Gamma: 0.5}, 65536)
}

func BenchmarkGRPOVJP_65536(b *testing.B) {
	benchPrefVJP(b, backend.OpGRPO, 4, backend.GRPOAttrs{Epsilon: 0.2, Beta: 0.04}, 65536)
}
