package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// peerRetrieve is PEER's product-key top-k retrieval, run per token: it argsorts
// the two sub-key score vectors (length n, n² = expert count) and the k'² Cartesian
// candidates. Benched at a million-expert config (n=1024 → 1,048,576 experts).
func benchPeerRetrieve(b *testing.B, n, topK, subKeyK int) {
	s1 := make([]float64, n)
	s2 := make([]float64, n)
	for i := range s1 {
		s1[i] = math.Sin(float64(i) * 0.031)
		s2[i] = math.Cos(float64(i) * 0.047)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _, _ = peerRetrieve(s1, s2, n, topK, subKeyK)
	}
}

func BenchmarkPeerRetrieve_1M(b *testing.B)   { benchPeerRetrieve(b, 1024, 16, 32) } // ~1M experts
func BenchmarkPeerRetrieve_256K(b *testing.B) { benchPeerRetrieve(b, 512, 16, 32) }
func BenchmarkPeerTopIndices(b *testing.B) {
	s := make([]float64, 1024)
	for i := range s {
		s[i] = math.Sin(float64(i) * 0.031)
	}
	b.ResetTimer()
	for range b.N {
		_ = peerTopIndices(s, 32)
	}
}

// BenchmarkPeerForward exercises PEER.Forward, and therefore the backend DISPATCH path, which the
// two retrieval benchmarks above do not: they call peerRetrieve and peerTopIndices directly. That
// gap is why a per-dispatch allocation defect in this package was invisible — nn ships 175
// benchmarks and they overwhelmingly target internal helpers rather than a Forward
// (BENCH-PROVE-THE-CODE-RAN-001).
func BenchmarkPeerForward(b *testing.B) {
	const dModel, n, topK, tokens = 64, 16, 8, 8
	p := NewPEER(tensor.F64, dModel, n, topK, 7)
	x := tensor.New(tensor.F64, tensor.Shape{tokens, dModel})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Sin(float64(i) * 0.017)
	}
	ctx := backend.NewContext()
	if _, _, _, err := p.Forward(ctx, x); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, _, err := p.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
