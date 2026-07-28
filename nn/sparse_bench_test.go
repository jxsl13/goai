package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// These three modules carry PS6005 (register-blocking) findings and had NO benchmark at
// all, so no change to them could be validated on this host — the rule's findings were
// neither actionable nor declinable. Sizes are chosen so the flagged loops dominate:
// Sinkhorn by the cost matrix, KDA and NSA by sequence length.

func benchTensor(seed float64, dims ...int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape(dims))
	s := t.Storage().F64()
	for i := range s {
		s[i] = math.Sin(seed + 1.7*float64(i))
	}
	return t
}

func uniformVec(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1 / float64(n)
	}
	return v
}

// BenchmarkSinkhorn times the entropic optimal-transport iteration. Its two inner loops
// are the PS6005 sites: u = r ⊘ (K v) reads v[j] independently of the output row, and
// v = c ⊘ (Kᵀ u) reads u[i] independently of the output column.
func benchSinkhorn(b *testing.B, n, iters int) {
	cost := benchTensor(0.3, n, n)
	r, c := uniformVec(n), uniformVec(n)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := nn.Sinkhorn(cost, r, c, 0.5, iters); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSinkhorn_256x256(b *testing.B) { benchSinkhorn(b, 256, 50) }
func BenchmarkSinkhorn_512x512(b *testing.B) { benchSinkhorn(b, 512, 50) }

// BenchmarkKDA times Kimi Delta Attention over a realistic sequence.
func BenchmarkKDA_seq512(b *testing.B) {
	const seq, dk, dv = 512, 64, 64
	q, k, v := benchTensor(1, seq, dk), benchTensor(2, seq, dk), benchTensor(3, seq, dv)
	a := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	beta := tensor.New(tensor.F64, tensor.Shape{seq, 1})
	for i := range seq {
		av := 0.4 + 0.5*math.Abs(math.Sin(float64(i)*1.7))
		beta.SetF64(0.3+0.4*math.Abs(math.Cos(float64(i))), i, 0)
		for c := range dk {
			a.SetF64(av, i, c)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := nn.KimiDeltaAttention(q, k, v, a, beta); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNSABranches times all three Native Sparse Attention branches at once, which is
// how the layer uses them.
func BenchmarkNSABranches_seq512(b *testing.B) {
	const seq, dm, heads, block = 512, 128, 4, 32
	q, k, v := benchTensor(1, seq, dm), benchTensor(2, seq, dm), benchTensor(3, seq, dm)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := nn.NSABranches(q, k, v, heads, block, 8, 64, 0); err != nil {
			b.Fatal(err)
		}
	}
}
