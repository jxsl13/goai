package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// DSAAttention had no benchmark, so neither the per-query sort's allocation cost nor any
// change to the selection could be validated.
func dsaInputs(seq, dm, idxWidth int) (q, k, v, qi, ki *tensor.Tensor) {
	mk := func(rows, cols int, seed float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(seed + 0.013*float64(i))
		}
		return x
	}
	return mk(seq, dm, 1), mk(seq, dm, 2), mk(seq, dm, 3), mk(seq, idxWidth, 4), mk(seq, idxWidth, 5)
}

func benchDSA(b *testing.B, seq, dm, heads, topK int) {
	q, k, v, qi, ki := dsaInputs(seq, dm, 32)
	w := []float64{1, 0.5}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.DSAAttention(q, k, v, qi, ki, w, heads, topK, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDSAAttention_seq512(b *testing.B)  { benchDSA(b, 512, 128, 4, 64) }
func BenchmarkDSAAttention_seq1024(b *testing.B) { benchDSA(b, 1024, 128, 4, 64) }

// TestDSADeterministic pins the selection against tied indexer scores. DSA ranks keys by
// ReLU'd dot products, which tie at exactly zero routinely; if the comparator orders on score
// alone, an unstable sort decides WHICH keys are attended and repeated runs can disagree.
func TestDSADeterministic(t *testing.T) {
	const seq, dm, heads, topK = 24, 32, 2, 5
	// Every indexer row identical => every score ties.
	flat := tensor.New(tensor.F64, tensor.Shape{seq, 32})
	for i := range flat.Storage().F64() {
		flat.Storage().F64()[i] = 0.3
	}
	q, k, v, _, _ := dsaInputs(seq, dm, 32)
	var first []float64
	for run := range 5 {
		out, err := nn.DSAAttention(q, k, v, flat, flat, []float64{1, 0.5}, heads, topK, 0)
		if err != nil {
			t.Fatal(err)
		}
		cur := make([]float64, out.Numel())
		for i := range cur {
			cur[i] = out.AtF64(tensor.Unravel(i, out.Shape())...)
		}
		if run == 0 {
			first = cur
			continue
		}
		for i := range cur {
			if math.Float64bits(cur[i]) != math.Float64bits(first[i]) {
				t.Fatalf("run %d differs from run 0 at %d: %g vs %g — tied indexer scores make the"+
					" attended set nondeterministic", run, i, cur[i], first[i])
			}
		}
	}
}
