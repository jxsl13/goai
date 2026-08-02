package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mlaBenchInputs builds an MLA backward at a shape where the O(seq²·dh) softmax
// recompute — the part PS4008 flags — actually dominates.
func mlaBenchInputs(seq, heads, dh, dR int) ([]*tensor.Tensor, *tensor.Tensor, backend.MLAAttrs) {
	hdh := heads * dh
	in := []*tensor.Tensor{
		moeF64(tensor.Shape{seq, hdh}, 1, false),
		moeF64(tensor.Shape{seq, hdh}, 2, false),
		moeF64(tensor.Shape{seq, hdh}, 3, false),
		moeF64(tensor.Shape{seq, heads * dR}, 4, false),
		moeF64(tensor.Shape{seq, dR}, 5, false),
	}
	g := moeF64(tensor.Shape{seq, hdh}, 6, false)
	return in, g, backend.MLAAttrs{Heads: heads, Causal: true, RoPEBase: 10000}
}

func benchMLAVJP(b *testing.B, seq, heads, dh, dR int) {
	vjp := vjps[backend.OpMLA]
	in, g, attrs := mlaBenchInputs(seq, heads, dh, dR)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, in, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLAVJPSeq128(b *testing.B) { benchMLAVJP(b, 128, 4, 32, 16) }
func BenchmarkMLAVJPSeq256(b *testing.B) { benchMLAVJP(b, 256, 4, 32, 16) }

// benchMLAVJPF32 covers the F32 arm, which had no cell. That gap is why the head split shipped
// for F64 only: the same transform applies here, but a symmetric edit with nothing to measure it
// against is an argument, not a result.
func benchMLAVJPF32(b *testing.B, seq, heads, dh, dR int) {
	vjp := vjps[backend.OpMLA]
	in64, g64, attrs := mlaBenchInputs(seq, heads, dh, dR)
	in := make([]*tensor.Tensor, len(in64))
	for i := range in64 {
		in[i] = moeCastF32(in64[i])
	}
	g := moeCastF32(g64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, in, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLAVJPSeq256F32(b *testing.B) { benchMLAVJPF32(b, 256, 4, 32, 16) }
