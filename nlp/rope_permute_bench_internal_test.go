package nlp

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// The RoPE weight permutations run once per tensor at MODEL LOAD, over full weight matrices, and had
// no benchmark — so PS1001's report on them was unrankable. Sizes are chosen to match real attention
// weights rather than the tiny fixtures the correctness tests use: a DeepSeek-V2-shaped kv_b is
// thousands of rows by hundreds of columns, and load time is a user-facing cost even though no
// end-to-end benchmark exercises it.
//
// deinterleaveRoPE is the heavier of the two: it copies the WHOLE matrix first and then overwrites
// the pe rows, so it pays out*in element moves before it permutes anything.

func ropeWeight(out, in int) *tensor.Tensor {
	d := make([]float64, out*in)
	for i := range d {
		d[i] = float64(i%251) * 0.125
	}
	return tensor.FromFloat64(tensor.Shape{out, in}, d)
}

func BenchmarkDeinterleaveRoPE(b *testing.B) {
	for _, g := range []struct{ out, in, heads, block, peOff, ropeDim int }{
		{2048, 512, 16, 128, 64, 64},
		{4096, 1024, 32, 128, 64, 64},
	} {
		w := ropeWeight(g.out, g.in)
		b.Run(ropeName(g.out, g.in), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = deinterleaveRoPE(w, g.heads, g.block, g.peOff, g.ropeDim)
			}
		})
	}
}

func BenchmarkPermuteInterleaveToSplit(b *testing.B) {
	for _, g := range []struct{ out, in, heads, headDim int }{
		{2048, 512, 16, 128},
		{4096, 1024, 32, 128},
	} {
		w := ropeWeight(g.out, g.in)
		b.Run(ropeName(g.out, g.in), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = permuteInterleaveToSplit(w, g.heads, g.headDim)
			}
		})
	}
}

func BenchmarkPermuteSplitToInterleave(b *testing.B) {
	for _, g := range []struct{ out, in, heads, headDim int }{
		{2048, 512, 16, 128},
		{4096, 1024, 32, 128},
	} {
		w := ropeWeight(g.out, g.in)
		b.Run(ropeName(g.out, g.in), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = permuteSplitToInterleave(w, g.heads, g.headDim)
			}
		})
	}
}

func BenchmarkSplitNeoXQKV(b *testing.B) {
	for _, g := range []struct{ heads, headDim, in int }{
		{16, 128, 512},
		{32, 128, 1024},
	} {
		w := ropeWeight(g.heads*3*g.headDim, g.in)
		b.Run(ropeName(g.heads*g.headDim, g.in), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _, _ = splitNeoXQKV(w, g.heads, g.headDim)
			}
		})
	}
}

func ropeName(out, in int) string { return itoaRope(out) + "x" + itoaRope(in) }

func itoaRope(n int) string {
	if n == 0 {
		return "0"
	}
	var d [12]byte
	i := len(d)
	for n > 0 {
		i--
		d[i] = byte('0' + n%10)
		n /= 10
	}
	return string(d[i:])
}
