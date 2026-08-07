package autograd

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkDoRAWeightBackward times the DoRA weight-decomposition VJP at a cache-exceeding weight size.
// Its two per-column loops walk DOWN columns (stride cols) — the PS1006 strided-reduction anti-pattern;
// this is the gate for a row-major (contiguous) restructure that stays bit-identical (each column's
// reduction keeps i-ascending order).
func BenchmarkDoRAWeightBackward(b *testing.B) {
	vjp := vjps[backend.OpDoRAWeight]
	for _, d := range []struct{ rows, cols int }{
		{2048, 2048},
		{4096, 1024},
	} {
		b.Run(doraName(d.rows, d.cols), func(b *testing.B) {
			rng := rand.New(rand.NewSource(5))
			v := tensor.New(tensor.F64, tensor.Shape{d.rows, d.cols})
			g := tensor.New(tensor.F64, tensor.Shape{d.rows, d.cols})
			m := tensor.New(tensor.F64, tensor.Shape{d.cols})
			for i, s := 0, v.Storage().F64(); i < len(s); i++ {
				s[i] = rng.NormFloat64()
			}
			for i, s := 0, g.Storage().F64(); i < len(s); i++ {
				s[i] = rng.NormFloat64()
			}
			for i, s := 0, m.Storage().F64(); i < len(s); i++ {
				s[i] = rng.NormFloat64()
			}
			in := []*tensor.Tensor{v, m}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := vjp(nil, in, nil, nil, g); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkDoRAWeightBackwardF32 covers the f32 fast path (same strided→row-major restructure).
func BenchmarkDoRAWeightBackwardF32(b *testing.B) {
	vjp := vjps[backend.OpDoRAWeight]
	rng := rand.New(rand.NewSource(5))
	const rows, cols = 2048, 2048
	v := tensor.New(tensor.F32, tensor.Shape{rows, cols})
	g := tensor.New(tensor.F32, tensor.Shape{rows, cols})
	m := tensor.New(tensor.F32, tensor.Shape{cols})
	for i, s := 0, v.Storage().F32(); i < len(s); i++ {
		s[i] = float32(rng.NormFloat64())
	}
	for i, s := 0, g.Storage().F32(); i < len(s); i++ {
		s[i] = float32(rng.NormFloat64())
	}
	for i, s := 0, m.Storage().F32(); i < len(s); i++ {
		s[i] = float32(rng.NormFloat64())
	}
	in := []*tensor.Tensor{v, m}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, in, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func doraName(r, c int) string {
	return "r" + itoaD(r) + "_c" + itoaD(c)
}

func itoaD(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
