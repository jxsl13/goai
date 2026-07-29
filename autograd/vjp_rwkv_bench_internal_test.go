package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The WKV VJP had no benchmark. It is O(seq^2 * d) and every access to k, v, g and the
// gradient buffers strides by d, so its column-walk findings are the most expensive in
// autograd — and were unvalidatable until now.
func wkvInputs(seq, d int, dt tensor.Dtype) (k, v, w, u, g *tensor.Tensor) {
	mk := func(shape tensor.Shape, fn func(i int) float64) *tensor.Tensor {
		t := tensor.New(dt, shape)
		for i := range t.Numel() {
			t.SetF64(fn(i), tensor.Unravel(i, t.Shape())...)
		}
		return t
	}
	k = mk(tensor.Shape{seq, d}, func(i int) float64 { return 0.3 * math.Sin(float64(i)*0.013) })
	v = mk(tensor.Shape{seq, d}, func(i int) float64 { return math.Cos(float64(i) * 0.017) })
	w = mk(tensor.Shape{d}, func(i int) float64 { return 0.5 + 0.2*math.Abs(math.Sin(float64(i))) })
	u = mk(tensor.Shape{d}, func(i int) float64 { return 0.1 * math.Cos(float64(i)*0.7) })
	g = mk(tensor.Shape{seq, d}, func(i int) float64 { return math.Sin(float64(i) * 0.023) })
	return
}

func benchWKV(b *testing.B, dt tensor.Dtype) {
	vjp := vjps[backend.OpWKV]
	k, v, w, u, g := wkvInputs(256, 64, dt)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(nil, []*tensor.Tensor{k, v, w, u}, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWKVVJP_F64(b *testing.B) { benchWKV(b, tensor.F64) }
func BenchmarkWKVVJP_F32(b *testing.B) { benchWKV(b, tensor.F32) }

func wkvSum(seq, d int, dt tensor.Dtype) uint64 {
	vjp := vjps[backend.OpWKV]
	k, v, w, u, g := wkvInputs(seq, d, dt)
	out, err := vjp(nil, []*tensor.Tensor{k, v, w, u}, nil, nil, g)
	if err != nil {
		return 0
	}
	h := uint64(14695981039346656037)
	for _, t := range out {
		for i := range t.Numel() {
			bits := math.Float64bits(t.AtF64(tensor.Unravel(i, t.Shape())...))
			for s := 0; s < 64; s += 8 {
				h = (h ^ ((bits >> s) & 0xff)) * 1099511628211
			}
		}
	}
	return h
}

// TestWKVVJPBitIdentical pins all four gradients bit-for-bit on both typed branches.
func TestWKVVJPBitIdentical(t *testing.T) {
	for _, c := range wkvGolden {
		dt := tensor.F64
		if c.f32 {
			dt = tensor.F32
		}
		if got := wkvSum(c.seq, c.d, dt); got != c.sum {
			t.Fatalf("seq=%d d=%d f32=%v: checksum %d, want %d", c.seq, c.d, c.f32, got, c.sum)
		}
	}
}

type wkvCase struct {
	seq, d int
	f32    bool
	sum    uint64
}
