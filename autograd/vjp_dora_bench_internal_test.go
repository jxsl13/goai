package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The DoRA VJP had no benchmark, so its column-walk findings were neither actionable nor
// declinable. Both typed branches are covered: each walks V and G down a column, striding
// by cols on every access.
func doraInputs(rows, cols int, dt tensor.Dtype) (v, m, g *tensor.Tensor) {
	v = tensor.New(dt, tensor.Shape{rows, cols})
	m = tensor.New(dt, tensor.Shape{cols})
	g = tensor.New(dt, tensor.Shape{rows, cols})
	set := func(t *tensor.Tensor, fn func(i int) float64) {
		for i := range t.Numel() {
			t.SetF64(fn(i), tensor.Unravel(i, t.Shape())...)
		}
	}
	set(v, func(i int) float64 { return 0.5 + math.Abs(math.Sin(float64(i)*0.013)) })
	set(m, func(i int) float64 { return 0.7 + 0.3*math.Cos(float64(i)*0.31) })
	set(g, func(i int) float64 { return math.Sin(float64(i) * 0.017) })
	return
}

func benchDoRA(b *testing.B, dt tensor.Dtype) {
	vjp := vjps[backend.OpDoRAWeight]
	v, m, g := doraInputs(1024, 512, dt)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(nil, []*tensor.Tensor{v, m}, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDoRAVJP_F64(b *testing.B) { benchDoRA(b, tensor.F64) }
func BenchmarkDoRAVJP_F32(b *testing.B) { benchDoRA(b, tensor.F32) }

func doraSum(rows, cols int, dt tensor.Dtype) uint64 {
	vjp := vjps[backend.OpDoRAWeight]
	v, m, g := doraInputs(rows, cols, dt)
	out, err := vjp(nil, []*tensor.Tensor{v, m}, nil, nil, g)
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

// TestDoRAVJPBitIdentical pins both typed branches bit-for-bit. Column counts cover every
// remainder class of a 4-way unroll, including cols below 4.
func TestDoRAVJPBitIdentical(t *testing.T) {
	for _, c := range doraGolden {
		dt := tensor.F64
		if c.f32 {
			dt = tensor.F32
		}
		if got := doraSum(c.rows, c.cols, dt); got != c.sum {
			t.Fatalf("rows=%d cols=%d f32=%v: checksum %d, want %d", c.rows, c.cols, c.f32, got, c.sum)
		}
	}
}

type doraCase struct {
	rows, cols int
	f32        bool
	sum        uint64
}
