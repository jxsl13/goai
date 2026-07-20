package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// slowPadGrad is the verbatim pre-fastpath per-element loop, the bit-identity oracle.
func slowPadGrad(grad *tensor.Tensor, pad int) *tensor.Tensor {
	rows, cols := grad.Shape()[0], grad.Shape()[1]
	out := tensor.New(grad.Dtype(), grad.Shape())
	for i := range rows {
		if i == pad {
			continue
		}
		for j := range cols {
			out.SetF64(grad.AtF64(i, j), i, j)
		}
	}
	return out
}

func TestPadGradHookFastPathBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64, tensor.F16, tensor.BF16} {
		for _, shape := range []tensor.Shape{{5, 3}, {1, 8}, {64, 48}} {
			g := tensor.Zeros(dt, shape)
			n := g.Numel()
			for i := range n {
				g.SetF64(float64(i%7)-3.5, tensor.Unravel(i, shape)...)
			}
			pad := shape[0] / 2
			emb := &Embedding{W: tensor.Zeros(dt, shape), PadIdx: pad}
			fast := emb.PadGradHook()(g)
			slow := slowPadGrad(g, pad)
			for i := range n {
				idx := tensor.Unravel(i, shape)
				if math.Float64bits(fast.AtF64(idx...)) != math.Float64bits(slow.AtF64(idx...)) {
					t.Fatalf("dt=%v shape=%v elem %d: fast %v != slow %v", dt, shape, i, fast.AtF64(idx...), slow.AtF64(idx...))
				}
			}
			// The pad row must be exactly zero.
			for j := range shape[1] {
				if fast.AtF64(pad, j) != 0 {
					t.Fatalf("dt=%v: pad row not zeroed at col %d", dt, j)
				}
			}
		}
	}
}

// Non-contiguous gradient falls through to the general path and stays correct.
func TestPadGradHookNonContiguous(t *testing.T) {
	base := tensor.Zeros(tensor.F64, tensor.Shape{3, 4})
	n := base.Numel()
	for i := range n {
		base.SetF64(float64(i)+1, tensor.Unravel(i, base.Shape())...)
	}
	view, err := base.Transpose(0, 1) // [4,3] non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	emb := &Embedding{W: tensor.Zeros(tensor.F64, tensor.Shape{4, 3}), PadIdx: 1}
	fast := emb.PadGradHook()(view)
	slow := slowPadGrad(view, 1)
	for i := range view.Numel() {
		idx := tensor.Unravel(i, view.Shape())
		if fast.AtF64(idx...) != slow.AtF64(idx...) {
			t.Fatalf("non-contiguous elem %d: %v != %v", i, fast.AtF64(idx...), slow.AtF64(idx...))
		}
	}
}

func BenchmarkPadGradHookFast(b *testing.B) {
	g := tensor.Zeros(tensor.F32, tensor.Shape{32000, 512})
	emb := &Embedding{W: tensor.Zeros(tensor.F32, tensor.Shape{32000, 512}), PadIdx: 0}
	h := emb.PadGradHook()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h(g)
	}
}

func BenchmarkPadGradHookSlow(b *testing.B) {
	g := tensor.Zeros(tensor.F32, tensor.Shape{32000, 512})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowPadGrad(g, 0)
	}
}
