package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestPoolBackwardTypedArmsMatchAccessorPath locks the typed pooling arms to the accessor path they
// bypass. The reference runs through poolAccessors itself — the same closures the exotic-dtype
// fallback uses — so the two differ in exactly one respect: whether each read and write goes
// through an indirect call.
//
// The comparison is exact. For F32 that is the substantive part: the accessor path widens every
// read to f64 and narrows after every add, and the typed arm has to reproduce that rounding rather
// than accumulate in f32, which would be both faster and wrong.
//
// The fixture deliberately includes a NaN window. Max pooling routes by equality, and NaN fails
// every comparison, so that window takes the fallback branch — the one path a benign fixture never
// reaches and the one most likely to be transcribed wrongly.
func TestPoolBackwardTypedArmsMatchAccessorPath(t *testing.T) {
	// OVERLAPPING windows (stride below kernel) are required, not incidental. With k=2,s=2 every
	// input element receives at most ONE accumulation, and 0+gv narrows the same whether the add is
	// done in f32 or widened to f64 first — so a fixture without overlap cannot tell the two apart,
	// and a mutation that accumulates in f32 passes it. k=3,s=1 makes interior elements receive
	// nine adds, where the rounding difference is visible.
	const n, c, hw, k, s = 2, 3, 8, 3, 1
	ho := (hw-k)/s + 1
	for _, op := range []backend.Op{backend.OpMaxPool2D, backend.OpAvgPool2D} {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			x := tensor.New(dt, tensor.Shape{n, c, hw, hw})
			for i := range x.Numel() {
				x.SetF64(math.Sin(float64(i)*0.11)*2, i/(c*hw*hw), (i/(hw*hw))%c, (i/hw)%hw, i%hw)
			}
			y := tensor.New(dt, tensor.Shape{n, c, ho, ho})
			g := tensor.New(dt, tensor.Shape{n, c, ho, ho})
			for pl := range n * c {
				for oy := range ho {
					for ox := range ho {
						m, sum := math.Inf(-1), 0.0
						for ky := range k {
							for kx := range k {
								v := x.AtF64(pl/c, pl%c, oy*s+ky, ox*s+kx)
								sum += v
								if v > m {
									m = v
								}
							}
						}
						if op == backend.OpAvgPool2D {
							m = sum / float64(k*k)
						}
						// One NaN OUTPUT, which is what actually reaches the fallback: max
						// pooling routes by equality against y, and a NaN in x alone does not
						// do it — the window max is then simply the largest non-NaN element and
						// the search finds it. Only a NaN y makes every comparison fail.
						if op == backend.OpMaxPool2D && pl == 0 && oy == 0 && ox == 0 {
							m = math.NaN()
						}
						y.SetF64(m, pl/c, pl%c, oy, ox)
						g.SetF64(math.Cos(float64(pl*3+oy*5+ox))*0.7, pl/c, pl%c, oy, ox)
					}
				}
			}
			got, err := vjps[op](nil, []*tensor.Tensor{x}, []*tensor.Tensor{y},
				backend.PoolAttrs{Kernel: k, Stride: s}, g)
			if err != nil {
				t.Fatalf("%v %v: %v", op, dt, err)
			}
			want := poolBackViaAccessors(op, x, y, g, n*c, hw, hw, ho, ho, k, s)
			for i := range x.Numel() {
				idx := tensor.Unravel(i, x.Shape())
				a, b := got[0].AtF64(idx...), want.AtF64(idx...)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("%v %v element %d: typed %v, accessor %v — not bit-identical",
						op, dt, i, a, b)
				}
			}
		}
	}
}

// poolBackViaAccessors is the pre-devirtualization loop, kept verbatim and still driven through
// poolAccessors, so it is the arm the typed paths must equal.
func poolBackViaAccessors(op backend.Op, x, y, g *tensor.Tensor, planes, h, w, ho, wo, k, s int) *tensor.Tensor {
	gx := tensor.New(x.Dtype(), x.Shape())
	getX, getY, getG, addGX := poolAccessors(x, y, g, gx)
	inv := 1 / float64(k*k)
	for pl := range planes {
		xB, yB := pl*h*w, pl*ho*wo
		for oy := range ho {
			for ox := range wo {
				if op == backend.OpAvgPool2D {
					gv := getG(yB+oy*wo+ox) * inv
					for ky := range k {
						row := xB + (oy*s+ky)*w + ox*s
						for kx := range k {
							addGX(row+kx, gv)
						}
					}
					continue
				}
				m := getY(yB + oy*wo + ox)
				gv := getG(yB + oy*wo + ox)
				routed := false
				for ky := 0; ky < k && !routed; ky++ {
					row := xB + (oy*s+ky)*w + ox*s
					for kx := 0; kx < k && !routed; kx++ {
						if getX(row+kx) == m {
							addGX(row+kx, gv)
							routed = true
						}
					}
				}
				if !routed {
					addGX(xB+oy*s*w+ox*s, gv)
				}
			}
		}
	}
	return gx
}
