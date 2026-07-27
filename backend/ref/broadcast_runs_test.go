package ref

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// broadcastShapes covers the three regimes the run-length hoist must handle:
// a trailing verbatim row (eff[last]==1), a trailing broadcast axis
// (eff[last]==0, the replicate path), and a mixed case where the run length and
// the odometer tick interact. A wrong run length shows up here and nowhere else.
var broadcastShapes = []struct {
	name string
	in   tensor.Shape
	out  tensor.Shape
}{
	{"row_256_to_256x256", tensor.Shape{256}, tensor.Shape{256, 256}},
	{"col_256x1_to_256x256", tensor.Shape{256, 1}, tensor.Shape{256, 256}},
	{"mixed_1x256x1_to_8x256x16", tensor.Shape{1, 256, 1}, tensor.Shape{8, 256, 16}},
	{"scalar_1_to_4x3", tensor.Shape{1}, tensor.Shape{4, 3}},
}

// broadcastNaive recomputes the expected output with the per-element form the
// hoisted kernel replaced: Unravel each output position, zero the broadcast
// axes, read the source. It is deliberately the slow, obviously-correct shape.
func broadcastNaive(t *testing.T, x *tensor.Tensor, outShape tensor.Shape) []float64 {
	t.Helper()
	offset, err := backend.BroadcastPlan(x.Shape(), outShape)
	if err != nil {
		t.Fatalf("BroadcastPlan: %v", err)
	}
	xs := x.Shape()
	ic := make([]int, x.Ndim())
	want := make([]float64, outShape.Numel())
	for pos := range want {
		oc := tensor.Unravel(pos, outShape)
		for a := range xs {
			if xs[a] == 1 {
				ic[a] = 0
			} else {
				ic[a] = oc[a+offset]
			}
		}
		want[pos] = x.AtF64(ic...)
	}
	return want
}

// TestBroadcastRunsMatchesPerElement pins the run-length hoist bit-for-bit
// against the per-element traversal. The op does no arithmetic — it is a
// same-dtype copy — so exact equality is the correct bar, not a tolerance.
func TestBroadcastRunsMatchesPerElement(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, sh := range broadcastShapes {
			t.Run(fmt.Sprintf("%v/%s", dt, sh.name), func(t *testing.T) {
				x := tensor.New(dt, sh.in)
				for i := range x.Numel() {
					// Values distinct per position so any mis-indexing is visible.
					x.SetF64(float64(i)*0.5+1, tensor.Unravel(i, sh.in)...)
				}
				want := broadcastNaive(t, x, sh.out)

				ctx := backend.NewContext()
				got, err := broadcastKernel(ctx, []*tensor.Tensor{x},
					backend.BroadcastAttrs{Shape: sh.out})
				if err != nil {
					t.Fatalf("broadcastKernel: %v", err)
				}
				o := got[0]
				if o.Numel() != len(want) {
					t.Fatalf("numel = %d, want %d", o.Numel(), len(want))
				}
				for i, w := range want {
					if g := o.AtF64(tensor.Unravel(i, sh.out)...); g != w {
						t.Fatalf("element %d = %v, want %v (not bit-identical)", i, g, w)
					}
				}
			})
		}
	}
}

func benchBroadcast(b *testing.B, in, out tensor.Shape) {
	x := tensor.New(tensor.F64, in)
	ctx := backend.NewContext()
	attrs := backend.BroadcastAttrs{Shape: out}
	b.ReportAllocs()
	b.SetBytes(int64(out.Numel() * 8))
	for b.Loop() {
		if _, err := broadcastKernel(ctx, []*tensor.Tensor{x}, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBroadcastReplicateF64 covers eff[last]==0 (the fill path), which the
// pre-existing BenchmarkBroadcastF64_256to256x256 never exercises.
func BenchmarkBroadcastReplicateF64(b *testing.B) {
	benchBroadcast(b, tensor.Shape{256, 1}, tensor.Shape{256, 256})
}

// BenchmarkBroadcastMixedF64 exercises the run-length computation against a
// non-trivial odometer tick.
func BenchmarkBroadcastMixedF64(b *testing.B) {
	benchBroadcast(b, tensor.Shape{1, 256, 1}, tensor.Shape{8, 256, 16})
}
