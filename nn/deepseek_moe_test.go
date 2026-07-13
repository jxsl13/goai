package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func moeProbe(seed int, rows, dim int) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{rows, dim})
	for i := range x.Numel() {
		x.SetF64(math.Sin(float64(seed*100+i)*0.9)*0.7, tensor.Unravel(i, x.Shape())...)
	}
	return x
}

// §T551 collapse: zero shared experts ⇒ DeepSeekMoE IS SparseMoE, bit-identically
// (same seed builds the identical routed layer).
func TestDeepSeekMoECollapsesToSparse(t *testing.T) {
	const dim, hidden, E, k = 8, 16, 4, 2
	ds, err := nn.NewDeepSeekMoE(tensor.F64, dim, hidden, 0, E, k, 5)
	if err != nil {
		t.Fatal(err)
	}
	plain := nn.NewSparseMoE(tensor.F64, dim, hidden, E, k, 5)
	x := moeProbe(1, 6, dim)
	ctx := backend.NewContext()
	yd, _, err := ds.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	yp, _, err := plain.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range yp.Numel() {
		idx := tensor.Unravel(i, yp.Shape())
		if yd.AtF64(idx...) != yp.AtF64(idx...) {
			t.Fatalf("zero-shared DeepSeekMoE diverges from SparseMoE at %v", idx)
		}
	}
}

// §T551 additivity: the shared experts contribute EXACTLY Σ Shared_s(x) on top of
// the routed output — y(DeepSeekMoE) − y(Routed) == Σ shared outputs, bit-wise in
// the summation order the layer uses.
func TestDeepSeekMoESharedAdditivity(t *testing.T) {
	const dim, hidden = 8, 16
	ds, err := nn.NewDeepSeekMoE(tensor.F64, dim, hidden, 2, 4, 2, 9)
	if err != nil {
		t.Fatal(err)
	}
	x := moeProbe(2, 5, dim)
	ctx := backend.NewContext()
	full, _, err := ds.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	routed, _, err := ds.Routed.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	want := routed
	for _, s := range ds.Shared {
		out, err := s.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{want, out}, nil)
		if err != nil {
			t.Fatal(err)
		}
		want = sum[0]
	}
	for i := range want.Numel() {
		idx := tensor.Unravel(i, want.Shape())
		if full.AtF64(idx...) != want.AtF64(idx...) {
			t.Fatalf("shared additivity broken at %v", idx)
		}
	}
}

// §T551: gradients reach BOTH halves — every shared-expert weight and the router
// receive non-nil gradients through a taped forward.
func TestDeepSeekMoEGradFlow(t *testing.T) {
	ds, err := nn.NewDeepSeekMoE(tensor.F64, 8, 16, 1, 4, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	x := moeProbe(3, 4, 8)
	tape := autograd.NewTape()
	ctx := tape.Context()
	y, _, err := ds.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	sum, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(sum[0]); err != nil {
		t.Fatal(err)
	}
	for i, p := range ds.Params() {
		if tape.Grad(p) == nil {
			t.Fatalf("param %d received no gradient", i)
		}
	}
}

// DeepSeekMoE: shared experts serve every token, the router picks specialists.
func ExampleDeepSeekMoE() {
	ds, _ := nn.NewDeepSeekMoE(tensor.F64, 8, 16, 1, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 8})
	y, gates, _ := ds.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape(), gates.Shape())
	// Output: (3, 8) (3, 4)
}
