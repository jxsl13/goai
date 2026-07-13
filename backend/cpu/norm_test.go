package cpu_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// §T596/§V9: the parallel norm kernels match ref within a few ULPs on both
// dtypes, forward and backward, across serial (small) and parallel (large)
// shapes. NOT bit-exact by design: the Go spec permits context-dependent FMA
// contraction (even across statements), so gc fuses differently between the
// ref and cpu function bodies — see the package comment in norm.go.
func TestNormKernelsMatchRefWithinUlps(t *testing.T) {
	shapes := []tensor.Shape{{3, 8}, {64, 512}, {257, 1024}} // serial + parallel
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, sh := range shapes {
			d := sh[len(sh)-1]
			x := tensor.Randn(dt, 1, sh)
			gamma := tensor.Randn(dt, 2, tensor.Shape{d})
			beta := tensor.Randn(dt, 3, tensor.Shape{d})
			g := tensor.Randn(dt, 4, sh)
			ctx := backend.NewContext()

			cpuB := cpuBackend(t)
			refB, _ := backend.Get(backend.Ref)
			for _, tc := range []struct {
				name string
				op   backend.Op
				in   []*tensor.Tensor
			}{
				{"rmsnorm", backend.OpRMSNorm, []*tensor.Tensor{x, gamma}},
				{"layernorm", backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}},
				{"rmsnorm-bwd", backend.OpRMSNormBackward, []*tensor.Tensor{x, gamma, g}},
				{"layernorm-bwd", backend.OpLayerNormBackward, []*tensor.Tensor{x, gamma, g}},
			} {
				got, err := backend.Execute(ctx.WithBackend(cpuB), tc.op, tc.in, nil)
				if err != nil {
					t.Fatalf("%s cpu: %v", tc.name, err)
				}
				want, err := backend.Execute(ctx.WithBackend(refB), tc.op, tc.in, nil)
				if err != nil {
					t.Fatalf("%s ref: %v", tc.name, err)
				}
				for o := range want {
					assertCloseUlps(t, got[o], want[o], fmt.Sprintf("%s/%s/%v/out%d", tc.name, dt, sh, o))
				}
			}
		}
	}
}

// assertCloseUlps compares within a tight relative tolerance (f64: ~4 ulps;
// f32 tensors were computed in f64 and rounded, so 1e-6 relative).
func assertCloseUlps(t *testing.T, got, want *tensor.Tensor, label string) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s: shape %v != %v", label, got.Shape(), want.Shape())
	}
	rtol := 1e-15
	if got.Dtype() == tensor.F32 {
		// ref accumulates cross-row sums THROUGH an F32 tensor (one rounding per
		// row); the cpu kernels sum in f64. The gap grows ~rows·2⁻²⁴, so the f32
		// budget covers the 257-row test case with margin while still catching
		// real math errors (those show at ≥1e-3).
		rtol = 5e-5
	}
	n := got.Numel()
	for i := range n {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		tol := rtol * math.Max(1, math.Max(math.Abs(g), math.Abs(w)))
		if math.Abs(g-w) > tol {
			t.Fatalf("%s [%d]: cpu %v != ref %v (tol %g)", label, i, g, w, tol)
		}
	}
}

// §T598/§V9: the parallel softmax matches ref within ulps across serial and
// parallel shapes and both dtypes (same standard and reasoning as the norms).
func TestSoftmaxKernelMatchesRefWithinUlps(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, sh := range []tensor.Shape{{3, 8}, {64, 512}, {257, 1024}} {
			x := tensor.Randn(dt, 9, sh)
			ctx := backend.NewContext()
			cpuB := cpuBackend(t)
			refB, _ := backend.Get(backend.Ref)
			got, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpSoftmax, []*tensor.Tensor{x}, nil)
			if err != nil {
				t.Fatal(err)
			}
			want, err := backend.Execute(ctx.WithBackend(refB), backend.OpSoftmax, []*tensor.Tensor{x}, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertCloseUlps(t, got[0], want[0], fmt.Sprintf("softmax/%s/%v", dt, sh))
		}
	}
}
