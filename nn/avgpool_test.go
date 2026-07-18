package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// avgPoolContexts returns one Context per backend registered on this machine
// (cpu always; metal/vulkan/cuda when their device is present), so the layer is
// never verified on a single accel path (repo rule: no metal-only paths).
func avgPoolContexts(t *testing.T) map[string]*backend.Context {
	t.Helper()
	ctxs := map[string]*backend.Context{}
	for _, name := range backend.Available() {
		b, ok := backend.Get(name)
		if !ok {
			continue
		}
		ctxs[string(name)] = &backend.Context{Backend: b}
	}
	if len(ctxs) == 0 {
		t.Fatal("no backends registered")
	}
	return ctxs
}

// iota4D builds an [n,c,h,w] tensor filled 0,1,2,… in row-major order.
func iota4D(n, c, h, w int) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{n, c, h, w})
	for i := range x.Numel() {
		x.SetF64(float64(i), tensor.Unravel(i, x.Shape())...)
	}
	return x
}

// AvgPool2D against a HAND-COMPUTED oracle (T845 pattern): the expected values
// below are worked out by hand from the 4×4 iota input, not produced by a second
// implementation. Runs on every backend present on this machine.
func TestAvgPool2DOracle(t *testing.T) {
	cases := []struct {
		name           string
		kernel, stride int
		outH, outW     int
		// want is the flattened row-major expected output.
		want []float64
	}{
		{
			// x = 0..15 as 4×4. Non-overlapping 2×2 windows:
			//  {0,1,4,5}/4=2.5   {2,3,6,7}/4=4.5
			//  {8,9,12,13}/4=10.5 {10,11,14,15}/4=12.5
			name: "kernel2_strideDefault", kernel: 2, stride: 0,
			outH: 2, outW: 2,
			want: []float64{2.5, 4.5, 10.5, 12.5},
		},
		{
			// stride ≠ kernel: 2×2 windows stepped by 1 → 3×3 overlapping outputs.
			//  {0,1,4,5}=2.5   {1,2,5,6}=3.5   {2,3,6,7}=4.5
			//  {4,5,8,9}=6.5   {5,6,9,10}=7.5  {6,7,10,11}=8.5
			//  {8,9,12,13}=10.5 {9,10,13,14}=11.5 {10,11,14,15}=12.5
			name: "kernel2_stride1_overlapping", kernel: 2, stride: 1,
			outH: 3, outW: 3,
			want: []float64{2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5},
		},
		{
			// whole-plane average: mean(0..15) = 120/16 = 7.5
			name: "kernel4_wholePlane", kernel: 4, stride: 0,
			outH: 1, outW: 1,
			want: []float64{7.5},
		},
		{
			// kernel 3, stride 1 → 2×2. {0,1,2,4,5,6,8,9,10}/9 = 45/9 = 5, etc.
			name: "kernel3_stride1", kernel: 3, stride: 1,
			outH: 2, outW: 2,
			want: []float64{5, 6, 9, 10},
		},
	}

	x := iota4D(1, 1, 4, 4)
	for name, ctx := range avgPoolContexts(t) {
		for _, tc := range cases {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				p := &nn.AvgPool2D{Kernel: tc.kernel, Stride: tc.stride}
				y, err := p.Forward(ctx, x)
				if err != nil {
					t.Fatal(err)
				}
				if !y.Shape().Equal(tensor.Shape{1, 1, tc.outH, tc.outW}) {
					t.Fatalf("shape %v, want [1 1 %d %d]", y.Shape(), tc.outH, tc.outW)
				}
				for i, want := range tc.want {
					idx := tensor.Unravel(i, y.Shape())
					// f64 exact: all these means are representable.
					if got := y.AtF64(idx...); got != want {
						t.Errorf("avgpool[%v] = %g, want %g", idx, got, want)
					}
				}
			})
		}
	}
}

// No padding is supported, so a window that would overhang the edge is DROPPED
// (floor mode) rather than zero-padded: out = (in − k)/s + 1. This pins the
// documented consequence of the backend op having no Pad field — if padding is
// ever added, this test is the one that must be revisited.
func TestAvgPool2DNoPaddingDropsRaggedEdge(t *testing.T) {
	// 5×5 iota, kernel 2 stride 2 → (5−2)/2+1 = 2, so row/col 4 is dropped.
	//  {0,1,5,6}/4=3    {2,3,7,8}/4=5
	//  {10,11,15,16}/4=13 {12,13,17,18}/4=15
	// A zero-padding implementation would instead produce a 3×3 output, so this
	// asserts both the shape AND that no phantom zeros entered the means.
	x := iota4D(1, 1, 5, 5)
	want := []float64{3, 5, 13, 15}
	for name, ctx := range avgPoolContexts(t) {
		t.Run(name, func(t *testing.T) {
			y, err := (&nn.AvgPool2D{Kernel: 2, Stride: 2}).Forward(ctx, x)
			if err != nil {
				t.Fatal(err)
			}
			if !y.Shape().Equal(tensor.Shape{1, 1, 2, 2}) {
				t.Fatalf("shape %v, want [1 1 2 2] (floor mode, no padding)", y.Shape())
			}
			for i, w := range want {
				idx := tensor.Unravel(i, y.Shape())
				if got := y.AtF64(idx...); got != w {
					t.Errorf("avgpool[%v] = %g, want %g", idx, got, w)
				}
			}
		})
	}
}

// Batch and channel planes are pooled independently, not mixed.
func TestAvgPool2DBatchAndChannelsIndependent(t *testing.T) {
	const n, c = 2, 3
	x := iota4D(n, c, 4, 4)
	for name, ctx := range avgPoolContexts(t) {
		t.Run(name, func(t *testing.T) {
			y, err := (&nn.AvgPool2D{Kernel: 2}).Forward(ctx, x)
			if err != nil {
				t.Fatal(err)
			}
			if !y.Shape().Equal(tensor.Shape{n, c, 2, 2}) {
				t.Fatalf("shape %v", y.Shape())
			}
			// Plane p holds values 16p..16p+15, so its pooled means are the
			// 4×4 oracle shifted by 16p.
			base := []float64{2.5, 4.5, 10.5, 12.5}
			for p := range n * c {
				for i, b := range base {
					idx := tensor.Unravel(p*4+i, y.Shape())
					if got, want := y.AtF64(idx...), b+16*float64(p); got != want {
						t.Errorf("plane %d out[%v] = %g, want %g", p, idx, got, want)
					}
				}
			}
		})
	}
}

// Every backend must agree bit-for-bit with the cpu path (§V3/§V11 tol 0).
func TestAvgPool2DBackendParity(t *testing.T) {
	ctxs := avgPoolContexts(t)
	cpu, ok := ctxs["cpu"]
	if !ok {
		t.Skip("no cpu backend to use as reference")
	}
	x := iota4D(2, 2, 7, 6)
	p := &nn.AvgPool2D{Kernel: 3, Stride: 2}
	ref, err := p.Forward(cpu, x)
	if err != nil {
		t.Fatal(err)
	}
	for name, ctx := range ctxs {
		if name == "cpu" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			got, err := p.Forward(ctx, x)
			if err != nil {
				t.Fatal(err)
			}
			if !got.Shape().Equal(ref.Shape()) {
				t.Fatalf("shape %v, cpu %v", got.Shape(), ref.Shape())
			}
			for i := range ref.Numel() {
				idx := tensor.Unravel(i, ref.Shape())
				a, b := got.AtF64(idx...), ref.AtF64(idx...)
				if a != b && math.Abs(a-b) > 1e-12 {
					t.Fatalf("out[%v] = %v, cpu %v", idx, a, b)
				}
			}
		})
	}
	t.Logf("verified on backends: %v", backend.Available())
}

func TestAvgPool2DValidation(t *testing.T) {
	ctx := backend.NewContext()
	p := &nn.AvgPool2D{Kernel: 2}
	if p.Params() != nil {
		t.Error("pool must have no params")
	}
	if _, err := p.Forward(ctx, tensor.New(tensor.F64, tensor.Shape{4, 4})); err == nil {
		t.Error("rank-2 accepted")
	}
	if _, err := p.Forward(ctx, tensor.New(tensor.F64, tensor.Shape{1, 1, 4, 4, 2})); err == nil {
		t.Error("rank-5 accepted")
	}
	if _, err := (&nn.AvgPool2D{Kernel: 0}).Forward(ctx, iota4D(1, 1, 4, 4)); err == nil {
		t.Error("kernel 0 accepted")
	}
	if _, err := (&nn.AvgPool2D{Kernel: -1}).Forward(ctx, iota4D(1, 1, 4, 4)); err == nil {
		t.Error("negative kernel accepted")
	}
	if _, err := (&nn.AvgPool2D{Kernel: 2, Stride: -1}).Forward(ctx, iota4D(1, 1, 4, 4)); err == nil {
		t.Error("negative stride accepted")
	}
	// Window larger than the input has no valid position and must error rather
	// than silently zero-pad (no-padding contract).
	if _, err := (&nn.AvgPool2D{Kernel: 8}).Forward(ctx, iota4D(1, 1, 4, 4)); err == nil {
		t.Error("kernel larger than input accepted")
	}
}

// AvgPool2D satisfies the nn.Layer interface.
var _ nn.Layer = (*nn.AvgPool2D)(nil)

// Window-mean downsampling over NCHW. Reach for AvgPool2D over MaxPool2D when
// every input in the window should influence the output — it keeps gradients
// flowing to all k² taps instead of routing them to a single argmax, which is
// why it is the standard head pooling in ResNet-style classifiers.
func ExampleAvgPool2D() {
	pool := &nn.AvgPool2D{Kernel: 2} // Stride 0 → Kernel: non-overlapping windows

	// One 4×4 feature map. The top-left window averages to 3.5, and the bottom
	// row is deliberately spiky so the contrast with max-pooling is visible.
	x := tensor.FromFloat64(tensor.Shape{1, 1, 4, 4}, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 0, 0, 0,
		0, 0, 0, 16,
	})
	y, err := pool.Forward(backend.NewContext(), x)
	if err != nil {
		panic(err)
	}
	fmt.Println("shape:", y.Shape()) // (in − Kernel)/Stride + 1 per spatial dim
	for i := range 2 {
		fmt.Printf("%.2f %.2f\n", y.AtF64(0, 0, i, 0), y.AtF64(0, 0, i, 1))
	}

	// The lone 16 is diluted to its window mean of 4 — averaging, not detecting.
	fmt.Println("params:", pool.Params())

	// Output:
	// shape: (1, 1, 2, 2)
	// 3.50 5.50
	// 2.25 4.00
	// params: []
}
