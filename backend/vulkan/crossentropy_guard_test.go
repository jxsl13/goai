//go:build vulkan && cgo

package vulkan_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

// ceGuardFixture builds a deterministic f32 logits/targets/cotangent triple, small enough
// that vulkan and ref agree bit-for-bit whenever vulkan has correctly fallen back to ref.
// masked=true puts -100 in two rows (only legal when the attrs under test actually mask
// -100; otherwise those targets are an out-of-range error on every backend).
func ceGuardFixture(masked bool) (z, tg, g *tensor.Tensor) {
	const b, c = 6, 5
	z = tensor.New(tensor.F32, tensor.Shape{b, c})
	for i := range b {
		for j := range c {
			z.SetF64(math.Sin(float64(i*c+j)*0.7)*1.5+0.25*float64(j), i, j)
		}
	}
	targets := []int{1, 4, 3, 2, 0, 2}
	if masked {
		targets = []int{1, -100, 3, -100, 0, 2}
	}
	tg = tensor.New(tensor.F32, tensor.Shape{b})
	for i, v := range targets {
		tg.SetF64(float64(v), i)
	}
	g = tensor.New(tensor.F32, tensor.Shape{})
	g.SetF64(1)
	return z, tg, g
}

// ceGuardVariants enumerates one non-default CrossEntropyAttrs per field of the struct,
// via reflection — so a field added later with no entry here fails the test rather than
// silently escaping the vulkan guard. It mirrors backend's TestCrossEntropyGuardCoversEveryField;
// this is the executable half, which actually runs the GPU kernel and compares.
func ceGuardVariants(t *testing.T) map[string]backend.CrossEntropyAttrs {
	t.Helper()
	out := map[string]backend.CrossEntropyAttrs{
		"LabelSmoothing": {LabelSmoothing: 0.1},
		"ZLoss":          {ZLoss: 1e-3},
		"IgnoreIndex":    {IgnoreIndex: -100, HasIgnoreIndex: true},
		"HasIgnoreIndex": {HasIgnoreIndex: true},
		"Reduction":      {Reduction: backend.ReductionSum},
		// Combinations at the intersection the sweep flagged as drift-prone.
		"IgnoreIndex+Smoothing": {IgnoreIndex: -100, HasIgnoreIndex: true, LabelSmoothing: 0.1},
		"IgnoreIndex+Sum":       {IgnoreIndex: -100, HasIgnoreIndex: true, Reduction: backend.ReductionSum},
		"IgnoreIndex+None+Smoothing": {IgnoreIndex: -100, HasIgnoreIndex: true,
			Reduction: backend.ReductionNone, LabelSmoothing: 0.1},
	}
	rt := reflect.TypeOf(backend.CrossEntropyAttrs{})
	for i := range rt.NumField() {
		if _, ok := out[rt.Field(i).Name]; !ok {
			t.Fatalf("CrossEntropyAttrs grew field %q with no vulkan guard variant: add one, and "+
				"confirm the fused kernel either handles it or falls back to the reference",
				rt.Field(i).Name)
		}
	}
	return out
}

// TestVulkanCrossEntropyGuardFallsBackForEveryField is the fallback invariant, executable
// half: for every non-default cross-entropy attribute, a vulkan kernel that does NOT
// implement it must produce BIT-IDENTICAL output to the reference.
//
// Bit-identical (not merely close) is the right assertion for those cases precisely
// because the expected outcome is that vulkan declines to run its fused kernel and
// delegates to ref — any numeric difference at all means a GPU kernel ran on a
// configuration it does not implement, which is the silent-wrong-answer defect this test
// exists to catch. Where a kernel DOES implement the attribute natively (the backward's
// label smoothing and z-loss) the comparison relaxes to ordinary f32 agreement.
func TestVulkanCrossEntropyGuardFallsBackForEveryField(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref := backend.Reference()

	for name, at := range ceGuardVariants(t) {
		t.Run(name, func(t *testing.T) {
			z, tg, g := ceGuardFixture(at.HasIgnoreIndex && at.IgnoreIndex == -100)
			for _, op := range []struct {
				name         string
				op           backend.Op
				in           []*tensor.Tensor
				mustDelegate bool
			}{
				// The forward kernel implements ONLY the all-default form, so every
				// variant here must delegate (IsBasic is false for all of them). The
				// backward kernel additionally implements label smoothing and z-loss on
				// the GPU, so it is only required to delegate for the row-set changing
				// fields — where it runs natively we merely check f32 agreement, which is
				// what TestVulkanCrossEntropyBackwardCrossReference already covers.
				{"forward", backend.OpCrossEntropy, []*tensor.Tensor{z, tg}, true},
				{"backward", backend.OpCrossEntropyBackward, []*tensor.Tensor{z, tg, g}, !at.IsUnmaskedMean()},
			} {
				gpu, err := backend.Execute(backend.NewContext().WithBackend(mb), op.op, op.in, at)
				if err != nil {
					t.Fatalf("%s on vulkan: %v", op.name, err)
				}
				want, err := backend.Execute(backend.NewContext().WithBackend(ref), op.op, op.in, at)
				if err != nil {
					t.Fatalf("%s on ref: %v", op.name, err)
				}
				if !gpu[0].Shape().Equal(want[0].Shape()) {
					t.Fatalf("%s: vulkan shape %v, ref %v", op.name, gpu[0].Shape(), want[0].Shape())
				}
				gs, ws := gpu[0].Contiguous().Storage().F32(), want[0].Contiguous().Storage().F32()
				for i := range gs {
					if math.IsNaN(float64(gs[i])) && math.IsNaN(float64(ws[i])) {
						continue
					}
					if op.mustDelegate {
						if gs[i] != ws[i] {
							t.Fatalf("%s elem %d: vulkan %v != ref %v — a fused GPU kernel ran on "+
								"attrs it does not implement (extend the guard)", op.name, i, gs[i], ws[i])
						}
						continue
					}
					// Natively supported here: f32 agreement only (§V11).
					if d := math.Abs(float64(gs[i] - ws[i])); d > 1e-5*math.Max(1, math.Abs(float64(ws[i]))) {
						t.Fatalf("%s elem %d: vulkan %v vs ref %v (|Δ| %.3g)", op.name, i, gs[i], ws[i], d)
					}
				}
			}
		})
	}
}
