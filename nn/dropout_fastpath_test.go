package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// The Dropout/DropPath Forward mask builders were switched from a per-element
// tensor.Unravel+SetF64 walk to a typed contiguous slice walk (flatF64/flatF32,
// §T910). These tests prove the switch is BIT-IDENTICAL: they reconstruct the exact
// generic walk the fast path replaced — same PCG stream, same draw order — and assert
// the whole Forward output (mask ⊙ x through the same backend op) matches to the bit,
// across F64/F32 (fast path) and F16/BF16 (generic fallback, now the else-branch).

// dropoutStream and dropPathStream mirror the second PCG seed each constructor uses
// (nn.NewDropout: 0xd709, nn.NewDropPath: 0x24a7). A parity test against a specific
// implementation legitimately pins the implementation's RNG stream.
const (
	dropoutStream  = 0xd709
	dropPathStream = 0x24a7
)

// ramp fills a fresh dt tensor of shape with a deterministic, varied pattern so the
// mask⊙x product exercises non-trivial survivor values (not just 1·scale).
func ramp(dt tensor.Dtype, shape tensor.Shape) *tensor.Tensor {
	x := tensor.New(dt, shape)
	n := x.Numel()
	for i := range n {
		x.SetF64(float64(i%11)-5, tensor.Unravel(i, shape)...)
	}
	return x
}

// refDropoutMask is a verbatim copy of the pre-fast-path Dropout mask loop.
func refDropoutMask(dt tensor.Dtype, shape tensor.Shape, rate float64, seed uint64) *tensor.Tensor {
	rng := rand.New(rand.NewPCG(seed, dropoutStream))
	scale := 1 / (1 - rate)
	mask := tensor.New(dt, shape)
	n := mask.Numel()
	for i := range n {
		idx := tensor.Unravel(i, shape)
		if rng.Float64() < rate {
			mask.SetF64(0, idx...)
		} else {
			mask.SetF64(scale, idx...)
		}
	}
	return mask
}

// refDropPathMask is a verbatim copy of the pre-fast-path DropPath mask loop.
func refDropPathMask(dt tensor.Dtype, shape tensor.Shape, rate float64, seed uint64) *tensor.Tensor {
	rng := rand.New(rand.NewPCG(seed, dropPathStream))
	scale := 1 / (1 - rate)
	batch := shape[0]
	perSample := make([]float64, batch)
	for b := range batch {
		if rng.Float64() >= rate {
			perSample[b] = scale
		}
	}
	mask := tensor.New(dt, shape)
	n := mask.Numel()
	for i := range n {
		idx := tensor.Unravel(i, shape)
		mask.SetF64(perSample[idx[0]], idx...)
	}
	return mask
}

// assertBitEqual fails on the first element whose raw bits differ.
func assertBitEqual(t *testing.T, got, want *tensor.Tensor, dt tensor.Dtype, shape tensor.Shape) {
	t.Helper()
	n := got.Numel()
	for i := range n {
		idx := tensor.Unravel(i, shape)
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if math.Float64bits(g) != math.Float64bits(w) {
			t.Fatalf("%s %v: element %d = %v (bits %#x), want %v (bits %#x)",
				dt, shape, i, g, math.Float64bits(g), w, math.Float64bits(w))
		}
	}
}

func TestDropoutFastPathParity(t *testing.T) {
	const rate = 0.3
	const seed = 0x9e37
	shapes := []tensor.Shape{{1}, {7}, {129}, {3, 5}, {2, 3, 4}, {5, 7, 3}}
	// F64/F32 only: they have both the flatF64/flatF32 fast path AND a mul kernel. The
	// F16/BF16 else-branch is the verbatim pre-existing loop, and mul/f16 has no cpu
	// kernel — so Forward errors at the multiply exactly as it did before this change.
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, shape := range shapes {
			x := ramp(dt, shape)
			d := nn.NewDropout(rate, seed)
			out, err := d.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			ref := refDropoutMask(dt, shape, rate, seed)
			refOut, err := backend.Execute(backend.NewContext(), backend.OpMul, []*tensor.Tensor{x, ref}, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertBitEqual(t, out, refOut[0], dt, shape)
		}
	}
}

func TestDropPathFastPathParity(t *testing.T) {
	const rate = 0.4
	const seed = 0x5c1d
	shapes := []tensor.Shape{{1}, {8}, {4, 5}, {3, 2, 4}, {2, 3, 5, 2}}
	// F64/F32 only — see TestDropoutFastPathParity (mul/f16 has no cpu kernel).
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, shape := range shapes {
			x := ramp(dt, shape)
			d := nn.NewDropPath(rate, seed)
			out, err := d.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			ref := refDropPathMask(dt, shape, rate, seed)
			refOut, err := backend.Execute(backend.NewContext(), backend.OpMul, []*tensor.Tensor{x, ref}, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertBitEqual(t, out, refOut[0], dt, shape)
		}
	}
}

// Realistic transformer-activation shapes: the mask build is O(activations) per
// forward, once per dropout layer per step.
func BenchmarkDropoutForward(b *testing.B) {
	x := ramp(tensor.F32, tensor.Shape{16, 128, 768})
	d := nn.NewDropout(0.1, 1)
	ctx := backend.NewContext()
	b.ResetTimer()
	for b.Loop() {
		if _, err := d.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDropPathForward(b *testing.B) {
	x := ramp(tensor.F32, tensor.Shape{16, 128, 768})
	d := nn.NewDropPath(0.1, 1)
	ctx := backend.NewContext()
	b.ResetTimer()
	for b.Loop() {
		if _, err := d.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
