package cpu_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// §V3/§V11 CROSS for the vexp-routed F32 cross-entropy (§T664): on the default
// build (and amd64, and F64 everywhere) the cpu backend registers no
// crossentropy kernels, so dispatch falls back to ref — asserted BIT-EXACT
// here. On the arm64 perf build the F32 kernels run the vexpRowF32 pipeline
// (crossentropy.go) — asserted within the ADR-0021 tolerant budget (rtol 2e-3,
// the 4-lane f32 row sum at c=4096 is ~6e-5). geluF32Tolerant gates on exactly
// the vexpNeon build combination. Covers label smoothing, z-loss, a d%4 tail
// shape, and the [256,4096] training shape (parallel rows).
func TestCPUCrossEntropyParity(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	rng := rand.New(rand.NewPCG(43, 47))

	mk := func(b, c int, scale float32) (*tensor.Tensor, *tensor.Tensor) {
		z := tensor.New(tensor.F32, tensor.Shape{b, c})
		zs := z.Storage().F32()
		for i := range zs {
			zs[i] = scale * (rng.Float32() - 0.5)
		}
		tg := tensor.New(tensor.F32, tensor.Shape{b})
		ts := tg.Storage().F32()
		for i := range ts {
			ts[i] = float32(rng.IntN(c))
		}
		return z, tg
	}
	upstream := tensor.New(tensor.F32, tensor.Shape{})
	upstream.SetF64(1.5)

	cases := []struct {
		label string
		b, c  int
		scale float32
		attrs backend.CrossEntropyAttrs
	}{
		{"plain[7x33]", 7, 33, 8, backend.CrossEntropyAttrs{}}, // c%4 tail lanes
		{"plain[64x512]", 64, 512, 12, backend.CrossEntropyAttrs{}},
		{"train[256x4096]", 256, 4096, 10, backend.CrossEntropyAttrs{}}, // parallel rows
		{"smooth[64x512]", 64, 512, 12, backend.CrossEntropyAttrs{LabelSmoothing: 0.1}},
		{"zloss[64x512]", 64, 512, 12, backend.CrossEntropyAttrs{ZLoss: 1e-4}},
		{"smooth+zloss[32x128]", 32, 128, 20, backend.CrossEntropyAttrs{LabelSmoothing: 0.05, ZLoss: 1e-3}},
	}
	const rtol, atol = 2e-3, 1e-9
	for _, tc := range cases {
		z, tg := mk(tc.b, tc.c, tc.scale)

		// Forward loss.
		lc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpCrossEntropy, []*tensor.Tensor{z, tg}, tc.attrs)
		if err != nil {
			t.Fatalf("%s cpu fwd: %v", tc.label, err)
		}
		lr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpCrossEntropy, []*tensor.Tensor{z, tg}, tc.attrs)
		if err != nil {
			t.Fatalf("%s ref fwd: %v", tc.label, err)
		}
		gl, rl := lc[0].AtF64(), lr[0].AtF64()
		if geluF32Tolerant {
			if math.Abs(gl-rl) > atol+rtol*math.Abs(rl) {
				t.Fatalf("%s: loss cpu %v vs ref %v", tc.label, gl, rl)
			}
		} else if gl != rl {
			t.Fatalf("%s: loss cpu %v != ref %v (default build must be exact)", tc.label, gl, rl)
		}

		// Backward gradient.
		dc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpCrossEntropyBackward, []*tensor.Tensor{z, tg, upstream}, tc.attrs)
		if err != nil {
			t.Fatalf("%s cpu bwd: %v", tc.label, err)
		}
		dr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpCrossEntropyBackward, []*tensor.Tensor{z, tg, upstream}, tc.attrs)
		if err != nil {
			t.Fatalf("%s ref bwd: %v", tc.label, err)
		}
		cg, rg := dc[0].Storage().F32(), dr[0].Storage().F32()
		var maxRel float64
		for i := range cg {
			cf, rf := float64(cg[i]), float64(rg[i])
			// atol floor: p underflows to ~0 for logits far below the row max;
			// scaled by g/b the gradient there is ≤ ~1e-9 either way.
			budget := 1e-9 + rtol*math.Abs(rf)
			if !geluF32Tolerant {
				budget = 0 // default build: cpu IS the ref fallback
			}
			if d := math.Abs(cf - rf); d > budget {
				t.Fatalf("%s: dz[%d] cpu %v vs ref %v", tc.label, i, cg[i], rg[i])
			}
			if math.Abs(rf) > 1e-12 {
				if rel := math.Abs(cf-rf) / math.Abs(rf); rel > maxRel {
					maxRel = rel
				}
			}
		}
		t.Logf("%s: loss cpu %.9g ref %.9g; dz max rel err %.2e (tolerant=%v)", tc.label, gl, rl, maxRel, geluF32Tolerant)
	}

	// Error semantics preserved: out-of-range target on the forward.
	z, tg := mk(4, 16, 8)
	tg.Storage().F32()[2] = 99
	if _, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpCrossEntropy, []*tensor.Tensor{z, tg}, nil); err == nil {
		t.Fatal("out-of-range target: want error, got nil")
	}
}

// --- §V22 A/B benchmarks (§T664): training loss shape [seq=256, vocab=4096].
// On the arm64 perf build _cpu runs the vexp kernels; _ref is the scalar-f64
// serial baseline (and on every other build _cpu falls back to it). ---

func benchCE(b *testing.B, name backend.Name, op backend.Op) {
	be, _ := backend.Get(name)
	ctx := backend.NewContext().WithBackend(be)
	rng := rand.New(rand.NewPCG(5, 9))
	z := tensor.New(tensor.F32, tensor.Shape{256, 4096})
	zs := z.Storage().F32()
	for i := range zs {
		zs[i] = 10 * (rng.Float32() - 0.5)
	}
	tg := tensor.New(tensor.F32, tensor.Shape{256})
	ts := tg.Storage().F32()
	for i := range ts {
		ts[i] = float32(rng.IntN(4096))
	}
	ins := []*tensor.Tensor{z, tg}
	if op == backend.OpCrossEntropyBackward {
		g := tensor.New(tensor.F32, tensor.Shape{})
		g.SetF64(1)
		ins = append(ins, g)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCrossEntropyF32_256x4096_cpu(b *testing.B) {
	benchCE(b, backend.CPU, backend.OpCrossEntropy)
}
func BenchmarkCrossEntropyF32_256x4096_ref(b *testing.B) {
	benchCE(b, backend.Ref, backend.OpCrossEntropy)
}
func BenchmarkCrossEntropyBackwardF32_256x4096_cpu(b *testing.B) {
	benchCE(b, backend.CPU, backend.OpCrossEntropyBackward)
}
func BenchmarkCrossEntropyBackwardF32_256x4096_ref(b *testing.B) {
	benchCE(b, backend.Ref, backend.OpCrossEntropyBackward)
}
