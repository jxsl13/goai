package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Guards cpu's cross-entropy against ref — but ONLY when cpu actually provides it.
//
// cpu/crossentropy.go registers itself inside `if vexpF32Fast`, the SIMD perf build,
// and for F32 only. In the DEFAULT build cpu registers nothing for OpCrossEntropy
// and the dispatcher falls back to ref, so an unguarded parity test here would
// compare ref against ref and pass while exercising none of the cpu code. That was
// this test's first form, and a panic probe caught it: neither cpu/crossentropy.go
// line 86 nor line 98 executed (PROC-012 — confirm the line runs before believing a
// probe).
//
// The skip below makes the vacuity visible instead of silent. Under the SIMD build
// the test becomes meaningful, and cpu and ref were measured to agree bit-for-bit
// there before this was relied upon.
func TestCPUCrossEntropyBitIdenticalToRef(t *testing.T) {
	rb, ok := backend.Get(backend.Ref)
	if !ok {
		t.Skip("ref backend unavailable")
	}
	cb, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend unavailable")
	}
	// Without a cpu-registered kernel the dispatcher falls back to ref and this
	// degenerates into ref-vs-ref. Skip loudly rather than pass vacuously.
	if _, has := cb.Kernel(backend.OpCrossEntropy, tensor.F32); !has {
		t.Skip("cpu registers no OpCrossEntropy kernel in this build (needs the vexpF32Fast SIMD build); parity would be ref-vs-ref")
	}
	for _, sz := range [][2]int{{1, 2}, {4, 3}, {8, 5}, {17, 11}} {
		b, c := sz[0], sz[1]
		for _, f32 := range []bool{false, true} {
			sh := tensor.Shape{b, c}
			seed := uint64(b*10 + c)
			var z *tensor.Tensor
			if f32 {
				z = bench.RandF32(sh, seed)
			} else {
				z = bench.RandF64(sh, seed)
			}
			td := make([]float64, b)
			for i := range td {
				td[i] = float64(i % c)
			}
			tg := tensor.FromFloat64(tensor.Shape{b}, td)
			if f32 {
				tg = tg.Cast(tensor.F32)
			}
			in := []*tensor.Tensor{z, tg}
			want, err := backend.Execute(backend.NewContext().WithBackend(rb), backend.OpCrossEntropy, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpCrossEntropy, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			// Bit-exact for F64 and on the default build; the f32-native SIMD lane
			// is compared within its own ADR budget (see f32NativeTolerant).
			tolerant := f32NativeTolerant(f32)
			var maxRel float64
			for o := range want {
				for i := range want[o].Numel() {
					co := tensor.Unravel(i, want[o].Shape())
					g, w := got[o].AtF64(co...), want[o].AtF64(co...)
					if tolerant {
						if !parityCloseF32(g, w) {
							t.Fatalf("b=%d c=%d f32=%v out[%d] elem %d: cpu %v vs ref %v exceeds the f32-native budget", b, c, f32, o, i, g, w)
						}
						if r := parityRelErr(g, w); r > maxRel {
							maxRel = r
						}
						continue
					}
					if math.Float64bits(g) != math.Float64bits(w) {
						t.Fatalf("b=%d c=%d f32=%v out[%d] elem %d: cpu %v != ref %v", b, c, f32, o, i, g, w)
					}
				}
			}
			if tolerant {
				t.Logf("b=%d c=%d: f32-native max rel err %.2e", b, c, maxRel)
			}
		}
	}
}
