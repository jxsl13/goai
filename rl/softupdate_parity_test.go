package rl

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestSoftUpdateParallelBitIdentical is the §V22 gate for fanning out the Polyak blend.
//
// Tolerance is ZERO. The blend carries no accumulation — every output index is written once
// from its own source element and its own previous value — so chunking cannot legitimately
// move a bit.
//
// THE REFERENCE MUST REPRODUCE THE IMPLEMENTATION'S EXPRESSION SHAPE, and that is not a
// stylistic point. The first version of this test computed the reference through
// Unravel/AtF64 into a separate slice with tau declared as a Go constant, and it failed by
// exactly 1 ulp on 65536-element parameters. The change under test was innocent: comparing
// the parallel result against a flat-slice serial loop with tau as a variable — the form the
// pre-change implementation actually used — gives zero differences across every parameter.
// The 1 ulp came from FMA contraction differing between the two expression shapes, so the
// original test was measuring the compiler, not the change. A reference written for
// readability rather than for shape-fidelity silently tests different arithmetic.
//
// The width is chosen so at least one parameter is ABOVE softUpdateParThreshold and at least
// one is below, covering both arms of the size guard: 256x256 = 65536 clears the threshold,
// the 256-element biases do not.
func TestSoftUpdateParallelBitIdentical(t *testing.T) {
	if 256*256 <= softUpdateParThreshold {
		t.Fatalf("test geometry no longer exercises the parallel arm: 65536 vs threshold %d", softUpdateParThreshold)
	}
	mk := func(seed uint64) *nn.Sequential {
		return nn.NewSequential(
			nn.NewLinear(tensor.F64, 256, 256, seed),
			nn.NewLinear(tensor.F64, 256, 1, seed+1),
		)
	}
	tau := 0.005 // a variable, exactly as softUpdate receives it
	got, want := mk(1), mk(2)
	ref, refTarget := mk(1), mk(2)

	// The pre-change implementation, transcribed verbatim: flat contiguous slices, ascending
	// index, one fused expression per element.
	src, dst := ref.Params(), refTarget.Params()
	for i := range src {
		so := src[i].Storage().F64()
		to := dst[i].Storage().F64()
		for j := range to {
			to[j] = tau*so[j] + (1-tau)*to[j]
		}
	}

	softUpdate(got, want, tau)

	gp, wp := want.Params(), dst
	if len(gp) != len(wp) {
		t.Fatalf("%d parameters, want %d", len(gp), len(wp))
	}
	for i := range gp {
		g, w := gp[i].Storage().F64(), wp[i].Storage().F64()
		if len(g) != len(w) {
			t.Fatalf("param %d: %d elements, want %d", i, len(g), len(w))
		}
		for j := range g {
			if math.Float64bits(g[j]) != math.Float64bits(w[j]) {
				t.Fatalf("param %d element %d: got %v (%016x), want %v (%016x)",
					i, j, g[j], math.Float64bits(g[j]), w[j], math.Float64bits(w[j]))
			}
		}
	}
}

// TestSoftUpdateChunkOrderIndependent runs the same update many times over freshly seeded
// networks and requires every run to agree bit-for-bit. parallel.Rows does not guarantee
// which worker takes which chunk, so a genuine cross-chunk dependency would show up as
// run-to-run variation rather than as a wrong answer on any single run — the failure mode
// the previous test cannot see.
func TestSoftUpdateChunkOrderIndependent(t *testing.T) {
	run := func() []float64 {
		online := nn.NewSequential(nn.NewLinear(tensor.F64, 256, 256, 3))
		target := nn.NewSequential(nn.NewLinear(tensor.F64, 256, 256, 4))
		// Perturb both so the blend is not dominated by structure in the initializer.
		rng := rand.New(rand.NewPCG(21, 22))
		for _, p := range append(online.Params(), target.Params()...) {
			s := p.Storage().F64()
			for i := range s {
				s[i] += rng.NormFloat64()
			}
		}
		softUpdate(online, target, 0.005)
		var out []float64
		for _, p := range target.Params() {
			out = append(out, p.Storage().F64()...)
		}
		return out
	}
	first := run()
	for r := 1; r < 8; r++ {
		got := run()
		if len(got) != len(first) {
			t.Fatalf("run %d: %d elements, want %d", r, len(got), len(first))
		}
		for i := range first {
			if math.Float64bits(got[i]) != math.Float64bits(first[i]) {
				t.Fatalf("run %d element %d: %016x != %016x — chunk assignment changed a result",
					r, i, math.Float64bits(got[i]), math.Float64bits(first[i]))
			}
		}
	}
}
