package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestCPUMHABitIdenticalToRef guards the PRODUCTION attention path. cpu overrides
// OpMHA, so cpu/mha is what every attention call executes while ref/mha is a
// fallback most callers never reach — and the ULP audit found the production side
// blind (R-01KYM6C5AWFKZ): one-ulp mutations of the QK product, the backward
// accumulation and the ALiBi bias all passed the backend/cpu suite, while the same
// probe on ref/mha turns its collapse tests red. Verification had landed on the
// wrong side of that split.
//
// This closes it by parity rather than by duplicating the kernel: ref/mha is already
// guarded, and the two backends were MEASURED to agree bit-for-bit before this test
// was written — 0 of 48 elements differing, causal and non-causal. That measurement
// mattered: the analogous collapse of flashattn onto MHA looked equally plausible
// and failed in 27 of 48 elements (R-01KYM5J5Z8EK9), so bitwise agreement is never
// assumed from algebraic equivalence.
//
// SCOPE, established by mutation in the order PROC-012 requires — confirm the line
// executes, then confirm its output is used, then read the result:
//   - QK inner product (mha.go:351): a one-ulp change turns this red. Guarded, in
//     both dtypes; the F32 branch is separate code and these cases cover it.
//   - ALiBi bias in the F32 branch (mha.go:338): a SIGN FLIP turns this red. Two
//     other probes stay green and BOTH are accounted for. A one-ulp scale is
//     absorbed — the bias is small beside the score it joins, so the perturbation
//     falls below that score's ulp (PROC-010). A +1 distance shift adds slopes[h]·1,
//     a per-head CONSTANT, to every score in the row, and softmax is invariant to a
//     uniform shift, so the output is genuinely unchanged and no test at any
//     tolerance could catch it. Verified, not assumed: a panic probe shows 338
//     executes, a nonzero-slope probe shows ALiBi is active, and zeroing the
//     branch's row[j] turns this test red, so its output does reach the comparison.
//   - the backward pass is guarded separately, bit-exactly in F64 and by tolerance
//     in f32, in mha_backward_parity_test.go. mhaBwdGemmF32 is gated by
//     `f32NativeKernels && seq >= mhaGemmMinSeq` and is UNREACHED in the default
//     build, so it is untestable here — unreached, not unguarded (R-01KYM78A9BEBC
//     corrects an earlier claim that conflated the two).
func TestCPUMHABitIdenticalToRef(t *testing.T) {
	rb, ok := backend.Get(backend.Ref)
	if !ok {
		t.Skip("ref backend unavailable")
	}
	cb, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend unavailable")
	}
	cases := []struct{ seq, heads, dk int }{
		{1, 1, 1}, {4, 1, 3}, {6, 2, 4}, {7, 3, 2}, {16, 4, 8},
		// dk=6 and seq=9 are the MIXED case the others miss: the jammed dot runs its 4-wide
		// body once and then a 2-element remainder, and the key loop runs two jammed passes
		// and then a remainder of one. Every other row is divisible one way or the other.
		{9, 2, 6},
	}
	for _, c := range cases {
		for _, f32 := range []bool{false, true} {
			for _, mode := range []struct {
				causal, alibi bool
				window        int
			}{
				{false, false, 0},
				{true, false, 0},
				{false, true, 0}, // ALiBi bias path — line 338 of mha.go is otherwise never run
				{true, true, 0},
				{true, false, 3}, // sliding window
			} {
				causal := mode.causal
				// Both dtypes: the F32 forward branch is a SEPARATE code path in cpu/mha
				// (guarded by `if isF32`), so an F64-only test leaves it entirely unrun.
				// cpu and ref were measured to agree bit-for-bit in F32 as well — 0 of 48
				// elements differing, with and without ALiBi — before relying on it here.
				shape := tensor.Shape{c.seq, c.heads * c.dk}
				seed := uint64(c.seq*10 + c.dk)
				var q, k, v *tensor.Tensor
				if f32 {
					q, k, v = bench.RandF32(shape, seed), bench.RandF32(shape, seed+1), bench.RandF32(shape, seed+2)
				} else {
					q, k, v = bench.RandF64(shape, seed), bench.RandF64(shape, seed+1), bench.RandF64(shape, seed+2)
				}
				at := backend.AttnAttrs{Heads: c.heads, Causal: causal, ALiBi: mode.alibi, Window: mode.window}
				in := []*tensor.Tensor{q, k, v}
				want, err := backend.Execute(backend.NewContext().WithBackend(rb), backend.OpMHA, in, at)
				if err != nil {
					t.Fatal(err)
				}
				got, err := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpMHA, in, at)
				if err != nil {
					t.Fatal(err)
				}
				if got[0].Numel() != want[0].Numel() {
					t.Fatalf("seq=%d heads=%d dk=%d mode=%+v: numel %d != %d",
						c.seq, c.heads, c.dk, mode, got[0].Numel(), want[0].Numel())
				}
				// Bit-exact for F64 and on the default build; the f32-native SIMD
				// lane is compared within its own ADR budget (see f32NativeTolerant).
				tolerant := f32NativeTolerant(f32)
				var maxRel float64
				for i := range want[0].Numel() {
					co := tensor.Unravel(i, want[0].Shape())
					g, w := got[0].AtF64(co...), want[0].AtF64(co...)
					if tolerant {
						if !parityCloseF32(g, w) {
							t.Fatalf("seq=%d heads=%d dk=%d f32=%v mode=%+v elem %d: cpu %v vs ref %v exceeds the f32-native budget",
								c.seq, c.heads, c.dk, f32, mode, i, g, w)
						}
						if r := parityRelErr(g, w); r > maxRel {
							maxRel = r
						}
						continue
					}
					if math.Float64bits(g) != math.Float64bits(w) {
						t.Fatalf("seq=%d heads=%d dk=%d f32=%v mode=%+v elem %d: cpu %v != ref %v",
							c.seq, c.heads, c.dk, f32, mode, i, g, w)
					}
				}
				if tolerant {
					t.Logf("seq=%d heads=%d dk=%d mode=%+v: f32-native max rel err %.2e",
						c.seq, c.heads, c.dk, mode, maxRel)
				}
			}
		}
	}
}
