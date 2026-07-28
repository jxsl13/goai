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
// SCOPE, verified by mutation rather than claimed: this guards the F64 FORWARD path.
// A one-ulp change to the QK inner product (mha.go:351) turns it red. It does NOT
// yet reach two paths that mutation shows are still blind —
//   - the F32 forward branch (mha.go:338, inside `if isF32`), because these cases
//     use F64 inputs. Extending there needs cpu-vs-ref F32 agreement to be MEASURED
//     first: ref accumulates in float64 per the ACCUM invariant while the cpu F32
//     branch may not, so the two need not agree bitwise.
//   - the backward pass (mhaBwdGemmBand, mha.go:596), which no case here invokes.
//
// Both are recorded in R-01KYM6C5AWFKZ as remaining work.
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
	}
	for _, c := range cases {
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
			q := bench.RandF64(tensor.Shape{c.seq, c.heads * c.dk}, uint64(c.seq*10+c.dk))
			k := bench.RandF64(tensor.Shape{c.seq, c.heads * c.dk}, uint64(c.seq*10+c.dk)+1)
			v := bench.RandF64(tensor.Shape{c.seq, c.heads * c.dk}, uint64(c.seq*10+c.dk)+2)
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
			for i := range want[0].Numel() {
				co := tensor.Unravel(i, want[0].Shape())
				g, w := got[0].AtF64(co...), want[0].AtF64(co...)
				if math.Float64bits(g) != math.Float64bits(w) {
					t.Fatalf("seq=%d heads=%d dk=%d mode=%+v elem %d: cpu %v != ref %v",
						c.seq, c.heads, c.dk, mode, i, g, w)
				}
			}
		}
	}
}
