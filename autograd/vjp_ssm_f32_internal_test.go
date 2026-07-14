package autograd

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The OpSSM VJP F32 fast path (§T-SSM devirtualisation) is NOT exercised by
// TestGradCheckAllOps (F64 only) nor the nn Mamba suite (also F64). This test
// closes that gap: the F32 fast path computes the scan internally in float64 and
// only rounds stored results to float32, so each F32 grad must equal
// float32(the F64 grad) within f32 rounding tolerance.
//
// Inputs (Gu & Dao 2023): u,delta [L,D]; A [D,N]; B,C [L,N]; optional D_skip [D];
// upstream g matches the output y[L,D]. delta must be positive for a stable scan.
func TestSSMF32Parity(t *testing.T) {
	vjp := vjps[backend.OpSSM]
	L, D, N := 6, 4, 3

	for _, hasSkip := range []bool{false, true} {
		u := moeF64(tensor.Shape{L, D}, 1, false)
		delta := moeF64(tensor.Shape{L, D}, 2, true) // positive → Ā = exp(Δ·A) well-conditioned
		A := moeF64(tensor.Shape{D, N}, 3, false)
		B := moeF64(tensor.Shape{L, N}, 4, false)
		C := moeF64(tensor.Shape{L, N}, 5, false)
		g := moeF64(tensor.Shape{L, D}, 6, false)
		in64 := []*tensor.Tensor{u, delta, A, B, C}
		if hasSkip {
			in64 = append(in64, moeF64(tensor.Shape{D}, 7, false))
		}

		o64, err := vjp(nil, in64, nil, nil, g)
		if err != nil {
			t.Fatal(err)
		}

		in32 := make([]*tensor.Tensor, len(in64))
		for i := range in64 {
			in32[i] = moeCastF32(in64[i])
		}
		o32, err := vjp(nil, in32, nil, nil, moeCastF32(g))
		if err != nil {
			t.Fatal(err)
		}

		names := []string{"du", "ddelta", "dA", "dB", "dC", "dDskip"}
		if len(o32) != len(o64) {
			t.Fatalf("hasSkip=%v: grad count %d vs %d", hasSkip, len(o32), len(o64))
		}
		for k := range o64 {
			moeAssertClose(t, fmt.Sprintf("ssm.%s(skip=%v)", names[k], hasSkip), o32[k], o64[k])
		}
	}
}
