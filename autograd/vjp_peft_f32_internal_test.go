package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// F32 parity for the PEFT VJP fast paths (§T642). gradcheck runs F64-only, so the
// F32 branches of OpDoRAWeight/OpIA3 need their own coverage. Each F32 branch
// computes in float64 and rounds the store, so it must equal float32(the F64 path).
// Helpers moeF64/moeCastF32/moeAssertClose live in vjp_moe_f32_internal_test.go.

func TestDoRAF32Parity(t *testing.T) {
	vjp := vjps[backend.OpDoRAWeight]
	rows, cols := 8, 4
	v := moeF64(tensor.Shape{rows, cols}, 1, true) // positive → nonzero column norms
	m := moeF64(tensor.Shape{cols}, 2, false)
	g := moeF64(tensor.Shape{rows, cols}, 3, false)
	o64, err := vjp(nil, []*tensor.Tensor{v, m}, nil, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(v), moeCastF32(m)}, nil, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "dora.dv", o32[0], o64[0])
	moeAssertClose(t, "dora.dm", o32[1], o64[1])
}

func TestIA3F32Parity(t *testing.T) {
	vjp := vjps[backend.OpIA3]
	rows, d := 6, 4
	x := moeF64(tensor.Shape{rows, d}, 4, false)
	l := moeF64(tensor.Shape{d}, 5, false)
	g := moeF64(tensor.Shape{rows, d}, 6, false)
	o64, err := vjp(nil, []*tensor.Tensor{x, l}, nil, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(x), moeCastF32(l)}, nil, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "ia3.dx", o32[0], o64[0])
	moeAssertClose(t, "ia3.dl", o32[1], o64[1])
}
