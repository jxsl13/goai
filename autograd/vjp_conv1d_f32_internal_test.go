package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The Conv1D/Softplus VJP F32 fast paths (§T) are NOT exercised by TestGradCheckAllOps
// (it runs in F64). These tests close that gap: each F32 fast path computes internally
// in float64 and only rounds the stored result to float32, so it must equal
// float32(the F64 path) within f32 rounding. Reuses moeF64/moeCastF32/moeAssertClose
// from vjp_moe_f32_internal_test.go (same package).

func TestSoftplusF32Parity(t *testing.T) {
	vjp := vjps[backend.OpSoftplus]
	x := moeF64(tensor.Shape{4, 5}, 1, false)
	g := moeF64(tensor.Shape{4, 5}, 2, false)

	o64, err := vjp(nil, []*tensor.Tensor{x}, nil, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(x)}, nil, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "softplus.dx", o32[0], o64[0])
}

func TestConv1DF32Parity(t *testing.T) {
	vjp := vjps[backend.OpConv1D]
	L, D, K := 7, 6, 3
	x := moeF64(tensor.Shape{L, D}, 10, false)
	w := moeF64(tensor.Shape{D, K}, 11, false)
	b := moeF64(tensor.Shape{D}, 12, false)
	g := moeF64(tensor.Shape{L, D}, 13, false) // out grad, shape == out == [L,D]

	in64 := []*tensor.Tensor{x, w, b}
	o64, err := vjp(nil, in64, nil, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	in32 := []*tensor.Tensor{moeCastF32(x), moeCastF32(w), moeCastF32(b)}
	o32, err := vjp(nil, in32, nil, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"conv1d.dx", "conv1d.dw", "conv1d.db"}
	for k := range o64 {
		moeAssertClose(t, names[k], o32[k], o64[k])
	}
}
