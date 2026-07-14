package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The OpWKV VJP F32 fast path (§T516 devirtualization) is NOT exercised by
// TestWKVGradCheck (F64 only) nor the nn RWKV suite (also F64). This test
// closes that gap: the F32 path computes internally in float64 and only rounds
// stored results to float32, so it must equal float32(the F64 path) within f32
// rounding. Reuses moeF64/moeCastF32/moeAssertClose from the MoE F32 test.
//
// Contract (read from vjp_rwkv.go): in = {k[seq,d], v[seq,d], w[d], u[d]},
// g = upstream [seq,d]; outputs = {dk[seq,d], dv[seq,d], dw[d], du[d]}. The
// decay w is used as `-(t-1-i)*w` so it must be a positive decay.
func TestWKVVJPF32Parity(t *testing.T) {
	vjp := vjps[backend.OpWKV]
	const seq, d = 6, 4

	k := moeF64(tensor.Shape{seq, d}, 1, false)
	v := moeF64(tensor.Shape{seq, d}, 2, false)
	u := moeF64(tensor.Shape{d}, 4, false)
	// w: moderate positive decays in [0.3, 1.1] to keep the exp/decay window
	// well-conditioned (mirrors the gradcheck's w setup).
	w := tensor.New(tensor.F64, tensor.Shape{d})
	for c, s := 0, w.Storage().F64(); c < d; c++ {
		s[c] = 0.3 + 0.2*float64(c)
	}
	g := moeF64(tensor.Shape{seq, d}, 7, false)

	in64 := []*tensor.Tensor{k, v, w, u}
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

	names := []string{"wkv.dk", "wkv.dv", "wkv.dw", "wkv.du"}
	for i := range o64 {
		moeAssertClose(t, names[i], o32[i], o64[i])
	}
}
