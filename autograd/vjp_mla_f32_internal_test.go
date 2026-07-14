package autograd

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMLAF32Parity guards the OpMLA VJP's F32 fast path. TestMLAOpGradCheck (nn)
// and TestGradCheckAllOps run in F64 only, so the F32 slice-indexed path would be
// unverified. The fast path computes internally in float64 and rounds stored
// results to float32 per accumulation, so it must equal float32(the F64 path).
//
// OpMLA contract (see vjp_mla.go): in = {qC[seq,hdh], kC[seq,hdh], vC[seq,hdh],
// qRpre[seq,heads*dR], kRpre[seq,dR]}, attrs = MLAAttrs, upstream grad g[seq,hdh].
// It returns grads {dqC, dkC, dvC, dqRpre, dkRpre} (none nil here).
func TestMLAF32Parity(t *testing.T) {
	vjp := vjps[backend.OpMLA]
	for _, causal := range []bool{false, true} {
		t.Run(fmt.Sprintf("causal=%v", causal), func(t *testing.T) {
			seq, heads, dh, dR := 5, 2, 4, 4
			hdh := heads * dh
			qC := moeF64(tensor.Shape{seq, hdh}, 1, false)
			kC := moeF64(tensor.Shape{seq, hdh}, 2, false)
			vC := moeF64(tensor.Shape{seq, hdh}, 3, false)
			qRpre := moeF64(tensor.Shape{seq, heads * dR}, 4, false)
			kRpre := moeF64(tensor.Shape{seq, dR}, 5, false)
			g := moeF64(tensor.Shape{seq, hdh}, 6, false)
			attrs := backend.MLAAttrs{Heads: heads, Causal: causal, RoPEBase: 10000}

			in64 := []*tensor.Tensor{qC, kC, vC, qRpre, kRpre}
			o64, err := vjp(nil, in64, nil, attrs, g)
			if err != nil {
				t.Fatal(err)
			}
			in32 := make([]*tensor.Tensor, len(in64))
			for i := range in64 {
				in32[i] = moeCastF32(in64[i])
			}
			o32, err := vjp(nil, in32, nil, attrs, moeCastF32(g))
			if err != nil {
				t.Fatal(err)
			}
			names := []string{"dqC", "dkC", "dvC", "dqRpre", "dkRpre"}
			for k := range o64 {
				moeAssertClose(t, names[k], o32[k], o64[k])
			}
		})
	}
}
