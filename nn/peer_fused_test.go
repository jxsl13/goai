package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// peerNopRecorder is a non-nil backend.Recorder that records nothing. Attaching it
// to a Context forces PEER.Forward down the one-hot dispatch (training) path while
// still executing eagerly, so the fused inference gather can be compared bit-for-bit
// against it.
type peerNopRecorder struct{}

func (peerNopRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

// TestPEERFusedGatherBitExact asserts the fused gate-score gather (ctx.Recorder==nil)
// is bit-identical to the one-hot OpMul+OpSum path (ctx.Recorder != nil) for y, the
// retrieval indices, and the gate softmax, across several product-key configs.
func TestPEERFusedGatherBitExact(t *testing.T) {
	cases := []struct{ tks, d, n, topK, heads int }{
		{5, 8, 8, 3, 1},
		{7, 16, 12, 4, 2},
		{4, 6, 32, 2, 3},
		{9, 12, 10, 5, 1},
	}
	for _, c := range cases {
		p := nn.NewPEER(tensor.F64, c.d, c.n, c.topK, 1234, nn.WithPEERHeads(c.heads))
		x := tensor.New(tensor.F64, tensor.Shape{c.tks, c.d})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = float64((i*2654435761)&0xffff)/32768.0 - 1.0
		}

		yF, idxF, gF, err := p.Forward(backend.NewContext(), x) // fused (nil recorder)
		if err != nil {
			t.Fatalf("fused forward: %v", err)
		}
		rec := backend.NewContext().WithRecorder(peerNopRecorder{})
		yD, idxD, gD, err := p.Forward(rec, x) // dispatch (non-nil recorder)
		if err != nil {
			t.Fatalf("dispatch forward: %v", err)
		}

		if !yF.Shape().Equal(yD.Shape()) || !gF.Shape().Equal(gD.Shape()) {
			t.Fatalf("cfg %+v: shape mismatch y %v/%v g %v/%v", c, yF.Shape(), yD.Shape(), gF.Shape(), gD.Shape())
		}
		yfs, yds := yF.Storage().F64(), yD.Storage().F64()
		for i := range yfs {
			if yfs[i] != yds[i] {
				t.Fatalf("cfg %+v: y[%d] fused=%v dispatch=%v (not bit-exact)", c, i, yfs[i], yds[i])
			}
		}
		gfs, gds := gF.Storage().F64(), gD.Storage().F64()
		for i := range gfs {
			if gfs[i] != gds[i] {
				t.Fatalf("cfg %+v: gates[%d] fused=%v dispatch=%v (not bit-exact)", c, i, gfs[i], gds[i])
			}
		}
		for t2 := range idxF {
			if len(idxF[t2]) != len(idxD[t2]) {
				t.Fatalf("cfg %+v: idx len mismatch row %d", c, t2)
			}
			for j := range idxF[t2] {
				if idxF[t2][j] != idxD[t2][j] {
					t.Fatalf("cfg %+v: idx[%d][%d] fused=%d dispatch=%d", c, t2, j, idxF[t2][j], idxD[t2][j])
				}
			}
		}
	}
}
