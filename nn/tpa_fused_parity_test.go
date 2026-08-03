package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// nopRecorder is a non-nil, no-op autograd Recorder: attaching it makes ctx.Recorder != nil
// so TPA.contract takes the einsum-DISPATCH path (fused path is gated on Recorder == nil),
// while ops still execute eagerly (Record tapes nothing).
type nopRecorder struct{}

func (nopRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

// TestTPAForwardFusedBitExactVsDispatch locks the fused inference contract (ctx.Recorder
// == nil) byte-for-byte to the einsum-dispatch path (forced by attaching a Recorder). The
// fused loop sums r ascending with identical products and ×(1/r), matching the einsum
// engine's ascending-r accumulation order, so it must be bit-identical.
func TestTPAForwardFusedBitExactVsDispatch(t *testing.T) {
	for _, cfg := range []struct {
		T, d, heads, dh int
		rope            bool
	}{
		{16, 64, 4, 16, false}, {32, 128, 8, 16, true}, {7, 96, 3, 32, false}, {64, 256, 8, 32, true},
	} {
		var opts []nn.TPAOption
		if cfg.rope {
			opts = append(opts, nn.WithTPARoPE(10000))
		}
		m, err := nn.NewTPA(tensor.F64, cfg.d, cfg.heads, cfg.dh, true, 7, opts...)
		if err != nil {
			t.Fatal(err)
		}
		x := tensor.New(tensor.F64, tensor.Shape{cfg.T, cfg.d})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = math.Sin(float64(i)*0.1) * 0.5
		}
		// fused: plain inference context (Recorder == nil)
		gf, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatalf("fused fwd: %v", err)
		}
		// dispatch: attach a recorder so contract takes the einsum path
		ctx := backend.NewContext()
		ctx.Recorder = nopRecorder{}
		gd, err := m.Forward(ctx, x)
		if err != nil {
			t.Fatalf("dispatch fwd: %v", err)
		}
		fs, ds := gf.Storage().F64(), gd.Storage().F64()
		for i := range fs {
			if math.Float64bits(fs[i]) != math.Float64bits(ds[i]) {
				t.Fatalf("cfg=%+v idx=%d fused=%v dispatch=%v", cfg, i, fs[i], ds[i])
			}
		}
	}
}
