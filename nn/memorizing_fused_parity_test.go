package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type memNopRecorder struct{}

func (memNopRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

// TestMemForwardFusedBitExactVsDispatch locks the fused F64 inference einsums
// (memScoresFusedF64 / memOutFusedF64, gated on Recorder == nil) byte-for-byte to the
// OpEinsum dispatch path (forced by a non-nil no-op recorder). Both sum the contracted
// axis ascending with identical products, matching the einsum engine order.
func TestMemForwardFusedBitExactVsDispatch(t *testing.T) {
	for _, cfg := range []struct {
		T, dim, heads, memN, topK int
	}{
		{16, 128, 4, 256, 16}, {32, 256, 8, 512, 32}, {7, 96, 3, 100, 20}, {64, 512, 8, 2048, 32},
	} {
		m, err := nn.NewMemorizingAttention(tensor.F64, cfg.dim, cfg.heads, 9,
			nn.WithMemorizingMemorySize(cfg.memN), nn.WithMemorizingTopK(cfg.topK))
		if err != nil {
			t.Fatal(err)
		}
		mk := func(rows int, s float64) *tensor.Tensor {
			tn := tensor.New(tensor.F64, tensor.Shape{rows, cfg.dim})
			st := tn.Storage().F64()
			for i := range st {
				st[i] = math.Sin(float64(i)*0.002 + s)
			}
			return tn
		}
		if err := m.Memory.AddSegment(mk(cfg.memN, 0.3), mk(cfg.memN, 0.7)); err != nil {
			t.Fatal(err)
		}
		x := mk(cfg.T, 0.1)
		gf, err := m.Forward(backend.NewContext(), x) // fused (Recorder == nil)
		if err != nil {
			t.Fatalf("fused: %v", err)
		}
		ctx := backend.NewContext()
		ctx.Recorder = memNopRecorder{}
		gd, err := m.Forward(ctx, x) // dispatch
		if err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		fs, ds := gf.Storage().F64(), gd.Storage().F64()
		for i := range fs {
			if math.Float64bits(fs[i]) != math.Float64bits(ds[i]) {
				t.Fatalf("cfg=%+v idx=%d fused=%v dispatch=%v", cfg, i, fs[i], ds[i])
			}
		}
	}
}
