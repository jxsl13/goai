package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type mtaNopRecorder struct{}

func (mtaNopRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

// TestMTAHeadConvFusedBitExactVsDispatch locks the fused headConv (Recorder == nil) byte-for-
// byte to the per-(g,o,p) OpSlice/OpMul/OpAdd dispatch (forced by a non-nil no-op recorder).
// Same products and same left-to-right accumulation order → bit-identical. Covers F32 and F64,
// several CH, and a non-identity HeadKernel.
func TestMTAHeadConvFusedBitExactVsDispatch(t *testing.T) {
	for _, d := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, cfg := range []struct{ T, dim, heads, cq, ck, ch int }{
			{16, 128, 8, 3, 5, 4}, {32, 256, 16, 6, 11, 8}, {12, 192, 12, 2, 3, 3},
			// The head mix splits its element range across workers once the work clears
			// parallelRows' threshold, and it walks one INPUT map at a time across all ch
			// outputs. The three configs above are at or below that threshold or leave a single
			// band, so they gate the arithmetic but not the split. These two do both: several
			// bands, and one group (heads == ch) as well as two.
			{64, 256, 16, 4, 7, 16}, {64, 256, 32, 4, 7, 16},
		} {
			m, err := nn.NewMultiTokenAttention(d, cfg.dim, cfg.heads, 3,
				nn.WithMTAKeyQueryKernel(cfg.cq, cfg.ck), nn.WithMTAHeadKernel(cfg.ch))
			if err != nil {
				t.Fatal(err)
			}
			// randomize HeadKernel so it's not the identity (exercise the full mix)
			hk := m.HeadKernel.Storage()
			for i := 0; i < m.HeadKernel.Numel(); i++ {
				v := math.Sin(float64(i)*0.7 + 1)
				co := tensor.Unravel(i, m.HeadKernel.Shape())
				m.HeadKernel.SetF64(v, co...)
			}
			_ = hk
			x := tensor.New(d, tensor.Shape{cfg.T, cfg.dim})
			for i := 0; i < x.Numel(); i++ {
				co := tensor.Unravel(i, x.Shape())
				x.SetF64(math.Sin(float64(i)*0.01), co...)
			}
			gf, err := m.Forward(backend.NewContext(), x) // fused
			if err != nil {
				t.Fatalf("fused d=%v %+v: %v", d, cfg, err)
			}
			ctx := backend.NewContext()
			ctx.Recorder = mtaNopRecorder{}
			gd, err := m.Forward(ctx, x) // dispatch
			if err != nil {
				t.Fatalf("dispatch d=%v %+v: %v", d, cfg, err)
			}
			if d == tensor.F64 {
				fs, ds := gf.Storage().F64(), gd.Storage().F64()
				for i := range fs {
					if math.Float64bits(fs[i]) != math.Float64bits(ds[i]) {
						t.Fatalf("d=%v cfg=%+v idx=%d fused=%v dispatch=%v", d, cfg, i, fs[i], ds[i])
					}
				}
			} else {
				fs, ds := gf.Storage().F32(), gd.Storage().F32()
				for i := range fs {
					if math.Float32bits(fs[i]) != math.Float32bits(ds[i]) {
						t.Fatalf("d=%v cfg=%+v idx=%d fused=%v dispatch=%v", d, cfg, i, fs[i], ds[i])
					}
				}
			}
		}
	}
}
