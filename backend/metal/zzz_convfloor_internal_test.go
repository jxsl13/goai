//go:build darwin && cgo

package metal

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// §T417 (§V22, the conv analog of the §T393 attention floor): the metal conv2d (im2col→MPS-GEMM→
// colout in ONE command buffer, §T341) measures ~0.84ms at N8/C16/HW32/F32/K3 while torch-mps is
// ~2× faster on conv overall. Before considering an MPSCNNConvolution/MPSGraph rewrite, measure the
// floor: the conv's GEMM alone ([F,C·K·K]·[C·K·K,N·ho·wo]) through the recorder with the SAME single
// dispatch round-trip. conv-total minus GEMM-only = what im2col+colout+host-transfers cost — the
// honest ceiling any fused/native-conv rewrite could recover.
func TestConvMatmulFloor(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const n, c, hw, f, k = 8, 16, 32, 32, 3
	const ho, wo = hw, hw // stride 1, pad 1
	M, K, N := f, c*k*k, n*ho*wo

	// conv total (the real op, via the backend).
	be, _ := backend.Get(backend.Metal)
	x := bench.RandF32(tensor.Shape{n, c, hw, hw}, 1)
	w := bench.RandF32(tensor.Shape{f, c, k, k}, 2)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{x, w}
	attrs := backend.ConvAttrs{Stride: 1, Pad: 1}
	convMs := func() float64 {
		if _, err := backend.Execute(ctx, backend.OpConv2D, ins, attrs); err != nil {
			t.Fatal(err)
		}
		const it = 30
		s := time.Now()
		for range it {
			if _, err := backend.Execute(ctx, backend.OpConv2D, ins, attrs); err != nil {
				t.Fatal(err)
			}
		}
		return float64(time.Since(s).Microseconds()) / it / 1000
	}()

	// GEMM-only floor: the same [M,K]·[K,N] through the recorder, one command buffer per iteration
	// (same single dispatch round-trip as the conv), device-resident operands (no per-iter upload —
	// isolates pure GEMM+dispatch; the conv also pays X-upload + Out-download + im2col + colout).
	a, _ := NewDeviceBufferF32(make([]float32, M*K))
	bm, _ := NewDeviceBufferF32(make([]float32, K*N))
	o, _ := NewDeviceBufferF32(make([]float32, M*N))
	defer a.Release()
	defer bm.Release()
	defer o.Release()
	gemmMs := func() float64 {
		run := func() {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			if err := r.MatMul(a, bm, o, M, K, N); err != nil {
				t.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			r.Free()
		}
		run()
		const it = 30
		s := time.Now()
		for range it {
			run()
		}
		return float64(time.Since(s).Microseconds()) / it / 1000
	}()

	gflop := 2 * float64(M) * float64(K) * float64(N) / 1e9
	t.Logf("conv2d N%d C%d HW%d F%d K%d: conv total %.3f ms (%.0f GFLOP/s) | GEMM-only floor %.3f ms (%.0f GFLOP/s) | overhead %.3f ms (%.0f%%)",
		n, c, hw, f, k, convMs, gflop/(convMs/1000), gemmMs, gflop/(gemmMs/1000), convMs-gemmMs, 100*(convMs-gemmMs)/convMs)
}
