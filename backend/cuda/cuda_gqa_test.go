//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Grouped-query attention (kvHeads < qHeads) must equal a per-head reference in
// which query head h attends to kv head h/(qHeads/kvHeads). Validates the
// pointer-array batched Sgemm and the kv-head sharing.
func TestCUDAGroupedQueryAttentionMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	rc := backend.NewContext().WithBackend(ref)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(rc, op, ins, a)
		must(t, e)
		return o[0]
	}
	for _, tc := range []struct {
		seq, qHeads, kvHeads, hd int
		causal                   bool
	}{
		{8, 8, 2, 16, true}, {8, 8, 2, 16, false}, {6, 6, 3, 8, true}, {5, 4, 4, 16, true},
	} {
		group := tc.qHeads / tc.kvHeads
		wq, wkv := tc.qHeads*tc.hd, tc.kvHeads*tc.hd
		q := bench.RandF32(tensor.Shape{tc.seq, wq}, 1)
		k := bench.RandF32(tensor.Shape{tc.seq, wkv}, 2)
		v := bench.RandF32(tensor.Shape{tc.seq, wkv}, 3)
		scale := 1 / math.Sqrt(float64(tc.hd))

		dq, _ := cuda.UploadF32(q)
		dk, _ := cuda.UploadF32(k)
		dv, _ := cuda.UploadF32(v)
		dout, err := cuda.GroupedQueryAttention(dq, dk, dv, tc.qHeads, tc.kvHeads, tc.causal)
		must(t, err)
		got, _ := dout.ToHost()
		dq.Free()
		dk.Free()
		dv.Free()
		dout.Free()

		want := tensor.New(tensor.F32, tensor.Shape{tc.seq, wq})
		for h := 0; h < tc.qHeads; h++ {
			kv := h / group
			qh, kh, vh := headSlice(q, h, tc.hd), headSlice(k, kv, tc.hd), headSlice(v, kv, tc.hd)
			kt, _ := kh.Transpose(0, 1)
			sc := ex(backend.OpMatMul, nil, qh, kt)
			masked := tensor.New(tensor.F32, tensor.Shape{tc.seq, tc.seq})
			for i := 0; i < tc.seq; i++ {
				for j := 0; j < tc.seq; j++ {
					if tc.causal && j > i {
						masked.SetF64(math.Inf(-1), i, j)
					} else {
						masked.SetF64(sc.AtF64(i, j)*scale, i, j)
					}
				}
			}
			oh := ex(backend.OpMatMul, nil, ex(backend.OpSoftmax, nil, masked), vh)
			for i := 0; i < tc.seq; i++ {
				for d := 0; d < tc.hd; d++ {
					want.SetF64(oh.AtF64(i, d), i, h*tc.hd+d)
				}
			}
		}
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, wv := got.AtF64(idx...), want.AtF64(idx...)
			if math.Abs(g-wv) > 5e-3*math.Max(1, math.Abs(wv)) {
				t.Fatalf("%v [%d]: gqa %v vs ref %v", tc, i, g, wv)
			}
		}
	}
}
