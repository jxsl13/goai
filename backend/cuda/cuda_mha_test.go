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

// headSlice extracts head h's contiguous [seq,hd] block from a [seq,heads*hd].
func headSlice(x *tensor.Tensor, h, hd int) *tensor.Tensor {
	seq := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{seq, hd})
	for i := 0; i < seq; i++ {
		for d := 0; d < hd; d++ {
			out.SetF64(x.AtF64(i, h*hd+d), i, d)
		}
	}
	return out
}

// Batched multi-head attention must equal a per-head reference (each head's
// single-head attention computed on the Pure-Go backend), for both causal and
// full attention. Validates the strided-batched Sgemm layout + per-head mask.
func TestCUDAMultiHeadAttentionMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	rc := backend.NewContext().WithBackend(ref)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(rc, op, ins, a)
		must(t, e)
		return o[0]
	}
	for _, tc := range []struct {
		seq, heads, hd int
		causal         bool
	}{
		{8, 4, 16, true}, {8, 4, 16, false}, {6, 3, 8, true}, {5, 1, 32, true},
	} {
		w := tc.heads * tc.hd
		q := bench.RandF32(tensor.Shape{tc.seq, w}, 1)
		k := bench.RandF32(tensor.Shape{tc.seq, w}, 2)
		v := bench.RandF32(tensor.Shape{tc.seq, w}, 3)
		scale := 1 / math.Sqrt(float64(tc.hd))

		// device batched MHA
		dq, _ := cuda.UploadF32(q)
		dk, _ := cuda.UploadF32(k)
		dv, _ := cuda.UploadF32(v)
		dout, err := cuda.MultiHeadAttention(dq, dk, dv, tc.heads, tc.causal)
		must(t, err)
		got, _ := dout.ToHost()
		dq.Free()
		dk.Free()
		dv.Free()
		dout.Free()

		// per-head reference
		want := tensor.New(tensor.F32, tensor.Shape{tc.seq, w})
		for h := 0; h < tc.heads; h++ {
			qh, kh, vh := headSlice(q, h, tc.hd), headSlice(k, h, tc.hd), headSlice(v, h, tc.hd)
			kt, _ := kh.Transpose(0, 1)
			sc := ex(backend.OpMatMul, nil, qh, kt) // [seq,seq]
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
			oh := ex(backend.OpMatMul, nil, ex(backend.OpSoftmax, nil, masked), vh) // [seq,hd]
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
				t.Fatalf("%v [%d]: mha %v vs ref %v", tc, i, g, wv)
			}
		}
	}
}
