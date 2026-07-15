//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// The f16 KV cache stores rounded keys/values, so flash-over-f16 must match
// flash-over-f32 within the f16 rounding budget (~2^-11 relative per element,
// partially cancelling over the hd-length dots) — NOT bit-exactly. The bar is
// 5e-3 on unit-scale inputs; the observed diff is logged so drift is visible.
// Rows are appended one by one through the f16 AppendDpos (round-to-nearest),
// exercising the whole cache path, not just the kernel.
func TestCUDAF16KVFlashParity(t *testing.T) {
	skipNoGPU(t)
	cases := []struct {
		qHeads, kvHeads, hd, seqKV, pos int
	}{
		{8, 2, 8, 10, 9},
		{8, 2, 8, 10, 0},
		{4, 4, 64, 33, 31},
		{32, 4, 64, 128, 127},   // TinyLlama-class
		{32, 4, 128, 512, 300},  // Mistral-class
		{8, 1, 128, 2048, 2047}, // MQA, full fixed-cache depth
	}
	pos, err := cuda.NewDevicePos()
	must(t, err)
	defer pos.Free()
	for _, c := range cases {
		t.Run(fmt.Sprintf("q%d_kv%d_hd%d_ctx%d_p%d", c.qHeads, c.kvHeads, c.hd, c.seqKV, c.pos), func(t *testing.T) {
			wq, wkv := c.qHeads*c.hd, c.kvHeads*c.hd
			q := bench.RandF32(tensor.Shape{1, wq}, 12)
			k := bench.RandF32(tensor.Shape{c.seqKV, wkv}, 13)
			v := bench.RandF32(tensor.Shape{c.seqKV, wkv}, 14)
			dq, _ := cuda.UploadF32(q)
			defer dq.Free()
			dk, _ := cuda.UploadF32(k)
			defer dk.Free()
			dv, _ := cuda.UploadF32(v)
			defer dv.Free()

			// f32 flash reference over the full uploaded cache.
			must(t, pos.Set(c.pos))
			ref, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 15))
			defer ref.Free()
			must(t, cuda.GroupedQueryAttentionKVDposFlashInto(dq, dk, dv, c.qHeads, c.kvHeads, pos, ref))

			// f16 cache filled row-by-row through the graph-capturable append.
			cf, err := cuda.NewKVCacheF16(c.seqKV, wkv)
			must(t, err)
			defer cf.Free()
			must(t, cf.ZeroCache())
			for i := 0; i <= c.pos; i++ {
				kr, _ := cuda.UploadF32(rowOf(k, i))
				vr, _ := cuda.UploadF32(rowOf(v, i))
				must(t, pos.Set(i))
				must(t, cf.AppendDpos(kr, vr, pos))
				kr.Free()
				vr.Free()
			}
			must(t, pos.Set(c.pos))
			out, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 16))
			defer out.Free()
			must(t, cuda.GroupedQueryAttentionKVF16DposFlashInto(dq, cf, c.qHeads, c.kvHeads, pos, out))

			hr, _ := ref.ToHost()
			hf, _ := out.ToHost()
			m := maxAbsDiff(hr, hf)
			t.Logf("f16-KV vs f32-KV flash maxAbs %.3e", m)
			if m > 5e-3 {
				t.Fatalf("f16 KV flash too far from f32: maxAbs %.3e", m)
			}
		})
	}
}

// rowOf extracts row i of a [rows, cols] tensor as a fresh [1, cols] tensor.
func rowOf(x *tensor.Tensor, i int) *tensor.Tensor {
	cols := x.Shape()[1]
	d := make([]float32, cols)
	for j := 0; j < cols; j++ {
		d[j] = float32(x.AtF64(i, j))
	}
	return tensor.FromFloat32(tensor.Shape{1, cols}, d)
}
