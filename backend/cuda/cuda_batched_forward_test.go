//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// §ROADMAP FRONT B / B4 — the TRUE aggregate serving throughput: a full batched decode forward
// (embed skipped; per layer: rmsnorm + QKV/O/gate/up/down GEMMs at M=batch + BatchedDecodeAttn
// over each sequence's paged KV) at TinyLlama dims, measured at batch=1/16/64. This is the
// number vs vLLM's ~4655 aggregate (n=64). Unlike B3 (attention only), the FFN/QKV GEMMs at
// batch dominate here — the honest e2e serving metric. Synthetic f16 weights; RoPE uses a
// uniform position (cost-identical; per-sequence positions are a correctness detail for B4b).
func benchBatchedForward(b *testing.B, batch, seqLen, layers int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden = 2048, 32, 4, 64, 5632
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(9))
	mkW := func(k, n int) *cuda.ResidentBF16 {
		t := tensor.New(tensor.F32, tensor.Shape{k, n})
		f := t.Storage().F32()
		for i := range f {
			f[i] = float32(rng.NormFloat64()) * 0.02
		}
		r, err := cuda.NewResidentBF16(t)
		if err != nil {
			b.Fatal(err)
		}
		return r
	}
	mkVec := func(n int) *cuda.ResidentVec {
		t := tensor.New(tensor.F32, tensor.Shape{n})
		f := t.Storage().F32()
		for i := range f {
			f[i] = 1.0
		}
		r, _ := cuda.NewResidentVec(t)
		return r
	}
	type layer struct {
		gAttn, gFFN    *cuda.ResidentVec
		wq, wk, wv, wo *cuda.ResidentBF16
		wg, wu, wd     *cuda.ResidentBF16
	}
	ls := make([]*layer, layers)
	for i := range ls {
		ls[i] = &layer{
			gAttn: mkVec(dim), gFFN: mkVec(dim),
			wq: mkW(dim, qW), wk: mkW(dim, kvW), wv: mkW(dim, kvW), wo: mkW(qW, dim),
			wg: mkW(dim, hidden), wu: mkW(dim, hidden), wd: mkW(hidden, dim),
		}
	}
	// paged KV: one sequence per batch row, each seeded with seqLen tokens.
	pool, err := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, seqLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	xf := make([]float32, batch*dim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	rq := backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: seqLen}
	rk := backend.RoPEAttrs{Base: 10000, Heads: kvHeads, PosOffset: seqLen}

	step := func() {
		x, _ := cuda.NewDeviceF32(batch, dim)
		x.UploadF32(xf)
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gAttn, 1e-5)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			dk.Free()
			dv.Free()
			da, err := pool.BatchedDecodeAttn(dq, seqs, qHeads, kvHeads)
			if err != nil {
				b.Fatal(err)
			}
			dq.Free()
			l.wo.MatMulAccInto(da, x)
			da.Free()
			dh2, _ := x.RMSNormTo(l.gFFN, 1e-5)
			dg, _ := l.wg.MatMulDevice(dh2)
			du, _ := l.wu.MatMulDevice(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			l.wd.MatMulAccInto(dg, x)
			dg.Free()
		}
		x.Free()
	}
	step()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		step()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkBatchedForward_b1_L22(b *testing.B)  { benchBatchedForward(b, 1, 128, 22) }
func BenchmarkBatchedForward_b16_L22(b *testing.B) { benchBatchedForward(b, 16, 128, 22) }
func BenchmarkBatchedForward_b64_L22(b *testing.B) { benchBatchedForward(b, 64, 128, 22) }
