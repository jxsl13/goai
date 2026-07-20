//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// A1 decode step INCLUDING the output projection (logits) — the honest full serving step. My other
// A1 benches measure the transformer layers only; vLLM's decode number includes the logits GEMM
// ([batch,dim]x[dim,vocab]) + sampling. Adding the logits head gives the fair like-for-like number.
func benchBatchedGraphA1Logits(b *testing.B, batch, seqLen, layers, vocab int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	ls, pool, seqs, x0 := bgBuild(b, batch, seqLen, layers)
	defer pool.Free()
	defer x0.Free()
	defer func() {
		for _, l := range ls {
			l.gAttn.Free()
			l.gFFN.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	dim, qW, kvW, hidden := bgDim, bgQHeads*bgHD, bgKVHeads*bgHD, bgHidden
	rng := rand.New(rand.NewSource(13))
	fnT := tensor.New(tensor.F32, tensor.Shape{dim})
	for i := range fnT.Storage().F32() {
		fnT.Storage().F32()[i] = 1.0
	}
	fnorm, _ := cuda.NewResidentVec(fnT)
	defer fnorm.Free()
	owT := tensor.New(tensor.F32, tensor.Shape{dim, vocab})
	for i := range owT.Storage().F32() {
		owT.Storage().F32()[i] = float32(rng.NormFloat64()) * 0.02
	}
	outW, _ := cuda.NewResidentBF16(owT)
	defer outW.Free()
	invQd, posDivQ := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen})
	invKd, posDivK := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgKVHeads, PosOffset: seqLen})
	iq := make([]float32, len(invQd))
	ik := make([]float32, len(invKd))
	for i := range invQd {
		iq[i] = float32(invQd[i])
	}
	for i := range invKd {
		ik[i] = float32(invKd[i])
	}
	iqd, _ := cuda.NewDeviceF32(1, len(iq))
	iqd.UploadF32(iq)
	defer iqd.Free()
	ikd, _ := cuda.NewDeviceF32(1, len(ik))
	ikd.UploadF32(ik)
	defer ikd.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()

	step := func() {
		x := cuda.AllocU16(batch * dim)
		cuda.CvtF32ToF16(x, x0.DevPtr(), batch*dim)
		for _, l := range ls {
			dh := cuda.AllocU16(batch * dim)
			cuda.RMSNormF16(x, dh, l.gAttn.VecPtr(), batch, dim, 1e-5)
			dq := cuda.AllocU16(batch * qW)
			dk := cuda.AllocU16(batch * kvW)
			dv := cuda.AllocU16(batch * kvW)
			cuda.GemmF16Pure(dh, l.wq.WPtr(), dq, batch, dim, qW)
			cuda.GemmF16Pure(dh, l.wk.WPtr(), dk, batch, dim, kvW)
			cuda.GemmF16Pure(dh, l.wv.WPtr(), dv, batch, dim, kvW)
			cuda.FreeDev(dh)
			cuda.RoPEF16(dq, iqd.DevPtr(), batch, bgQHeads, bgHD, seqLen, posDivQ)
			cuda.RoPEF16(dk, ikd.DevPtr(), batch, bgKVHeads, bgHD, seqLen, posDivK)
			cuda.FreeDev(dk)
			cuda.FreeDev(dv)
			dqf, _ := cuda.NewDeviceF32(batch, qW)
			cuda.CvtF16ToF32(dqf.DevPtr(), dq, batch*qW)
			cuda.FreeDev(dq)
			daf, _ := pool.BatchedDecodeAttnViewGQAf16(dqf, view, bgQHeads, bgKVHeads)
			dqf.Free()
			da := cuda.AllocU16(batch * qW)
			cuda.CvtF32ToF16(da, daf.DevPtr(), batch*qW)
			daf.Free()
			tmp := cuda.AllocU16(batch * dim)
			cuda.GemmF16Pure(da, l.wo.WPtr(), tmp, batch, qW, dim)
			cuda.FreeDev(da)
			cuda.AddF16(x, tmp, batch*dim)
			cuda.FreeDev(tmp)
			dh2 := cuda.AllocU16(batch * dim)
			cuda.RMSNormF16(x, dh2, l.gFFN.VecPtr(), batch, dim, 1e-5)
			dg := cuda.AllocU16(batch * hidden)
			du := cuda.AllocU16(batch * hidden)
			cuda.GemmF16Pure(dh2, l.wg.WPtr(), dg, batch, dim, hidden)
			cuda.GemmF16Pure(dh2, l.wu.WPtr(), du, batch, dim, hidden)
			cuda.FreeDev(dh2)
			cuda.SwiGLUF16(dg, du, batch*hidden)
			cuda.FreeDev(du)
			tmp2 := cuda.AllocU16(batch * dim)
			cuda.GemmF16Pure(dg, l.wd.WPtr(), tmp2, batch, hidden, dim)
			cuda.FreeDev(dg)
			cuda.AddF16(x, tmp2, batch*dim)
			cuda.FreeDev(tmp2)
		}
		// output head: final norm + logits GEMM [batch,dim] x [dim,vocab]
		xn := cuda.AllocU16(batch * dim)
		cuda.RMSNormF16(x, xn, fnorm.VecPtr(), batch, dim, 1e-5)
		cuda.FreeDev(x)
		logits := cuda.AllocU16(batch * vocab)
		cuda.GemmF16Pure(xn, outW.WPtr(), logits, batch, dim, vocab)
		cuda.FreeDev(xn)
		cuda.FreeDev(logits)
	}
	step()
	cuda.GraphSync()
	if err := cuda.CaptureBegin(); err != nil {
		b.Skipf("capture: %v", err)
	}
	step()
	g, err := cuda.CaptureEnd()
	if err != nil {
		b.Skipf("capture end: %v", err)
	}
	defer g.Free()
	g.Launch()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		g.Launch()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkBatchedGraphA1Logits_b512(b *testing.B) {
	benchBatchedGraphA1Logits(b, 512, 128, 22, 32000)
}
func BenchmarkBatchedGraphA1Logits_b768(b *testing.B) {
	benchBatchedGraphA1Logits(b, 768, 128, 22, 32000)
}

// Full-step (layers+final-norm+logits GEMM) at longer context: logits is context-INDEPENDENT so it
// shifts the whole decode-throughput curve down uniformly vs layers-only — the honest complete-step
// number to compare against vLLM (whose figure includes the head + sampling). Iw8.
func BenchmarkBatchedGraphA1Logits_b512_len256(b *testing.B) {
	benchBatchedGraphA1Logits(b, 512, 256, 22, 32000)
}
