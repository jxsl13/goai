//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// A1 prefill throughput: prefill is compute-bound (M=seq, big GEMMs + WMMA attention), so A1's
// conversion-elimination should help LESS than decode's 1.47x. Measures it. Shared WMMA causal
// attention (f32); only GEMMs+elementwise differ f16 vs f32.
func benchA1Prefill(b *testing.B, seq, layers int, a1 bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden = 2048, 32, 4, 64, 5632
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(5))
	mkW := func(k, n int) *cuda.ResidentBF16 {
		tt := tensor.New(tensor.F32, tensor.Shape{k, n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = float32(rng.NormFloat64()) * 0.02
		}
		r, _ := cuda.NewResidentBF16(tt)
		return r
	}
	mkV := func(n int) *cuda.ResidentVec {
		tt := tensor.New(tensor.F32, tensor.Shape{n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = 1.0
		}
		r, _ := cuda.NewResidentVec(tt)
		return r
	}
	type L struct {
		gA, gF                     *cuda.ResidentVec
		wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
	}
	ls := make([]*L, layers)
	for i := range ls {
		ls[i] = &L{gA: mkV(dim), gF: mkV(dim), wq: mkW(dim, qW), wk: mkW(dim, kvW), wv: mkW(dim, kvW),
			wo: mkW(qW, dim), wg: mkW(dim, hidden), wu: mkW(dim, hidden), wd: mkW(hidden, dim)}
	}
	defer func() {
		for _, l := range ls {
			l.gA.Free()
			l.gF.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	xf := make([]float32, seq*dim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	x0, _ := cuda.NewDeviceF32(seq, dim)
	x0.UploadF32(xf)
	defer x0.Free()
	rq := backend.RoPEAttrs{Base: 10000, Heads: qHeads}
	rk := backend.RoPEAttrs{Base: 10000, Heads: kvHeads}
	invD, _ := backend.RoPEFreqs(hd, rq)
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()

	f32step := func() {
		x, _ := cuda.NewDeviceF32(seq, dim)
		x.CopyFrom(x0)
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gA, 1e-5)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			da, _ := cuda.GroupedQueryAttentionWMMA(dq, dk, dv, qHeads, kvHeads)
			dq.Free()
			dk.Free()
			dv.Free()
			l.wo.MatMulAccInto(da, x)
			da.Free()
			dh2, _ := x.RMSNormTo(l.gF, 1e-5)
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
	a1step := func() {
		x, _ := cuda.F16FromF32(x0)
		for _, l := range ls {
			dh, _ := cuda.NewDeviceF16(seq, dim)
			x.RMSNormInto(l.gA, 1e-5, dh)
			dq, _ := l.wq.MatMulF16(dh)
			dk, _ := l.wk.MatMulF16(dh)
			dv, _ := l.wv.MatMulF16(dh)
			dh.Free()
			dq.RoPE(invDev, rq)
			dk.RoPE(invDev, rk)
			dqf, _ := dq.ToF32()
			dkf, _ := dk.ToF32()
			dvf, _ := dv.ToF32()
			dq.Free()
			dk.Free()
			dv.Free()
			da, _ := cuda.GroupedQueryAttentionWMMA(dqf, dkf, dvf, qHeads, kvHeads)
			dqf.Free()
			dkf.Free()
			dvf.Free()
			da16, _ := cuda.F16FromF32(da)
			da.Free()
			l.wo.MatMulF16AddInto(da16, x) // x += da·wo (fused residual)
			da16.Free()
			dh2, _ := cuda.NewDeviceF16(seq, dim)
			x.RMSNormInto(l.gF, 1e-5, dh2)
			dg, _ := l.wg.MatMulF16(dh2)
			du, _ := l.wu.MatMulF16(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			l.wd.MatMulF16AddInto(dg, x) // x += dg·wd (fused residual)
			dg.Free()
		}
		x.Free()
	}
	step := f32step
	if a1 {
		step = a1step
	}
	step()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		step()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkPrefill_seq128_f32(b *testing.B) { benchA1Prefill(b, 128, 22, false) }
func BenchmarkPrefill_seq128_a1(b *testing.B)  { benchA1Prefill(b, 128, 22, true) }
