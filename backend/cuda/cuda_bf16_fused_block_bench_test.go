//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// BenchmarkBf16FusedBlock measures a FULL transformer-block training step (RMSNorm + attention +
// residual + RMSNorm + SwiGLU-FFN + residual + loss + full backward + DeviceAdam), all bf16-cached,
// two ways:
//   - unfused: QKV as 3 separate projections, gate/up as 2 — the current best (bf16 weight cache #1039)
//   - fused:   QKV concatenated into one [D,3D] GEMM, gate/up into one [D,2H], with CopyCols to slice
//     Q/K/V (attention) and gate/up (SwiGLU) and to re-concat their grads on the backward.
//
// Both compute the identical block; the only difference is projection GEMM shape/count (15→6 of the
// block's ~21 GEMMs), so this converts the isolated 1.29× projection-cluster win (BenchmarkFusedQKV /
// #1040) into an integrated end-to-end step delta.
func BenchmarkBf16FusedBlock(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const L, D, H = 512, 1024, 2816
	const eps = 1e-5
	scale := float32(1.0 / math.Sqrt(float64(D)))
	rng := rand.New(rand.NewSource(91))
	must := func(e error) {
		if e != nil {
			b.Fatal(e)
		}
	}
	rm := func(r, c int, s float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(r, c)
		must(err)
		h := make([]float32, r*c)
		for i := range h {
			h[i] = float32(rng.NormFloat64()) * s
		}
		must(d.UploadF32(h))
		return d
	}
	nf := func(r, c int) *cuda.DeviceF32 { d, err := cuda.NewDeviceF32(r, c); must(err); return d }
	nbf := func(r, c int) *cuda.DeviceBf16 { d, err := cuda.NewDeviceBf16(r, c); must(err); return d }

	x, target := rm(L, D, 1), rm(L, D, 1)
	g1, g2 := nf(1, D), nf(1, D)
	g1.UploadF32(onesF(D))
	g2.UploadF32(onesF(D))
	dg1, dg2 := nf(1, D), nf(1, D)

	// Shared per-block scratch (both modes run sequentially in their own b.Run).
	h1, h2, y, out := nf(L, D), nf(L, D), nf(L, D), nf(L, D)
	Q, K, V, a := nf(L, D), nf(L, D), nf(L, D), nf(L, D)
	S := nf(L, L)
	gate, up, ff := nf(L, H), nf(L, H), nf(L, H)
	dout, dy, dA := nf(L, D), nf(L, D), nf(L, D)
	dQ, dK, dV, dP, dS := nf(L, D), nf(L, D), nf(L, D), nf(L, L), nf(L, L)
	dh1, dh2, tmp, dxN := nf(L, D), nf(L, D), nf(L, D), nf(L, D)
	dff, dgate, dup := nf(L, H), nf(L, H), nf(L, H)

	// ---- UNFUSED (baseline = bf16 weight cache) ----
	unfusedBench := func(b *testing.B) {
		Wq, Wk, Wv, Wo := rm(D, D, 0.05), rm(D, D, 0.05), rm(D, D, 0.05), rm(D, D, 0.05)
		Wg, Wu, Wd := rm(D, H, 0.05), rm(D, H, 0.05), rm(H, D, 0.05)
		Wc := []*cuda.DeviceBf16{nbf(D, D), nbf(D, D), nbf(D, D), nbf(D, D), nbf(D, H), nbf(D, H), nbf(H, D)}
		W := []*cuda.DeviceF32{Wq, Wk, Wv, Wo, Wg, Wu, Wd}
		dWq, dWk, dWv, dWo := nf(D, D), nf(D, D), nf(D, D), nf(D, D)
		dWg, dWu, dWd := nf(D, H), nf(D, H), nf(H, D)
		params := []*cuda.DeviceF32{Wq, Wk, Wv, Wo, Wg, Wu, Wd, g1, g2}
		grads := []*cuda.DeviceF32{dWq, dWk, dWv, dWo, dWg, dWu, dWd, dg1, dg2}
		opt, err := cuda.NewDeviceAdam([]int{D * D, D * D, D * D, D * D, D * H, D * H, H * D, D, D}, 1e-3, 0.9, 0.999, 1e-8, 0)
		must(err)
		defer opt.Free()
		step := func() {
			for i, w := range W {
				must(Wc[i].FromF32(w))
			}
			must(cuda.RMSNormForward(h1, x, g1, eps))
			must(cuda.MatMulWBf16(h1, Wc[0], Q))
			must(cuda.MatMulWBf16(h1, Wc[1], K))
			must(cuda.MatMulWBf16(h1, Wc[2], V))
			must(cuda.MatMulGradXBf16(Q, K, S)) // S=Q·Kᵀ
			must(cuda.Scale(S, scale))
			must(S.Softmax())
			must(cuda.MatMulBf16(S, V, a))
			must(cuda.MatMulWBf16(a, Wc[3], y)) // Wo
			must(y.Add(x))
			must(cuda.RMSNormForward(h2, y, g2, eps))
			must(cuda.MatMulWBf16(h2, Wc[4], gate))
			must(cuda.MatMulWBf16(h2, Wc[5], up))
			must(cuda.SwiGLUForward(ff, gate, up))
			must(cuda.MatMulWBf16(ff, Wc[6], out)) // Wd
			must(out.Add(y))
			must(cuda.SubScaled(dout, out, target, 2.0/float32(L*D)))
			must(cuda.MatMulGradXWBf16(dout, Wc[6], dff))
			must(cuda.MatMulGradWBf16(ff, dout, dWd))
			must(cuda.SwiGLUBackward(dgate, dup, gate, up, dff))
			must(cuda.MatMulGradXWBf16(dgate, Wc[4], dh2))
			must(cuda.MatMulGradXWBf16(dup, Wc[5], tmp))
			must(dh2.Add(tmp))
			must(cuda.MatMulGradWBf16(h2, dgate, dWg))
			must(cuda.MatMulGradWBf16(h2, dup, dWu))
			must(cuda.RMSNormBackward(dy, dg2, y, dh2, g2, eps))
			must(dy.Add(dout))
			must(cuda.MatMulGradXWBf16(dy, Wc[3], dA))
			must(cuda.MatMulGradWBf16(a, dy, dWo))
			must(cuda.MatMulGradWBf16(S, dA, dV)) // dV=Pᵀ·dO
			must(cuda.MatMulGradXBf16(dA, V, dP)) // dP=dO·Vᵀ
			must(cuda.SoftmaxBackward(dS, S, dP))
			must(cuda.Scale(dS, scale))
			must(cuda.MatMulBf16(dS, K, dQ))
			must(cuda.MatMulGradWBf16(dS, Q, dK))
			must(cuda.MatMulGradXWBf16(dQ, Wc[0], dh1))
			must(cuda.MatMulGradXWBf16(dK, Wc[1], tmp))
			must(dh1.Add(tmp))
			must(cuda.MatMulGradXWBf16(dV, Wc[2], tmp))
			must(dh1.Add(tmp))
			must(cuda.MatMulGradWBf16(h1, dQ, dWq))
			must(cuda.MatMulGradWBf16(h1, dK, dWk))
			must(cuda.MatMulGradWBf16(h1, dV, dWv))
			must(cuda.RMSNormBackward(dxN, dg1, x, dh1, g1, eps))
			must(opt.Step(params, grads))
		}
		step()
		cuda.GraphSync()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			step()
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/step")
	}

	// ---- FUSED (concat QKV & gate/up + CopyCols) ----
	fusedBench := func(b *testing.B) {
		Wqkv, WGU := rm(D, 3*D, 0.05), rm(D, 2*H, 0.05)
		Wo, Wd := rm(D, D, 0.05), rm(H, D, 0.05)
		WqkvC, WGUC, WoC, WdC := nbf(D, 3*D), nbf(D, 2*H), nbf(D, D), nbf(H, D)
		QKV, GU := nf(L, 3*D), nf(L, 2*H)
		dQKV, dGU := nf(L, 3*D), nf(L, 2*H)
		dWqkv, dWGU, dWo, dWd := nf(D, 3*D), nf(D, 2*H), nf(D, D), nf(H, D)
		params := []*cuda.DeviceF32{Wqkv, WGU, Wo, Wd, g1, g2}
		grads := []*cuda.DeviceF32{dWqkv, dWGU, dWo, dWd, dg1, dg2}
		opt, err := cuda.NewDeviceAdam([]int{D * 3 * D, D * 2 * H, D * D, H * D, D, D}, 1e-3, 0.9, 0.999, 1e-8, 0)
		must(err)
		defer opt.Free()
		step := func() {
			must(WqkvC.FromF32(Wqkv))
			must(WGUC.FromF32(WGU))
			must(WoC.FromF32(Wo))
			must(WdC.FromF32(Wd))
			must(cuda.RMSNormForward(h1, x, g1, eps))
			must(cuda.MatMulWBf16(h1, WqkvC, QKV)) // fused QKV
			must(cuda.CopyCols(Q, QKV, 0, 0, D))
			must(cuda.CopyCols(K, QKV, 0, D, D))
			must(cuda.CopyCols(V, QKV, 0, 2*D, D))
			must(cuda.MatMulGradXBf16(Q, K, S)) // S=Q·Kᵀ
			must(cuda.Scale(S, scale))
			must(S.Softmax())
			must(cuda.MatMulBf16(S, V, a))
			must(cuda.MatMulWBf16(a, WoC, y))
			must(y.Add(x))
			must(cuda.RMSNormForward(h2, y, g2, eps))
			must(cuda.MatMulWBf16(h2, WGUC, GU)) // fused gate/up
			must(cuda.CopyCols(gate, GU, 0, 0, H))
			must(cuda.CopyCols(up, GU, 0, H, H))
			must(cuda.SwiGLUForward(ff, gate, up))
			must(cuda.MatMulWBf16(ff, WdC, out))
			must(out.Add(y))
			must(cuda.SubScaled(dout, out, target, 2.0/float32(L*D)))
			must(cuda.MatMulGradXWBf16(dout, WdC, dff))
			must(cuda.MatMulGradWBf16(ff, dout, dWd))
			must(cuda.SwiGLUBackward(dgate, dup, gate, up, dff))
			must(cuda.CopyCols(dGU, dgate, 0, 0, H)) // re-concat gate/up grads
			must(cuda.CopyCols(dGU, dup, H, 0, H))
			must(cuda.MatMulGradXWBf16(dGU, WGUC, dh2))
			must(cuda.MatMulGradWBf16(h2, dGU, dWGU))
			must(cuda.RMSNormBackward(dy, dg2, y, dh2, g2, eps))
			must(dy.Add(dout))
			must(cuda.MatMulGradXWBf16(dy, WoC, dA))
			must(cuda.MatMulGradWBf16(a, dy, dWo))
			must(cuda.MatMulGradWBf16(S, dA, dV))
			must(cuda.MatMulGradXBf16(dA, V, dP))
			must(cuda.SoftmaxBackward(dS, S, dP))
			must(cuda.Scale(dS, scale))
			must(cuda.MatMulBf16(dS, K, dQ))
			must(cuda.MatMulGradWBf16(dS, Q, dK))
			must(cuda.CopyCols(dQKV, dQ, 0, 0, D)) // re-concat QKV grads
			must(cuda.CopyCols(dQKV, dK, D, 0, D))
			must(cuda.CopyCols(dQKV, dV, 2*D, 0, D))
			must(cuda.MatMulGradXWBf16(dQKV, WqkvC, dh1))
			must(cuda.MatMulGradWBf16(h1, dQKV, dWqkv))
			must(cuda.RMSNormBackward(dxN, dg1, x, dh1, g1, eps))
			must(opt.Step(params, grads))
		}
		step()
		cuda.GraphSync()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			step()
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/step")
	}

	b.Run("unfused", unfusedBench)
	b.Run("fused", fusedBench)
}
