package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DoRA weight-decomposition VJP (Liu et al. 2024, arXiv:2402.09353 Eq. 5). With
// n[j] = ‖V[:,j]‖, W'[i,j] = m[j]·V[i,j]/n[j], and s[j] = Σ_i g[i,j]·V[i,j], the
// exact gradients (differentiating THROUGH the column norm) are:
//
//	∂m[j]   = s[j] / n[j]
//	∂V[a,j] = (m[j]/n[j])·( g[a,j] − V[a,j]·s[j]/n[j]² )
//
// This is the true gradient of the forward, so it passes a finite-difference
// gradient check (§V2). NOTE: the reference PEFT/DoRA implementation DETACHES the
// column norm in its backward (dropping the V·s/n² term) purely to save memory —
// a deliberate approximation; we compute the exact gradient instead, a documented
// divergence in the backward only (the forward is identical) (§R70).
func init() {
	RegisterVJP(backend.OpDoRAWeight, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		v, m := in[0], in[1]
		rows, cols := v.Shape()[0], v.Shape()[1]

		dv := tensor.New(v.Dtype(), v.Shape())
		dm := tensor.New(m.Dtype(), m.Shape())
		// Typed fast paths (dtype-switch once → []T, no per-element AtF64/SetF64;
		// column norm/sums stay f64; single stores round once, §base-perf/C25/§T642).
		vc, mc, gc := v.Contiguous(), m.Contiguous(), g.Contiguous()
		switch v.Dtype() {
		case tensor.F64:
			if mc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
				vs, ms, gs := vc.Storage().F64(), mc.Storage().F64(), gc.Storage().F64()
				dvs, dms := dv.Storage().F64(), dm.Storage().F64()
				// Both passes walk a COLUMN of V and G at stride cols, so every row touches
				// its own cache line to consume one element — twice over, once per column
				// (PS6011). Four adjacent columns per pass read v[i, j..j+3] and g[i, j..j+3]
				// from the same line. Each accumulator still sums over ascending i, so the
				// norms and dots are bit-identical.
				//
				// mj/n and n*n are hoisted out of the write loop. s/(n*n) is NOT: `v*s/(n*n)`
				// associates left, so it is (v*s)/(n*n), and pulling s/(n*n) out would change
				// the arithmetic. A reciprocal multiply is a separate open decision.
				var sc, nn2, sv [4]float64 // mj/n, n*n, and s — carried, never reconstructed
				var okc [4]bool
				j := 0
				for ; j+4 <= cols; j += 4 {
					var ss0, ss1, ss2, ss3, s0, s1, s2, s3 float64
					for i := 0; i < rows; i++ {
						base := i*cols + j
						vq := vs[base : base+4 : base+4]
						gq := gs[base : base+4 : base+4]
						ss0 += vq[0] * vq[0]
						ss1 += vq[1] * vq[1]
						ss2 += vq[2] * vq[2]
						ss3 += vq[3] * vq[3]
						s0 += gq[0] * vq[0]
						s1 += gq[1] * vq[1]
						s2 += gq[2] * vq[2]
						s3 += gq[3] * vq[3]
					}
					all := true
					for b, pair := range [4][2]float64{{ss0, s0}, {ss1, s1}, {ss2, s2}, {ss3, s3}} {
						n := math.Sqrt(pair[0])
						okc[b] = n != 0
						if !okc[b] {
							all = false
							continue
						}
						sv[b] = pair[1]
						dms[j+b] = pair[1] / n
						sc[b], nn2[b] = ms[j+b]/n, n*n
					}
					if !all {
						// A zero column is left untouched rather than written with a zero
						// scale, so a group containing one writes column by column.
						for b := range 4 {
							if !okc[b] {
								continue
							}
							for i := 0; i < rows; i++ {
								dvs[i*cols+j+b] = sc[b] * (gs[i*cols+j+b] - vs[i*cols+j+b]*sv[b]/nn2[b])
							}
						}
						continue
					}
					for i := 0; i < rows; i++ {
						base := i*cols + j
						vq := vs[base : base+4 : base+4]
						gq := gs[base : base+4 : base+4]
						dvs[base] = sc[0] * (gq[0] - vq[0]*sv[0]/nn2[0])
						dvs[base+1] = sc[1] * (gq[1] - vq[1]*sv[1]/nn2[1])
						dvs[base+2] = sc[2] * (gq[2] - vq[2]*sv[2]/nn2[2])
						dvs[base+3] = sc[3] * (gq[3] - vq[3]*sv[3]/nn2[3])
					}
				}
				for ; j < cols; j++ {
					var ss, s float64
					for i := 0; i < rows; i++ {
						x := vs[i*cols+j]
						ss += x * x
						s += gs[i*cols+j] * x
					}
					n := math.Sqrt(ss)
					if n == 0 {
						continue
					}
					dms[j] = s / n
					an, d2 := ms[j]/n, n*n
					for i := 0; i < rows; i++ {
						dvs[i*cols+j] = an * (gs[i*cols+j] - vs[i*cols+j]*s/d2)
					}
				}
				return []*tensor.Tensor{dv, dm}, nil
			}
		case tensor.F32:
			if mc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
				vs, ms, gs := vc.Storage().F32(), mc.Storage().F32(), gc.Storage().F32()
				dvs, dms := dv.Storage().F32(), dm.Storage().F32()
				// Same column blocking as the F64 branch; arithmetic stays in float64 and
				// only the store rounds, exactly as before.
				var sc, nn2, sv [4]float64
				var okc [4]bool
				j := 0
				for ; j+4 <= cols; j += 4 {
					var ss0, ss1, ss2, ss3, s0, s1, s2, s3 float64
					for i := 0; i < rows; i++ {
						base := i*cols + j
						vq := vs[base : base+4 : base+4]
						gq := gs[base : base+4 : base+4]
						x0, x1, x2, x3 := float64(vq[0]), float64(vq[1]), float64(vq[2]), float64(vq[3])
						ss0 += x0 * x0
						ss1 += x1 * x1
						ss2 += x2 * x2
						ss3 += x3 * x3
						s0 += float64(gq[0]) * x0
						s1 += float64(gq[1]) * x1
						s2 += float64(gq[2]) * x2
						s3 += float64(gq[3]) * x3
					}
					all := true
					for b, pair := range [4][2]float64{{ss0, s0}, {ss1, s1}, {ss2, s2}, {ss3, s3}} {
						n := math.Sqrt(pair[0])
						okc[b] = n != 0
						if !okc[b] {
							all = false
							continue
						}
						sv[b] = pair[1]
						dms[j+b] = float32(pair[1] / n)
						sc[b], nn2[b] = float64(ms[j+b])/n, n*n
					}
					if !all {
						for b := range 4 {
							if !okc[b] {
								continue
							}
							for i := 0; i < rows; i++ {
								dvs[i*cols+j+b] = float32(sc[b] * (float64(gs[i*cols+j+b]) - float64(vs[i*cols+j+b])*sv[b]/nn2[b]))
							}
						}
						continue
					}
					for i := 0; i < rows; i++ {
						base := i*cols + j
						vq := vs[base : base+4 : base+4]
						gq := gs[base : base+4 : base+4]
						dvs[base] = float32(sc[0] * (float64(gq[0]) - float64(vq[0])*sv[0]/nn2[0]))
						dvs[base+1] = float32(sc[1] * (float64(gq[1]) - float64(vq[1])*sv[1]/nn2[1]))
						dvs[base+2] = float32(sc[2] * (float64(gq[2]) - float64(vq[2])*sv[2]/nn2[2]))
						dvs[base+3] = float32(sc[3] * (float64(gq[3]) - float64(vq[3])*sv[3]/nn2[3]))
					}
				}
				for ; j < cols; j++ {
					var ss, s float64
					for i := 0; i < rows; i++ {
						x := float64(vs[i*cols+j])
						ss += x * x
						s += float64(gs[i*cols+j]) * x
					}
					n := math.Sqrt(ss)
					if n == 0 {
						continue
					}
					dms[j] = float32(s / n)
					an, d2 := float64(ms[j])/n, n*n
					for i := 0; i < rows; i++ {
						dvs[i*cols+j] = float32(an * (float64(gs[i*cols+j]) - float64(vs[i*cols+j])*s/d2))
					}
				}
				return []*tensor.Tensor{dv, dm}, nil
			}
		}
		for j := range cols { // generic fallback (exotic/mixed dtype)
			var ss, s float64
			for i := range rows {
				x := v.AtF64(i, j)
				ss += x * x
				s += g.AtF64(i, j) * x
			}
			n := math.Sqrt(ss)
			if n == 0 {
				continue // degenerate zero column → zero gradients
			}
			mj := m.AtF64(j)
			dm.SetF64(s/n, j)
			for i := range rows {
				dv.SetF64(mj/n*(g.AtF64(i, j)-v.AtF64(i, j)*s/(n*n)), i, j)
			}
		}
		return []*tensor.Tensor{dv, dm}, nil
	})
}
