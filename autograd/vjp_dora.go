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
				// PS1006: both the per-column norm/sum reductions and the ∂V write walked DOWN columns
				// (stride cols → one cache line touched per row). Restructure to ROW-MAJOR (i outer, j
				// inner) with per-column accumulators so V/g/∂V stream contiguously. Bit-identical: each
				// column's reduction still accumulates in i-ascending order, and the ∂V expression is
				// unchanged element-for-element.
				ssA := make([]float64, cols) // Σ_i V[i,j]²
				sA := make([]float64, cols)  // Σ_i g[i,j]·V[i,j]
				for i := 0; i < rows; i++ {
					base := i * cols
					for j := 0; j < cols; j++ {
						x := vs[base+j]
						ssA[j] += x * x
						sA[j] += gs[base+j] * x
					}
				}
				nA := make([]float64, cols)
				for j := 0; j < cols; j++ {
					n := math.Sqrt(ssA[j])
					nA[j] = n
					if n != 0 {
						dms[j] = sA[j] / n
					}
				}
				for i := 0; i < rows; i++ {
					base := i * cols
					for j := 0; j < cols; j++ {
						n := nA[j]
						if n == 0 {
							continue
						}
						dvs[base+j] = ms[j] / n * (gs[base+j] - vs[base+j]*sA[j]/(n*n))
					}
				}
				return []*tensor.Tensor{dv, dm}, nil
			}
		case tensor.F32:
			if mc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
				vs, ms, gs := vc.Storage().F32(), mc.Storage().F32(), gc.Storage().F32()
				dvs, dms := dv.Storage().F32(), dm.Storage().F32()
				// PS1006 row-major restructure (see the F64 path) — bit-identical: same i-ascending
				// per-column reduction order and the same ∂V expression element-for-element.
				ssA := make([]float64, cols)
				sA := make([]float64, cols)
				for i := 0; i < rows; i++ {
					base := i * cols
					for j := 0; j < cols; j++ {
						x := float64(vs[base+j])
						ssA[j] += x * x
						sA[j] += float64(gs[base+j]) * x
					}
				}
				nA := make([]float64, cols)
				for j := 0; j < cols; j++ {
					n := math.Sqrt(ssA[j])
					nA[j] = n
					if n != 0 {
						dms[j] = float32(sA[j] / n)
					}
				}
				for i := 0; i < rows; i++ {
					base := i * cols
					for j := 0; j < cols; j++ {
						n := nA[j]
						if n == 0 {
							continue
						}
						dvs[base+j] = float32(float64(ms[j]) / n * (float64(gs[base+j]) - float64(vs[base+j])*sA[j]/(n*n)))
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
