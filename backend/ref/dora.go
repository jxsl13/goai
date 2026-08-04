package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// doraWeightKernel is the fused DoRA weight decomposition (Liu et al. 2024,
// arXiv:2402.09353, Eq. 5). Given the directional matrix V[in,out] = W₀ + ΔV
// (base weight plus the LoRA update) and the trainable magnitude vector m[out],
// it rebuilds the effective weight by normalizing each output column to unit
// length and rescaling it by its magnitude:
//
//	n[j]     = ‖V[:,j]‖₂            (column / per-output-neuron L2 norm)
//	W'[i,j]  = m[j] · V[i,j] / n[j]
//
// So the direction of each output neuron's weight is V/‖V‖ and its length is the
// separately-learned m — the decomposition that lets DoRA update magnitude and
// direction independently. At init m = ‖W₀‖ and ΔV = 0, giving W' = W₀ exactly.
// Accumulation in f64 (§V10). inputs: [V, m].
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func doraWeightKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: doraweight wants (V, m), got %d inputs", len(in))
	}
	v, m := in[0], in[1]
	if v.Ndim() != 2 {
		return nil, fmt.Errorf("ref: doraweight V must be [in,out], got %v", v.Shape())
	}
	rows, cols := v.Shape()[0], v.Shape()[1]
	if m.Ndim() != 1 || m.Shape()[0] != cols {
		return nil, fmt.Errorf("ref: doraweight m must be [%d], got %v", cols, m.Shape())
	}

	out := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element AtF64/SetF64 dispatch. Column accumulation keeps the SAME
	// ascending-i order and the same mj·v/n expression — bit-identical.
	if vs, vok := f64Data(v); vok {
		if ms, mok := f64Data(m); mok {
			if os, flush, ook := outF64(out); ook {
				// Reblock the two column-strided passes (j-outer / i-inner, striding a
				// row-major buffer by `cols` per step — cache-hostile, one column touches
				// `rows` distinct lines) into row-major i-outer / j-inner sweeps with a
				// per-column accumulator array. Each `nrm[j]` still sums i ascending, and
				// the write keeps the exact `mj·v/n` expression (norm looked up per column,
				// NOT pre-folded to mj/n which would round differently) — bit-identical.
				nrm := make([]float64, cols)
				for i := 0; i < rows; i++ {
					vrow := vs[i*cols : i*cols+cols]
					for j, x := range vrow {
						//perfscan:ignore PS3017 reference oracle: intentionally simple, correctness baseline not an optimization target
						nrm[j] += x * x
					}
				}
				for j := range nrm {
					nrm[j] = math.Sqrt(nrm[j])
				}
				for i := 0; i < rows; i++ {
					base := i * cols
					vrow := vs[base : base+cols]
					orow := os[base : base+cols]
					for j, x := range vrow {
						if n := nrm[j]; n > 0 {
							//perfscan:ignore PS5003 reference oracle: intentionally simple, correctness baseline not an optimization target
							orow[j] = ms[j] * x / n
						} else {
							orow[j] = 0
						}
					}
				}
				flush()
				return []*tensor.Tensor{out}, nil
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for j := range cols {
		var ss float64
		for i := range rows {
			x := v.AtF64(i, j)
			ss += x * x
		}
		n := math.Sqrt(ss)
		mj := m.AtF64(j)
		for i := range rows {
			if n > 0 {
				out.SetF64(mj*v.AtF64(i, j)/n, i, j)
			} else {
				out.SetF64(0, i, j)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpDoRAWeight, tensor.F32, doraWeightKernel)
	std.add(backend.OpDoRAWeight, tensor.F64, doraWeightKernel)
}
