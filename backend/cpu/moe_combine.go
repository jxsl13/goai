package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// moeCombineKernelCPU parallelizes the MoE expert-mixture out[t,j] = Σ_i (w[t,i]/Σw)·
// expert_i[t,j] over the independent tokens t (disjoint output rows). Byte-identical to
// the serial ref kernel (same per-token denom, denom>0 guard, ascending-i mixture,
// single f32 store-round). F64/F32 only; exotic falls back to ref.
func moeCombineKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) < 2 {
		return nil, fmt.Errorf("cpu: moecombine wants (weights, expert_0, …), got %d inputs", len(in))
	}
	w := in[0]
	experts := in[1:]
	e := len(experts)
	if w.Ndim() != 2 || w.Shape()[1] != e {
		return nil, fmt.Errorf("cpu: moecombine weights must be [T,%d], got %v", e, w.Shape())
	}
	tks := w.Shape()[0]
	if experts[0].Ndim() != 2 {
		return nil, fmt.Errorf("cpu: moecombine expert must be [T,D]")
	}
	d := experts[0].Shape()[1]
	for i, ex := range experts {
		if ex.Ndim() != 2 || ex.Shape()[0] != tks || ex.Shape()[1] != d {
			return nil, fmt.Errorf("cpu: moecombine expert %d shape %v != [%d,%d]", i, ex.Shape(), tks, d)
		}
	}
	out := tensor.NewOn(ctx.Device(), experts[0].Dtype(), tensor.Shape{tks, d})

	switch out.Dtype() {
	case tensor.F64:
		wc := w.Contiguous()
		if wc.Dtype() != tensor.F64 {
			return moeCombineFallback(ctx, in, out, w, experts, tks, d, e)
		}
		ecs := make([][]float64, e)
		for i, ex := range experts {
			ci := ex.Contiguous()
			if ci.Dtype() != tensor.F64 {
				return moeCombineFallback(ctx, in, out, w, experts, tks, d, e)
			}
			ecs[i] = ci.Storage().F64()
		}
		ws := wc.Storage().F64()
		os := out.Storage().F64()
		parallelWork(tks, d*e, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				wbase := t * e
				var denom float64
				//perfscan:ignore PS3010 denom sum over few experts (low trip); reassoc breaks byte-identity
				for i := 0; i < e; i++ {
					denom += ws[wbase+i]
				}
				base := t * d
				for j := 0; j < d; j++ {
					var acc float64
					if denom > 0 {
						//perfscan:ignore PS3010 mixture reduction over few experts; reassoc breaks byte-identity
						for i := 0; i < e; i++ {
							acc += (ws[wbase+i] / denom) * ecs[i][base+j]
						}
					}
					os[base+j] = acc
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		wc := w.Contiguous()
		if wc.Dtype() != tensor.F32 {
			return moeCombineFallback(ctx, in, out, w, experts, tks, d, e)
		}
		ecs := make([][]float32, e)
		for i, ex := range experts {
			ci := ex.Contiguous()
			if ci.Dtype() != tensor.F32 {
				return moeCombineFallback(ctx, in, out, w, experts, tks, d, e)
			}
			ecs[i] = ci.Storage().F32()
		}
		ws := wc.Storage().F32()
		os := out.Storage().F32()
		parallelWork(tks, d*e, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				wbase := t * e
				var denom float64
				//perfscan:ignore PS3010 denom sum over few experts (low trip); reassoc breaks byte-identity
				for i := 0; i < e; i++ {
					denom += float64(ws[wbase+i])
				}
				base := t * d
				for j := 0; j < d; j++ {
					var acc float64
					if denom > 0 {
						//perfscan:ignore PS3010 mixture reduction over few experts; reassoc breaks byte-identity
						for i := 0; i < e; i++ {
							acc += (float64(ws[wbase+i]) / denom) * float64(ecs[i][base+j])
						}
					}
					os[base+j] = float32(acc)
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	return moeCombineFallback(ctx, in, out, w, experts, tks, d, e)
}

// moeCombineFallback is the exotic/non-contiguous serial path (AtF64/SetF64), matching ref.
func moeCombineFallback(_ *backend.Context, _ []*tensor.Tensor, out, w *tensor.Tensor, experts []*tensor.Tensor, tks, d, e int) ([]*tensor.Tensor, error) {
	for t := 0; t < tks; t++ {
		var denom float64
		//perfscan:ignore PS1005 exotic/non-contiguous fallback path (declined-dtype), ref-matching
		for i := 0; i < e; i++ {
			denom += w.AtF64(t, i)
		}
		//perfscan:ignore PS1005 exotic/non-contiguous fallback path (declined-dtype), ref-matching
		for j := 0; j < d; j++ {
			var acc float64
			if denom > 0 {
				//perfscan:ignore PS1005,PS3010 exotic/non-contiguous fallback path (declined-dtype), ref-matching | reduction over few experts in exotic fall
				for i := 0; i < e; i++ {
					acc += (w.AtF64(t, i) / denom) * experts[i].AtF64(t, j)
				}
			}
			out.SetF64(acc, t, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMoECombine, tensor.F32, moeCombineKernelCPU)
	std.add(backend.OpMoECombine, tensor.F64, moeCombineKernelCPU)
}
