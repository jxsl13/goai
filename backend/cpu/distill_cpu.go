package cpu

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// distillKernelCPU is the knowledge-distillation KL loss with (a) both softmax exp passes
// vectorised via simd.ExpSumF64 and (b) the two per-element math.Log eliminated: since
// log p[j] = zt[j]/T − mt − log Σt and log q[j] = zs[j]/T − ms − log Σs, the KL term
// collapses to p[j]·((zt[j]−zs[j])/T + C) with a per-ROW constant C = (ms−mt)+(logΣs−logΣt)
// — 2 logs/row instead of 2c. Tolerance-gated (ADR-0021 f64 vexp), gated by vexpF64Fast;
// F32 / non-rank-2 / bad temp fall back to the reference kernel.
func distillKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && vexpF64Fast &&
		in[0].Dtype() == tensor.F64 && in[1].Dtype() == tensor.F64 &&
		in[0].Ndim() == 2 && in[1].Ndim() == 2 {
		zs, zt := in[0], in[1]
		b, c := zs.Shape()[0], zs.Shape()[1]
		pa, _ := attrs.(backend.DistillAttrs)
		pa = pa.WithDefaults()
		temp := pa.Temperature
		if b > 0 && zt.Shape()[0] == b && zt.Shape()[1] == c && temp > 0 {
			ss := zs.Contiguous().Storage().F64()
			ts := zt.Contiguous().Storage().F64()
			invT := 1 / temp
			p := make([]float64, c)
			qs := make([]float64, c)
			tsc := make([]float64, c)
			ssc := make([]float64, c)
			var total float64
			for i := 0; i < b; i++ {
				zti := ts[i*c : i*c+c]
				zsi := ss[i*c : i*c+c]
				mt := math.Inf(-1)
				for j, z := range zti {
					v := z * invT
					tsc[j] = v
					if v > mt {
						mt = v
					}
				}
				sumT := simd.ExpSumF64(p, tsc, mt) // p[j] = exp(zt[j]/T − mt)
				ms := math.Inf(-1)
				for j, z := range zsi {
					v := z * invT
					ssc[j] = v
					if v > ms {
						ms = v
					}
				}
				sumS := simd.ExpSumF64(qs, ssc, ms) // Σs only (qs is scratch)
				invSumT := 1 / sumT
				cc := (ms - mt) + (math.Log(sumS) - math.Log(sumT))
				var kl float64
				for j := 0; j < c; j++ {
					if pj := p[j] * invSumT; pj > 0 {
						kl += pj * ((zti[j]-zsi[j])*invT + cc)
					}
				}
				total += temp * temp * kl
			}
			out := tensor.NewOn(ctx.Device(), zs.Dtype(), tensor.Shape{})
			out.SetF64(total / float64(b))
			return []*tensor.Tensor{out}, nil
		}
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpDistill, in, attrs)
}

func init() { std.add(backend.OpDistill, tensor.F64, distillKernelCPU) }
