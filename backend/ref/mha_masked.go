package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaMaskedKernel is attention with a FREE additive mask input (§T507):
// (Q[sq,dm], K[sk,kv·dk], V, mask[sq,sk]) — score(i,j) += mask[i][j] before the
// softmax, with −Inf entries excluding key j for query i entirely. This is the
// primitive tree attention (Medusa's MedusaTreeMask) and grouped-position
// schemes (Self-Extend) need; heads/GQA/scale follow AttnAttrs like OpMHA, but
// causal/window/ALiBi are NOT applied — the mask expresses the structure.
// A fully-masked row outputs zeros. Inference-only: no VJP is registered.
func mhaMaskedKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: mha_masked wants (Q,K,V,mask), got %d inputs", len(in))
	}
	q, k, v, mask := in[0], in[1], in[2], in[3]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 || mask.Ndim() != 2 {
		return nil, fmt.Errorf("ref: mha_masked needs rank-2 Q,K,V,mask")
	}
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	if mask.Shape()[0] != sq || mask.Shape()[1] != sk {
		return nil, fmt.Errorf("ref: mha_masked mask must be [%d,%d], got %v", sq, sk, mask.Shape())
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: mha_masked dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("ref: mha_masked heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("ref: mha_masked K/V must be [sk,%d], got %v/%v", kvHeads*dk, k.Shape(), v.Shape())
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})
	row := make([]float64, sk)
	for h := range heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range sq {
			m := math.Inf(-1)
			for j := range sk {
				mv := mask.AtF64(i, j)
				if math.IsInf(mv, -1) {
					row[j] = math.Inf(-1)
					continue
				}
				var s float64
				for d := range dk {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s = s*scale + mv
				row[j] = s
				if s > m {
					m = s
				}
			}
			var sum float64
			for j := range sk {
				if math.IsInf(row[j], -1) {
					row[j] = 0
					continue
				}
				row[j] = math.Exp(row[j] - m)
				sum += row[j]
			}
			for d := range dk {
				var o float64
				if sum > 0 {
					for j := range sk {
						o += (row[j] / sum) * v.AtF64(j, kvOff+d)
					}
				}
				out.SetF64(o, i, qOff+d)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMHAMasked, tensor.F32, mhaMaskedKernel)
	std.add(backend.OpMHAMasked, tensor.F64, mhaMaskedKernel)
}
