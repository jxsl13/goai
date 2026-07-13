package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaSelectKernel is attention over TWO score sources merged in ONE softmax
// (§T512): (Q1,K1, Q2,K2, V, sel[sq,sk]) — for pair (i,j) the pre-softmax score
// is qₕ·kₕ from source 1 when sel[i][j]==0 and from source 2 when sel[i][j]==1;
// −Inf excludes the pair. This is the primitive Self-Extend needs: source 1
// carries neighbor-RoPE-rotated q/k (true relative positions), source 2 the
// grouped-rotated ones, and the selector picks per pair — the paper's merged
// score matrix under a single softmax, which an additive mask cannot express.
// heads/GQA/scale follow AttnAttrs; causal/window/ALiBi are NOT applied (the
// selector expresses the structure). A fully-excluded row outputs zeros.
// Inference-only: no VJP is registered.
func mhaSelectKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 6 {
		return nil, fmt.Errorf("ref: mha_select wants (Q1,K1,Q2,K2,V,sel), got %d inputs", len(in))
	}
	q1, k1, q2, k2, v, sel := in[0], in[1], in[2], in[3], in[4], in[5]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("ref: mha_select needs rank-2 inputs")
		}
	}
	sq, dm := q1.Shape()[0], q1.Shape()[1]
	sk := k1.Shape()[0]
	if !q2.Shape().Equal(q1.Shape()) || !k2.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("ref: mha_select source shapes differ: Q %v/%v, K %v/%v",
			q1.Shape(), q2.Shape(), k1.Shape(), k2.Shape())
	}
	if sel.Shape()[0] != sq || sel.Shape()[1] != sk {
		return nil, fmt.Errorf("ref: mha_select sel must be [%d,%d], got %v", sq, sk, sel.Shape())
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: mha_select dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("ref: mha_select heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k1.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("ref: mha_select K/V must be [sk,%d], got %v/%v", kvHeads*dk, k1.Shape(), v.Shape())
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q1.Dtype(), tensor.Shape{sq, dm})
	row := make([]float64, sk)
	for h := range heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range sq {
			m := math.Inf(-1)
			for j := range sk {
				sv := sel.AtF64(i, j)
				if math.IsInf(sv, -1) {
					row[j] = math.Inf(-1)
					continue
				}
				q, k := q1, k1
				if sv != 0 {
					q, k = q2, k2
				}
				var s float64
				for d := range dk {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s *= scale
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
	std.add(backend.OpMHASelect, tensor.F32, mhaSelectKernel)
	std.add(backend.OpMHASelect, tensor.F64, mhaSelectKernel)
}
