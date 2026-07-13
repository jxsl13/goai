package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DoRALinear is a Weight-Decomposed Low-Rank Adapter (Liu et al. 2024,
// arXiv:2402.09353, §R70) — DoRA. It refines LoRA (§R27) by splitting each
// output neuron's weight into a MAGNITUDE and a DIRECTION and training them
// separately. For our [in,out] weight convention, with the LoRA update
// ΔW = (alpha/r)·A·B:
//
//	V  = W + ΔW                       // directional matrix (base + low-rank update)
//	W' = m ⊙ V / ‖V‖_col              // rescale each output column to length m_j
//	y  = x·W'
//
// where ‖V‖_col[j] = ‖V[:,j]‖₂ is the per-output-column norm and m[out] is the
// trainable magnitude, initialized to ‖W‖_col so the adapter starts as an exact
// no-op (B=0 ⇒ V=W ⇒ W'=W). Only A, B and m train; W is frozen. This extra
// magnitude degree of freedom lets DoRA match full fine-tuning more closely than
// LoRA at the same rank. Freezing m at ‖V‖_col recovers plain LoRA.
type DoRALinear struct {
	W     *tensor.Tensor // [in, out], frozen base
	A     *tensor.Tensor // [in, r]
	B     *tensor.Tensor // [r, out]
	M     *tensor.Tensor // [out], trainable magnitude
	R     int            // LoRA rank
	Alpha float64        // LoRA scaling α; effective scale α/R
}

// NewDoRA wraps a frozen base weight w[in,out] with a rank-r DoRA adapter: A is
// Kaiming-initialized, B is zero, and the magnitude m is set to the column norms
// of w so the layer initially reproduces the base exactly.
func NewDoRA(w *tensor.Tensor, r int, alpha float64, seed uint64) (*DoRALinear, error) {
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: DoRA base must be [in,out], got %v", w.Shape())
	}
	in, out := w.Shape()[0], w.Shape()[1]
	if r <= 0 || r > in || r > out {
		return nil, fmt.Errorf("nn: DoRA rank %d out of range for [%d,%d]", r, in, out)
	}
	a := tensor.New(w.Dtype(), tensor.Shape{in, r})
	KaimingUniform(a, in, seed)
	b := tensor.New(w.Dtype(), tensor.Shape{r, out})
	Zeros(b)
	m := tensor.New(w.Dtype(), tensor.Shape{out})
	for j := range out {
		var ss float64
		for i := range in {
			v := w.AtF64(i, j)
			ss += v * v
		}
		m.SetF64(math.Sqrt(ss), j) // m = ‖W[:,j]‖ per output column
	}
	return &DoRALinear{W: w, A: a, B: b, M: m, R: r, Alpha: alpha}, nil
}

func (d *DoRALinear) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes y = x · (m ⊙ (W + (alpha/r)·A·B) / ‖·‖_col).
func (d *DoRALinear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	ab, err := d.exec(ctx, backend.OpMatMul, nil, d.A, d.B) // ΔW direction [in,out]
	if err != nil {
		return nil, err
	}
	v, err := d.exec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: d.Alpha / float64(d.R)}, ab, d.W) // (alpha/r)·AB + W
	if err != nil {
		return nil, err
	}
	wPrime, err := d.exec(ctx, backend.OpDoRAWeight, nil, v, d.M) // m·V/‖V‖_col
	if err != nil {
		return nil, err
	}
	return d.exec(ctx, backend.OpMatMul, nil, x, wPrime)
}

// Params returns the trainable adapter matrices A, B and the magnitude m (W is
// frozen).
func (d *DoRALinear) Params() []*tensor.Tensor { return []*tensor.Tensor{d.A, d.B, d.M} }
