package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LSQ — Learned Step Size Quantization (Esser, McKinstry, Bablani, Appuswamy & Modha 2020,
// "Learned Step Size Quantization", ICLR 2020, arXiv:1902.08153). Unlike the post-training
// quantizers here (GPTQ/AWQ/NF4/SmoothQuant/LLM.int8/Wanda/SparseGPT), LSQ is
// QUANTIZATION-AWARE TRAINING: it fake-quantizes weights/activations during the forward
// pass with a LEARNABLE step size s that is trained by gradient descent alongside the
// weights, so the network adapts to low-bit inference. The fake-quantizer is
//
//	v̂ = round(clip(v/s, −Qn, Qp)) · s
//
// and its round uses the straight-through estimator so gradients reach both v and s. The
// key LSQ contribution is the analytic step-size gradient and a gradient SCALE that keeps
// s's updates balanced with the weights' — see LSQQuantize.

// LSQBounds returns the integer quantization bounds (Qn, Qp) for a bits-bit quantizer:
// SIGNED (weights) → (2^{b−1}, 2^{b−1}−1); UNSIGNED (e.g. post-ReLU activations) → (0, 2^b−1).
func LSQBounds(bits int, signed bool) (qn, qp int) {
	if signed {
		return 1 << (bits - 1), (1 << (bits - 1)) - 1
	}
	return 0, (1 << bits) - 1
}

// LSQGradScale returns the step-size gradient scale g = 1/√(numel·Qp) (§2.2): the loss
// gradient w.r.t. s is multiplied by g so its magnitude matches the weight gradients.
func LSQGradScale(numel, qp int) float64 {
	return 1 / math.Sqrt(float64(numel)*float64(qp))
}

// LSQInitStep returns the paper's step-size initialization s = 2·mean(|v|)/√Qp.
func LSQInitStep(v *tensor.Tensor, qp int) float64 {
	n := v.Numel()
	if n == 0 {
		return 1
	}
	var sum float64
	for i := range n {
		sum += math.Abs(v.AtF64(tensor.Unravel(i, v.Shape())...))
	}
	return 2 * (sum / float64(n)) / math.Sqrt(float64(qp))
}

// LSQQuantize fake-quantizes v[rows,cols] with the learnable scalar step size s (shape
// [1,1]) over the integer range [−Qn, Qp], returning v̂ = round(clip(v/s,−Qn,Qp))·s. It is
// differentiable w.r.t. BOTH v and s:
//
//   - ∂v̂/∂v is 1 inside the clip interval and 0 outside (straight-through round gated by
//     the clip), so gradients flow to unclipped values only.
//   - ∂v̂/∂s follows the LSQ estimator (round(v/s)−v/s inside, −Qn/Qp when saturated),
//     obtained automatically from the (÷s, clip, STE-round, ×s) composition, then multiplied
//     by gradScale (use LSQGradScale for the paper's 1/√(N·Qp)).
//
// The gradient scale is applied with the standard trick s' = g·s + sg((1−g)·s), which has
// value s but scales its gradient by g. round is computed with a StopGradient straight
// through (as in FSQ). qn ≥ 0, qp > 0 (see LSQBounds).
func LSQQuantize(ctx *backend.Context, v, s *tensor.Tensor, qn, qp int, gradScale float64) (*tensor.Tensor, error) {
	if v.Ndim() != 2 {
		return nil, fmt.Errorf("nn: LSQQuantize wants rank-2 v[rows,cols], got %v", v.Shape())
	}
	if s.Numel() != 1 {
		return nil, fmt.Errorf("nn: LSQQuantize step size s must be a single scalar [1,1], got %v", s.Shape())
	}
	if qp <= 0 || qn < 0 {
		return nil, fmt.Errorf("nn: LSQQuantize needs qn≥0 and qp>0, got qn=%d qp=%d", qn, qp)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// s' = g·s + sg((1−g)·s): value s, gradient scaled by g (LSQ §2.4).
	gs, err := ex(backend.OpMul, nil, s, scalarTensor(s.Dtype(), gradScale))
	if err != nil {
		return nil, err
	}
	rest, err := ex(backend.OpMul, nil, s, scalarTensor(s.Dtype(), 1-gradScale))
	if err != nil {
		return nil, err
	}
	restSG, err := ex(backend.OpStopGradient, nil, rest)
	if err != nil {
		return nil, err
	}
	sUsed, err := ex(backend.OpAdd, nil, gs, restSG) // [1,1], broadcasts over v
	if err != nil {
		return nil, err
	}
	// v/s' clipped to [−Qn, Qp].
	scaled, err := ex(backend.OpDiv, nil, v, sUsed)
	if err != nil {
		return nil, err
	}
	vbar, err := ex(backend.OpClip, backend.ClipAttrs{Lo: float64(-qn), Hi: float64(qp)}, scaled)
	if err != nil {
		return nil, err
	}
	// round with a straight-through gradient: v̂bar = vbar + sg(round(vbar) − vbar).
	rounded := lsqRound(vbar)
	delta, err := ex(backend.OpSub, nil, rounded, vbar)
	if err != nil {
		return nil, err
	}
	deltaSG, err := ex(backend.OpStopGradient, nil, delta)
	if err != nil {
		return nil, err
	}
	vhat, err := ex(backend.OpAdd, nil, vbar, deltaSG) // forward = round(vbar), grad via vbar
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMul, nil, vhat, sUsed) // dequantize
}

// lsqRound returns round(vbar) elementwise as a fresh tensor — the value half of
// the LSQ straight-through round. For a contiguous, offset-0 vbar it rounds
// through the typed backing slice instead of the per-element AtF64(Unravel)/
// SetF64 dispatch; the round sweeps the full weight/activation tensor on every
// QAT forward. Bit-identical to the general path (see the slowLSQRound oracle) —
// the f64→dtype narrowing mirrors storage.setF64; F16/BF16 fall through.
func lsqRound(vbar *tensor.Tensor) *tensor.Tensor {
	rounded := tensor.New(vbar.Dtype(), vbar.Shape())
	n := vbar.Numel()
	if vbar.IsContiguous() && vbar.Offset() == 0 {
		switch vbar.Dtype() {
		case tensor.F64:
			vd, rd := vbar.Storage().F64(), rounded.Storage().F64()
			// Each rd[i] depends only on vd[i] (no cross-element reduction), so a
			// disjoint-range fan-out is bit-identical to the serial loop — the round
			// sweeps the full tensor on every QAT forward.
			parallelRange(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					rd[i] = math.Round(vd[i])
				}
			})
			return rounded
		case tensor.F32:
			vd, rd := vbar.Storage().F32(), rounded.Storage().F32()
			parallelRange(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					rd[i] = float32(math.Round(float64(vd[i])))
				}
			})
			return rounded
		}
	}
	for i := range n {
		c := tensor.Unravel(i, vbar.Shape())
		rounded.SetF64(math.Round(vbar.AtF64(c...)), c...)
	}
	return rounded
}
