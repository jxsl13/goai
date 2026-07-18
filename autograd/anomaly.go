package autograd

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Anomaly detection (torch.autograd.set_detect_anomaly counterpart).
//
// # For the AI professional
//
// In anomaly mode the tape scans every value it can observe for NaN/±Inf at the
// two seams it already owns:
//
//   - FORWARD: [Tape.Record] (the backend.Recorder hook that Execute calls after
//     each op) scans the op's outputs. The Recorder interface cannot return an
//     error and package backend is a lower layer this package does not modify,
//     so a forward anomaly is STORED on the tape — [Tape.Err] exposes it
//     immediately, and every Backward variant returns it before running (a
//     forward anomaly can therefore never be silently swallowed). Panicking was
//     rejected: the package reserves panics for programming errors (duplicate
//     VJP registration, wrong-dtype storage access); an overflow in user data is
//     a numerical condition, and numerical conditions are errors here.
//   - BACKWARD: runBackward scans the gradients each VJP produces, and on a hit
//     ALSO scans the incoming output cotangent(s) to attribute blame: a finite
//     cotangent means THIS node's VJP created the non-finite value ("created");
//     a non-finite cotangent means the node merely passed on damage from
//     downstream — a non-finite Backward seed, or a fan-out accumulation that
//     overflowed ("propagated"). That attribution is the debugging value: it
//     points at the op whose backward rule needs attention rather than at every
//     op the poison subsequently flowed through. Because every VJP output is
//     scanned as it is produced, the first hit in the reverse walk IS the
//     creating node whenever the seed was finite.
//
// All four float dtypes are scanned (f32/f64 directly, f16/bf16 via their raw
// exponent bits — the small dtypes are where overflow bites first). The scans
// run ONLY in anomaly mode: when off (the default), the sole cost is one bool
// test per recorded op and per backward node (see
// BenchmarkForwardBackwardAnomalyOff).
//
// Checkpointed segments ([Checkpoint]) run their first forward without
// recording, so their interiors are scanned during the backward-pass
// recomputation (the sub-tape inherits anomaly mode), not on the first forward.
//
// # For the newcomer
//
// Training sometimes "blows up": the loss becomes NaN (not-a-number) and every
// step after that is garbage. The hard part is that the NaN you SEE is usually
// far from the op that MADE it — it spreads through every later computation.
// Anomaly mode makes the tape check every intermediate result the moment it is
// computed, so the error names the exact operation (and element) where numbers
// first went bad, and during the backward pass it tells you whether that
// operation caused the problem or just passed it along. Turn it on only while
// hunting such a bug — the checking makes every step a bit slower.

// TapeOption configures a Tape at construction ([NewTape]/[NewTapeOn]).
type TapeOption func(*Tape)

// WithAnomalyDetection enables anomaly detection on the tape: every forward
// output and every backward gradient is scanned for NaN/±Inf as it is produced,
// and the first hit is reported as a detailed error (op, tape index, element,
// created-vs-propagated for gradients). See the package-level discussion in
// anomaly.go. Off by default; when off the tape pays only a bool test per op.
func WithAnomalyDetection() TapeOption {
	return func(t *Tape) { t.anomaly = true }
}

// Err returns the anomaly stored by a forward-pass scan (nil when none, and
// always nil outside anomaly mode). Checking it right after the forward pass
// pinpoints forward overflows without waiting for Backward — but waiting is
// safe too: every Backward variant returns this error before doing any work.
func (t *Tape) Err() error { return t.anomalyErr }

// checkForward scans a just-recorded node's outputs; called from Record only in
// anomaly mode. It returns the error to store rather than storing it, keeping
// Record's fast path free of error plumbing.
func (t *Tape) checkForward(idx int, op backend.Op, inputs, outputs []*tensor.Tensor) error {
	for j, o := range outputs {
		if pos, v, bad := firstNonFinite(o); bad {
			return fmt.Errorf(
				"autograd: anomaly: forward op %v (tape index %d) produced %s at output %d element %d %v; inputs: %s",
				op, idx, classify(v), j, pos, tensor.Unravel(pos, o.Shape()), describe(inputs))
		}
	}
	return nil
}

// checkBackward scans the gradients a node's VJP produced (gins, nil slots are
// non-differentiable inputs) and, on a hit, attributes it: "created" when every
// incoming output cotangent (gouts) was finite, "propagated" when the poison
// was already in the cotangent. idx is the node's tape index.
func (t *Tape) checkBackward(idx int, n node, gouts, gins []*tensor.Tensor) error {
	for k, gin := range gins {
		if gin == nil {
			continue
		}
		pos, v, bad := firstNonFinite(gin)
		if !bad {
			continue
		}
		// Attribute: was the incoming cotangent already non-finite?
		for j, gout := range gouts {
			if gout == nil {
				continue
			}
			if gpos, gv, gbad := firstNonFinite(gout); gbad {
				return fmt.Errorf(
					"autograd: anomaly: backward of op %v (tape index %d) PROPAGATED a non-finite cotangent: incoming cotangent for output %d already had %s at element %d; grad for input %d has %s at element %d %v (the creating op is downstream — a non-finite Backward seed or an overflowed accumulation); inputs: %s",
					n.op, idx, j, classify(gv), gpos, k, classify(v), pos, tensor.Unravel(pos, gin.Shape()), describe(n.inputs))
			}
		}
		return fmt.Errorf(
			"autograd: anomaly: backward of op %v (tape index %d) CREATED %s in the grad for input %d at element %d %v (incoming cotangent was finite — this op's VJP produced the first non-finite value); inputs: %s",
			n.op, idx, classify(v), k, pos, tensor.Unravel(pos, gin.Shape()), describe(n.inputs))
	}
	return nil
}

// firstNonFinite returns the logical row-major position and (widened) value of
// the first NaN/±Inf element of x, or bad=false when all elements are finite
// (or the dtype is not a float). Contiguous f32/f64 scan their storage flat;
// contiguous f16/bf16 scan raw bits (exponent all-ones = Inf/NaN — no widening
// per element); anything non-contiguous falls back to the logical accessor.
func firstNonFinite(x *tensor.Tensor) (pos int, val float64, bad bool) {
	if !x.Dtype().IsFloat() || x.Numel() == 0 {
		return 0, 0, false
	}
	if x.IsContiguous() {
		off, n := x.Offset(), x.Numel()
		switch x.Dtype() {
		case tensor.F64:
			for i, v := range x.Storage().F64()[off : off+n] {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return i, v, true
				}
			}
			return 0, 0, false
		case tensor.F32:
			for i, v := range x.Storage().F32()[off : off+n] {
				if v64 := float64(v); math.IsNaN(v64) || math.IsInf(v64, 0) {
					return i, v64, true
				}
			}
			return 0, 0, false
		case tensor.F16, tensor.BF16:
			expMask := uint16(0x7C00) // binary16: exponent all-ones ⇒ Inf/NaN
			if x.Dtype() == tensor.BF16 {
				expMask = 0x7F80 // bfloat16 exponent field
			}
			for i, bits := range x.Storage().U16()[off : off+n] {
				if bits&expMask == expMask {
					return i, x.AtF64(tensor.Unravel(i, x.Shape())...), true
				}
			}
			return 0, 0, false
		}
	}
	for i := 0; i < x.Numel(); i++ {
		v := x.AtF64(tensor.Unravel(i, x.Shape())...)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return i, v, true
		}
	}
	return 0, 0, false
}

// classify names a non-finite value for error messages.
func classify(v float64) string {
	switch {
	case math.IsInf(v, 1):
		return "+Inf"
	case math.IsInf(v, -1):
		return "-Inf"
	default:
		return "NaN"
	}
}

// describe renders a tensor list as "[shape/dtype, ...]" for anomaly messages.
func describe(ts []*tensor.Tensor) string {
	s := "["
	for i, x := range ts {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%v/%v", x.Shape(), x.Dtype())
	}
	return s + "]"
}
