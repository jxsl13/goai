package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// MaxAttentionLogits returns, per head, the maximum pre-softmax attention logit
// max_{i,j} scale·⟨qᵢ,kⱼ⟩ over a probe batch's projected activations q,k
// [seq, heads·dk] (§T550, the quantity Kimi K2's QK-Clip monitors — exploding
// max logits are the instability Muon induces at scale). causal restricts to
// j ≤ i. Host utility (training-loop instrumentation, not on the tape).
func MaxAttentionLogits(q, k *tensor.Tensor, heads int, scale float64, causal bool) ([]float64, error) {
	if q.Ndim() != 2 || k.Ndim() != 2 || q.Shape()[1] != k.Shape()[1] {
		return nil, fmt.Errorf("nn: MaxAttentionLogits wants q,k [seq,dm] with equal dm, got %v/%v", q.Shape(), k.Shape())
	}
	dm := q.Shape()[1]
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("nn: dm %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	sq, sk := q.Shape()[0], k.Shape()[0]
	out := make([]float64, heads)
	dm2 := k.Shape()[1]
	// F64 fast path: q[i] and k[j] were re-dispatched via AtF64 across the O(heads·seq²·dk)
	// score scan (q per j, k per i). Hoist the query row, walk k's row contiguously, and
	// break the serial-dot FADD latency chain with four independent accumulators.
	// TOLERANCE-GATED ~1 ulp: the 4-way reassociation of the dk-dot shifts each score (and
	// thus the per-head max, possibly its argmax) by ~1 ulp vs the ascending-d AtF64
	// fallback below — well within the QK-clip tests' 1e-9/1e-12 gates.
	if qs, ks := flatF64(q), flatF64(k); qs != nil && ks != nil {
		for h := range heads {
			m := math.Inf(-1)
			off := h * dk
			for i := range sq {
				qrow := qs[i*dm+off : i*dm+off+dk : i*dm+off+dk]
				jmax := sk
				if causal && i+1 < sk {
					jmax = i + 1
				}
				for j := range jmax {
					krow := ks[j*dm2+off : j*dm2+off+dk : j*dm2+off+dk]
					var s0, s1, s2, s3 float64
					d := 0
					for ; d+4 <= dk; d += 4 {
						s0 += qrow[d] * krow[d]
						s1 += qrow[d+1] * krow[d+1]
						s2 += qrow[d+2] * krow[d+2]
						s3 += qrow[d+3] * krow[d+3]
					}
					s := s0 + s1 + s2 + s3
					for ; d < dk; d++ {
						s += qrow[d] * krow[d]
					}
					if s*scale > m {
						m = s * scale
					}
				}
			}
			out[h] = m
		}
		return out, nil
	}
	for h := range heads {
		m := math.Inf(-1)
		off := h * dk
		for i := range sq {
			jmax := sk
			if causal && i+1 < sk {
				jmax = i + 1
			}
			for j := range jmax {
				var s float64
				for d := range dk {
					s += q.AtF64(i, off+d) * k.AtF64(j, off+d)
				}
				if s*scale > m {
					m = s * scale
				}
			}
		}
		out[h] = m
	}
	return out, nil
}

// QKClip applies Kimi K2's QK-Clip (Team Kimi 2025, arXiv:2507.20534 — the
// MuonClip ingredient): for every head h whose observed max attention logit
// exceeds tau, the head's query and key projection columns are rescaled IN PLACE
// by √(tau/maxLogit[h]) each, so the head's logits shrink by exactly
// tau/maxLogit[h] — capping them at tau — while heads within budget stay
// BIT-untouched. Wq/Wk are [dm, heads·dk] with head h owning columns
// h·dk..(h+1)·dk (the MHA layout). Applied after each optimizer step (typically
// with Muon, whose orthogonalized updates are what drives logit growth at
// scale); returns the number of clipped heads. maxLogits comes from
// MaxAttentionLogits on a probe forward.
func QKClip(wq, wk *tensor.Tensor, heads int, maxLogits []float64, tau float64) (int, error) {
	if wq.Ndim() != 2 || wk.Ndim() != 2 {
		return 0, fmt.Errorf("nn: QKClip wants rank-2 Wq/Wk")
	}
	dm := wq.Shape()[1]
	if wk.Shape()[1] != dm {
		return 0, fmt.Errorf("nn: QKClip Wq/Wk output dims differ: %d vs %d", dm, wk.Shape()[1])
	}
	if heads <= 0 || dm%heads != 0 || len(maxLogits) != heads {
		return 0, fmt.Errorf("nn: QKClip needs dm %d divisible by heads %d and one max logit per head", dm, heads)
	}
	if tau <= 0 {
		return 0, fmt.Errorf("nn: QKClip needs tau > 0, got %g", tau)
	}
	dk := dm / heads
	clipped := 0
	for h := range heads {
		if maxLogits[h] <= tau {
			continue
		}
		gamma := math.Sqrt(tau / maxLogits[h])
		for _, w := range []*tensor.Tensor{wq, wk} {
			rows := w.Shape()[0]
			for r := range rows {
				for c := h * dk; c < (h+1)*dk; c++ {
					w.SetF64(w.AtF64(r, c)*gamma, r, c)
				}
			}
		}
		clipped++
	}
	return clipped, nil
}
