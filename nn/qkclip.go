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
