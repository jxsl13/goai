package nn

import "math"

// OneCycle default hyperparameters (PyTorch torch.optim.lr_scheduler.OneCycleLR).
const (
	OneCyclePctStart       = 0.3  // fraction of steps spent ramping LR up
	OneCycleDivFactor      = 25.0 // initial_lr = maxLR / div_factor
	OneCycleFinalDivFactor = 1e4  // min_lr = initial_lr / final_div_factor
	OneCycleMaxMomentum    = 0.95 // momentum at the LR extremes (start/end)
	OneCycleBaseMomentum   = 0.85 // momentum at the LR peak
)

// oneCycleAnnealCos is the PyTorch cosine annealing function anneal_cos(a,b,p) = b + (a−b)/2·(1+cos(πp)).
func oneCycleAnnealCos(a, b, p float64) float64 { return b + (a-b)/2*(1+math.Cos(math.Pi*p)) }

// oneCycleStepUp is the (clamped) phase-1 length round(pctStart·total).
func oneCycleStepUp(total int, pctStart float64) int {
	up := int(math.Round(pctStart * float64(total)))
	if up < 1 {
		up = 1
	}
	if up >= total {
		up = total - 1
	}
	return up
}

// OneCycleLR returns the 1cycle-policy learning rate at training step `step` of `total` (Smith
// 2019, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates",
// arXiv:1708.07120; the PyTorch OneCycleLR schedule). Unlike the monotone-decay schedules here
// (WarmupCosine, InverseSqrt, WSD), 1cycle first RAISES the LR from a small initial_lr up to maxLR
// over the first pctStart fraction of training, then LOWERS it below the start toward min_lr —
// large mid-training rates act as regularization ("super-convergence"). With cosine annealing:
//
//	initial_lr = maxLR / divFactor ,  min_lr = initial_lr / finalDivFactor
//	stepUp     = round(pctStart · total)
//	phase 1 (step < stepUp): anneal_cos(initial_lr, maxLR, step/stepUp)
//	phase 2 (step ≥ stepUp): anneal_cos(maxLR, min_lr, (step−stepUp)/(total−stepUp))
//	anneal_cos(a, b, p) = b + (a − b)/2·(1 + cos(π·p))
//
// so the LR is maxLR/divFactor at step 0, maxLR at stepUp, and min_lr at the final step. Defaults
// (any non-positive argument): pctStart 0.3, divFactor 25, finalDivFactor 1e4. Momentum is cycled
// inversely in the full policy — that half is a separate concern (a parked follow-up); this returns
// the LR curve. Steps past `total` clamp to min_lr.
//
// Convention: `total` is the number of steps, with the peak at round(pctStart·total) and step
// `total` returning min_lr. PyTorch's OneCycleLR instead indexes 0..total_steps−1 and places the
// peak at pct_start·total_steps − 1 (a quirk of its scheduler initializing at step −1) — the same
// curve and anchors up to a one-step boundary offset.
func OneCycleLR(step, total int, maxLR, pctStart, divFactor, finalDivFactor float64) float64 {
	if pctStart <= 0 || pctStart >= 1 {
		pctStart = OneCyclePctStart
	}
	if divFactor <= 0 {
		divFactor = OneCycleDivFactor
	}
	if finalDivFactor <= 0 {
		finalDivFactor = OneCycleFinalDivFactor
	}
	initialLR := maxLR / divFactor
	minLR := initialLR / finalDivFactor
	if total <= 1 {
		return maxLR
	}
	stepUp := oneCycleStepUp(total, pctStart)
	switch {
	case step <= 0:
		return initialLR
	case step >= total:
		return minLR
	case step < stepUp:
		return oneCycleAnnealCos(initialLR, maxLR, float64(step)/float64(stepUp))
	default:
		return oneCycleAnnealCos(maxLR, minLR, float64(step-stepUp)/float64(total-stepUp))
	}
}

// OneCycleMomentum returns the 1cycle-policy momentum at training step `step` of `total`, the
// companion to OneCycleLR (Smith 2019, arXiv:1708.07120; PyTorch OneCycleLR, §R233). Momentum is
// cycled INVERSELY to the LR: as the LR rises to its peak the momentum FALLS from maxMomentum to
// baseMomentum, then rises back as the LR decays — high momentum when the LR is small and low
// momentum when the LR is large keeps the effective step size steadier. Same cosine annealing and
// phase boundary as OneCycleLR:
//
//	phase 1 (step < stepUp): anneal_cos(maxMomentum, baseMomentum, step/stepUp)
//	phase 2 (step ≥ stepUp): anneal_cos(baseMomentum, maxMomentum, (step−stepUp)/(total−stepUp))
//
// so it is maxMomentum at step 0, baseMomentum at the peak (where the LR is maxLR), and back to
// maxMomentum at the end. Defaults (non-positive argument): maxMomentum 0.95, baseMomentum 0.85,
// pctStart 0.3. Steps past `total` clamp to maxMomentum. Same total-#steps / one-step boundary
// convention as OneCycleLR.
func OneCycleMomentum(step, total int, maxMomentum, baseMomentum, pctStart float64) float64 {
	if maxMomentum <= 0 {
		maxMomentum = OneCycleMaxMomentum
	}
	if baseMomentum <= 0 {
		baseMomentum = OneCycleBaseMomentum
	}
	if pctStart <= 0 || pctStart >= 1 {
		pctStart = OneCyclePctStart
	}
	if total <= 1 {
		return maxMomentum
	}
	stepUp := oneCycleStepUp(total, pctStart)
	switch {
	case step <= 0:
		return maxMomentum
	case step >= total:
		return maxMomentum
	case step < stepUp:
		return oneCycleAnnealCos(maxMomentum, baseMomentum, float64(step)/float64(stepUp))
	default:
		return oneCycleAnnealCos(baseMomentum, maxMomentum, float64(step-stepUp)/float64(total-stepUp))
	}
}
