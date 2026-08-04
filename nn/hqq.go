package nn

import "math"

// HQQ — Half-Quadratic Quantization (Badri & Shrivastava 2023, "Half-Quadratic Quantization
// of Large Machine Learning Models", mobiusml.github.io; github.com/mobiusml/hqq). Unlike
// the calibration-aware quantizers here (GPTQ/AWQ/SmoothQuant need activations) HQQ is
// DATA-FREE: it quantizes weights alone. Its trick is to fit the asymmetric affine
// quantizer W_q = round(W/s + z), Ŵ = s·(W_q − z) by minimizing a ROBUST (Lp, p<1) norm of
// the quantization error instead of the L2 norm, so a few outlier weights don't drag the
// zero-point off the bulk. Only the zero-point z is optimized (the scale s is fixed from the
// group range); a half-quadratic split makes each step closed-form (a shrinkage on the error
// and a mean update on z).

// HQQOption configures HQQuantize (functional-options idiom, §C12).
type HQQOption func(*hqqCfg)

type hqqCfg struct {
	p        float64 // Lp-norm exponent (robustness), default 0.7
	iters    int     // half-quadratic iterations, default 20
	betaInit float64 // initial β
	kappa    float64 // β growth per iteration
}

// WithHQQLpNorm sets the robust-norm exponent p ∈ (0,1] used when fitting each group's
// quantization zero-point.
//
// In plain terms: HQQ minimizes the quantization error under an Lp norm; a smaller p cares
// less about a few big outlier weights and more about the bulk, giving a fit that a handful of
// extreme values can't drag around. Boundary behavior — p=1 is the ordinary soft-threshold
// (least-absolute); as p→0 the fit is increasingly outlier-robust but the solver is less
// smooth. Must be in (0, 1].
//
// Default 0.7 (research-grounded: the HQQ reference value, §R203 — robust to weight outliers
// while keeping the half-quadratic solve well-behaved).
func WithHQQLpNorm(p float64) HQQOption { return func(c *hqqCfg) { c.p = p } }

// WithHQQIters sets the number of half-quadratic optimization iterations per group.
//
// In plain terms: how many refinement passes the solver runs to fit each block's zero-point —
// more passes = a slightly better fit at more compute. Boundary behavior — too few leaves the
// zero-point under-optimized (higher quantization error); beyond convergence extra iterations
// don't help. Default 20 (research-grounded: the HQQ reference iteration count, §R203 — the
// solver converges within ~20 steps).
func WithHQQIters(n int) HQQOption { return func(c *hqqCfg) { c.iters = n } }

// HQQuantize quantizes w to `bits`-bit asymmetric integers in per-group blocks of size
// groupSize, optimizing each group's zero-point with the half-quadratic solver. It returns
// the integer codes (∈ [0, 2^bits−1]) and the per-group scale and zero-point; recover the
// weights with DequantizeHQQ. A pure-f64, data-free post-training quantizer. With 0
// iterations it is plain round-to-nearest (the initialization).
func HQQuantize(w []float64, bits, groupSize int, opts ...HQQOption) (codes []int, scale, zero []float64) {
	cfg := hqqCfg{p: 0.7, iters: 20, betaInit: 1, kappa: 1.01}
	for _, o := range opts {
		o(&cfg)
	}
	if groupSize <= 0 {
		groupSize = len(w)
	}
	n := len(w)
	maxLevel := float64(int(1)<<bits - 1) // 2^bits − 1
	codes = make([]int, n)
	ng := (n + groupSize - 1) / groupSize
	scale = make([]float64, ng)
	zero = make([]float64, ng)

	// Each group's optimization is self-contained (its own scale/zero/codes and a private
	// q scratch; shrinkLp is pure), so the group loop fans out over GOMAXPROCS bit-identically
	// to the serial loop. The per-group q allocation is hoisted to one reused per-worker buffer
	// (round() fully overwrites it before any read), dropping ~ng allocations to ~workers.
	parallelRows(ng, groupSize, func(glo, ghi int) {
		//perfscan:ignore PS6008 one-time offline weight quantization, cold path
		qbuf := make([]float64, groupSize)
		for g := glo; g < ghi; g++ {
			lo, hi := g*groupSize, min((g+1)*groupSize, n)
			wg := w[lo:hi]

			// fixed scale from the group range; zero-point maps min→0, max→maxLevel.
			mn, mx := wg[0], wg[0]
			for _, v := range wg {
				//perfscan:ignore PS3082 minmax in one-time weight quantization loop, cold
				mn, mx = math.Min(mn, v), math.Max(mx, v)
			}
			s := (mx - mn) / maxLevel
			if s == 0 {
				s = 1 // degenerate constant group
			}
			z := -mn / s

			q := qbuf[:len(wg)] // current integer levels (float-valued); overwritten by round()
			// THE CLAMP IS A COMPARISON CHAIN, NOT TWO CALLS. math.Min and math.Max are
			// function calls that carry the full NaN and signed-zero contract, and a profile of
			// this quantizer put 29% of it in archMin and archMax alone — more than twice its
			// own arithmetic.
			//
			// BIT-IDENTICAL, INCLUDING THE TWO CASES THAT MAKE THE NAIVE REWRITE WRONG.
			// `r <= 0` rather than `r < 0` is what reproduces math.Max(0, -0) == +0; written
			// with `<` the negative zero would survive. And NaN compares false against both
			// bounds, so it falls through unchanged, which is what math.Min and math.Max also
			// return when either operand is NaN.
			round := func() {
				//perfscan:ignore PS5001 divide in one-time quantization iteration, cold
				for i, v := range wg {
					r := math.Round(v/s + z)
					if r <= 0 {
						r = 0
					} else if r > maxLevel {
						r = maxLevel
					}
					q[i] = r
				}
			}
			round()

			// half-quadratic: alternate the Lp shrinkage of the error and the closed-form z.
			beta := cfg.betaInit
			for range cfg.iters {
				// (a) W_e = shrink_p(W − Ŵ, β), Ŵ = s(q − z).
				// (b) z = mean( q − (W − W_e)/s ), the argmin of the quadratic in z.
				//
				// The shrink zeroes |d| ≤ (1/β)·|d|^(p−1), and that test has a CLOSED FORM:
				// multiplying both sides by |d|^(1−p) (positive) gives |d|^(2−p) ≤ 1/β, so the
				// shrink returns 0 exactly when |d| ≤ (1/β)^(1/(2−p)). That bound depends only on
				// β and p, so it is one pow per ITERATION instead of one per weight — and math.Pow
				// is 55% of this quantizer's profile.
				//
				// The cut is pulled DOWN by a relative margin so this stays bit-identical rather
				// than merely equivalent in real arithmetic. Below cutLow the old code provably
				// returns 0, so the skip changes nothing; a value inside the margin falls through
				// to the original path and still evaluates its pow. The margin is many orders
				// above the rounding of the cut itself and costs only the vanishing fraction of
				// weights that land inside it.
				cutLow := math.Pow(1/beta, 1/(2-cfg.p)) * (1 - 1e-12)
				var zsum float64
				//perfscan:ignore PS4003 shrinkLp transcendental in offline quantization, cold
				for i, v := range wg {
					wHat := s * (q[i] - z)
					d := v - wHat
					we := 0.0
					if d > cutLow || d < -cutLow { // outside the certainly-zero band
						we = shrinkLp(d, beta, cfg.p)
					}
					zsum += q[i] - (v-we)/s
				}
				z = zsum / float64(len(wg))
				round()
				beta *= cfg.kappa
			}

			scale[g], zero[g] = s, z
			for i := range wg {
				codes[lo+i] = int(q[i])
			}
		}
	})
	return codes, scale, zero
}

// DequantizeHQQ reconstructs weights from HQQ codes and per-group scale/zero:
// ŵ[i] = scale[g]·(codes[i] − zero[g]).
func DequantizeHQQ(codes []int, scale, zero []float64, groupSize int) []float64 {
	if groupSize <= 0 {
		groupSize = len(codes)
	}
	out := make([]float64, len(codes))
	for i, c := range codes {
		g := i / groupSize
		out[i] = scale[g] * (float64(c) - zero[g])
	}
	return out
}

// shrinkLp is the proximal operator of the Lp norm (generalized soft-threshold):
// sign(x)·max(|x| − (1/β)·|x|^(p−1), 0). For p=1 it is the ordinary soft-threshold; for
// p<1 the threshold grows for small |x| (driving them to 0) and shrinks for large |x|
// (preserving outliers). x=0 maps to 0.
func shrinkLp(x, beta, p float64) float64 {
	a := math.Abs(x)
	if a == 0 {
		return 0
	}
	var thr float64
	if p == 1 {
		thr = 1 / beta
	} else {
		thr = (1 / beta) * math.Pow(a, p-1)
	}
	if a <= thr {
		return 0
	}
	return math.Copysign(a-thr, x)
}
