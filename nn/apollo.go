package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// APOLLO (Zhu, Zhang, Hao et al. 2024, arXiv:2412.05270): memory-efficient
// FULL-parameter training by Approximated Gradient Scaling. APOLLO's insight is
// that most of the benefit of low-rank optimizers (and of AdamW itself, reformulated
// as an adaptive learning-rate rule over the raw gradient) comes from CHANNEL-WISE
// gradient scaling, not from the exact SVD subspace. So for a matrix parameter
// W[m,n] APOLLO projects the gradient into an AUXILIARY rank-r space with a cheap
// seeded RANDOM Gaussian projection P (no SVD): R = P·G. It runs Adam-style moment
// updates THERE — the moments are O(r·max(m,n)) instead of O(m·n), the memory win —
// forms the normalized update Ř = m̂/(√v̂+ε), and converts it into one scaling
// factor per channel j (along the larger dimension):
//
//	s_j = ‖Ř[:,j]‖₂ / ‖R[:,j]‖₂
//
// That scaling is applied to the RAW FULL-RANK gradient — W ← W − lr·α·(G·diag(s))
// — with a Norm-Growth Limiter (‖update‖ may grow at most γ× per step) taming
// early-training spikes. The projection is resampled from a fresh seed every Gap
// steps. Non-matrix parameters use plain Adam.
//
// Contrast with GaLore (arXiv:2403.03507, nn.GaLore): GaLore computes its subspace
// by SVD, runs Adam in it, and projects the UPDATE back — the applied update is
// low-rank. APOLLO's projection is random (no SVD cost), its low-rank space is only
// auxiliary bookkeeping for the scaling estimate, and both the weight AND its
// update stay full-rank; only the optimizer state is compressed. The headline:
// AdamW-level pre-training quality at SGD-like optimizer memory, and the rank-1
// APOLLO-Mini variant (WithAPOLLOMini) needs just O(max(m,n)) state per matrix.
type APOLLO struct {
	Params []*tensor.Tensor // parameters this optimizer updates
	LR     float64          // learning rate
	Rank   int              // auxiliary projection rank r (default 128, capped at the reduced dim)
	Scale  float64          // update scale α (default 1; WithAPOLLOMini sets √128)
	Gap    int              // steps between random-projection reseeds (default 200)
	Beta1  float64          // Adam first-moment decay (default 0.9)
	Beta2  float64          // Adam second-moment decay (default 0.999)
	Eps    float64          // Adam denominator epsilon (default 1e-8)
	// Gamma is the Norm-Growth Limiter threshold γ: the scaled gradient's norm may
	// grow at most γ× between consecutive steps, else it is clipped back to
	// γ·‖previous‖ (arXiv:2412.05270 eq. 4, a gentler gradient clipping that
	// prevents early-stage loss spikes). Boundary behavior: →1 freezes the update
	// norm; large γ never clips. SPECIAL VALUE: 0 = disabled. Default 1.01 (the
	// paper's reference value).
	Gamma float64
	// Mini selects APOLLO-Mini: a rank-1 auxiliary space with a SINGLE tensor-wise
	// scaling factor s = ‖Ř‖₂/‖R‖₂ per matrix instead of per-channel factors —
	// SGD-level optimizer memory. Set via WithAPOLLOMini, which also applies the
	// paper's compensating scale α=√128 (rank-1 estimates run smaller than
	// higher-rank ones).
	Mini bool

	rng *rand.Rand // seeded projection sampler (deterministic across runs)
	t   int
	st  []*apolloState // per-parameter state (nil until first use)
}

// apolloState holds only the random projection's two-word seed, low-rank Adam
// moments and the Norm-Growth Limiter's previous update norm. The projection is
// regenerated from the seed per step, matching the paper's no-matrix-storage claim.
type apolloState struct {
	seed1, seed2 uint64
	rank         int
	left         bool      // true: project rows (m≤n), R=P·G; false: project cols, R=G·Pᵀ
	m, v         []float64 // Adam moments in the auxiliary low-rank (or full, for non-matrix) space
	prev         float64   // ‖previous scaled gradient‖ for the Norm-Growth Limiter (0 = none yet)
}

// APOLLOOption configures an APOLLO optimizer (functional-options idiom, §C12).
type APOLLOOption func(*APOLLO)

// WithAPOLLORank sets the auxiliary projection rank r — the dimension of the random
// low-rank space APOLLO keeps its Adam moment state in.
//
// In plain terms: APOLLO never needs the full m×n optimizer state; it estimates its
// per-channel scaling from an r-dimensional random sketch of the gradient. r is that
// sketch's size. Boundary behavior — small r saves the most memory and only makes the
// scaling estimate noisier (APOLLO is robust to low rank: the paper halves GaLore's
// rank with negligible loss); large r sharpens the estimate but grows the state
// toward O(m·n). Default 128 (the paper defaults to a quarter of the original
// dimension and reports rank-robustness; 128 mirrors nn.GaLore's default so the two
// optimizers swap in place, arXiv:2412.05270 §5.4).
func WithAPOLLORank(r int) APOLLOOption { return func(a *APOLLO) { a.Rank = r } }

// WithAPOLLOScale sets the update scale α applied to the scaled full-rank gradient.
//
// In plain terms: a constant multiplier on every APOLLO update — the paper's theory
// says the random-projection estimate runs √(r/n) small, and α compensates. Boundary
// behavior — too small underuses the update; too large overshoots. Default 1 (the
// paper folds the correction into the learning rate for standard APOLLO); APOLLO-Mini
// uses √128 (arXiv:2412.05270 §4.2).
func WithAPOLLOScale(alpha float64) APOLLOOption { return func(a *APOLLO) { a.Scale = alpha } }

// WithAPOLLOGap sets the number of steps between random-projection reseeds.
//
// In plain terms: APOLLO periodically redraws its random sketch directions so the
// scaling estimate is not forever tied to one draw; this is how many steps it reuses
// one projection. Unlike GaLore's gap there is no SVD to amortize — a reseed is just
// fresh Gaussian samples. Boundary behavior — small gap decorrelates faster but
// churns the moment state's meaning; large gap keeps one sketch for longer. Default
// 200 (the paper's reference cadence, matching GaLore's, arXiv:2412.05270 §4.1.2).
func WithAPOLLOGap(gap int) APOLLOOption { return func(a *APOLLO) { a.Gap = gap } }

// WithAPOLLOBetas sets the inner Adam's moment EMA decays β₁, β₂ (applied within the
// auxiliary low-rank space).
//
// In plain terms: same momentum/variance smoothing as Adam, just done on APOLLO's
// random sketch of the gradient. Boundary behavior as in Adam. Defaults 0.9, 0.999
// (standard Adam values, arXiv:2412.05270 Algorithm 1).
func WithAPOLLOBetas(b1, b2 float64) APOLLOOption {
	return func(a *APOLLO) { a.Beta1, a.Beta2 = b1, b2 }
}

// WithAPOLLOMini selects APOLLO-Mini (arXiv:2412.05270 §4.2): rank-1 auxiliary space
// and ONE tensor-wise scaling factor s = ‖Ř‖₂/‖R‖₂ per matrix parameter.
//
// In plain terms: the extreme end of APOLLO's memory/quality dial — the optimizer
// state per matrix shrinks to O(max(m,n)), SGD-territory, and the whole matrix shares
// a single adaptive step size. The rank-1 estimate runs systematically small, so the
// paper compensates with scale α=√128, which this option also applies (override with
// a later WithAPOLLOScale if needed).
func WithAPOLLOMini() APOLLOOption {
	return func(a *APOLLO) { a.Mini, a.Rank, a.Scale = true, 1, math.Sqrt(128) }
}

// NewAPOLLO builds an APOLLO optimizer over params with learning rate lr and the
// paper's defaults (rank 128, scale 1, gap 200, Adam β 0.9/0.999, norm-growth γ 1.01).
// The seed makes the random projections — and therefore the whole trajectory —
// deterministic.
func NewAPOLLO(params []*tensor.Tensor, lr float64, seed uint64, opts ...APOLLOOption) *APOLLO {
	a := &APOLLO{
		Params: params, LR: lr, Rank: 128, Scale: 1, Gap: 200,
		Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, Gamma: 1.01,
		rng: rand.New(rand.NewPCG(seed, 0xa90110)),
	}
	for _, o := range opts {
		o(a)
	}
	a.st = make([]*apolloState, len(params))
	return a
}

// apolloProjection regenerates a Gaussian projection for a [rows,cols]
// gradient: r rows of N(0, 1/r) entries over the SMALLER dimension, so the sketch
// R keeps the larger dimension as its channel axis (paper: P[r,m] with m≤n,
// R = P·G[r,n]; mirrored to R = G·Pᵀ[m,r] when rows>cols). By the
// Johnson–Lindenstrauss lemma the 1/r-variance draw approximately preserves
// channel norms, which is all the scaling estimate needs.
func apolloProjection(rows, cols, rank int, seed1, seed2 uint64) (proj [][]float64, left bool) {
	left = rows <= cols
	d := rows
	if !left {
		d = cols
	}
	r := min(rank, d)
	std := 1 / math.Sqrt(float64(r))
	rng := rand.New(rand.NewPCG(seed1, seed2))
	proj = make([][]float64, r)
	for k := range r {
		proj[k] = make([]float64, d)
		for i := range d {
			proj[k][i] = rng.NormFloat64() * std
		}
	}
	return proj, left
}

func (a *APOLLO) reseed(st *apolloState, rows, cols int) {
	st.seed1, st.seed2 = a.rng.Uint64(), a.rng.Uint64()
	st.rank = min(a.Rank, min(rows, cols))
	st.left = rows <= cols
}

func (st *apolloState) projection(rows, cols int) [][]float64 {
	proj, _ := apolloProjection(rows, cols, st.rank, st.seed1, st.seed2)
	return proj
}

// Step applies one APOLLO update. Parameters with a nil gradient are skipped.
func (a *APOLLO) Step(grad GradFn) error {
	a.t++
	b1c := 1 - math.Pow(a.Beta1, float64(a.t))
	b2c := 1 - math.Pow(a.Beta2, float64(a.t))

	for pi, p := range a.Params {
		gt := grad(p)
		if gt == nil {
			continue
		}
		if !gt.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: APOLLO grad shape %v != param %v", gt.Shape(), p.Shape())
		}
		st := a.st[pi]

		// Non-matrix parameters: plain Adam on the full gradient (the paper applies
		// the projected scaling only to 2-D weights, as GaLore does).
		if p.Ndim() != 2 {
			if st == nil {
				st = &apolloState{m: make([]float64, p.Numel()), v: make([]float64, p.Numel())}
				a.st[pi] = st
			}
			for i := range p.Numel() {
				idx := tensor.Unravel(i, p.Shape())
				gv := gt.AtF64(idx...)
				st.m[i] = a.Beta1*st.m[i] + (1-a.Beta1)*gv
				st.v[i] = a.Beta2*st.v[i] + (1-a.Beta2)*gv*gv
				upd := (st.m[i] / b1c) / (math.Sqrt(st.v[i]/b2c) + a.Eps)
				p.SetF64(p.AtF64(idx...)-a.LR*upd, idx...)
			}
			continue
		}

		mat := matAt(gt)
		rows, cols := p.Shape()[0], p.Shape()[1]
		if st == nil {
			st = &apolloState{}
			a.reseed(st, rows, cols)
			r := st.rank
			red := r * cols
			if !st.left {
				red = rows * r
			}
			st.m, st.v = make([]float64, red), make([]float64, red)
			a.st[pi] = st
		} else if a.Gap > 0 && a.t%a.Gap == 0 {
			a.reseed(st, rows, cols) // fresh random seed; keep moments
		}

		// Sketch the gradient (R = P·G or G·Pᵀ) and run Adam on the sketch only.
		proj := st.projection(rows, cols)
		red := galoreProjectDown(mat, proj, st.left)
		upd := make([]float64, len(red))
		for i := range red {
			st.m[i] = a.Beta1*st.m[i] + (1-a.Beta1)*red[i]
			st.v[i] = a.Beta2*st.v[i] + (1-a.Beta2)*red[i]*red[i]
			upd[i] = (st.m[i] / b1c) / (math.Sqrt(st.v[i]/b2c) + a.Eps)
		}

		// Channel-wise (or tensor-wise for Mini) scaling factors s = ‖Ř‖/‖R‖.
		// The sketch's channel axis is the larger dim: columns j when left
		// (red is [r,cols] row-major), rows i otherwise (red is [rows,r]).
		r := st.rank
		ratio := func(nUpd, nRaw float64) float64 {
			if nRaw == 0 {
				return 0 // zero sketch ⇒ no scaling information; skip the channel
			}
			return math.Sqrt(nUpd) / math.Sqrt(nRaw)
		}
		var s []float64
		switch {
		case a.Mini: // single tensor-wise factor
			var nu, nr float64
			for i := range red {
				nu += upd[i] * upd[i]
				nr += red[i] * red[i]
			}
			s = []float64{ratio(nu, nr)}
		case st.left: // s[j] over columns of the [r,cols] sketch
			s = make([]float64, cols)
			for j := range cols {
				var nu, nr float64
				for k := range r {
					nu += upd[k*cols+j] * upd[k*cols+j]
					nr += red[k*cols+j] * red[k*cols+j]
				}
				s[j] = ratio(nu, nr)
			}
		default: // s[i] over rows of the [rows,r] sketch
			s = make([]float64, rows)
			for i := range rows {
				var nu, nr float64
				for k := range r {
					nu += upd[i*r+k] * upd[i*r+k]
					nr += red[i*r+k] * red[i*r+k]
				}
				s[i] = ratio(nu, nr)
			}
		}

		// Scale the RAW full-rank gradient channel-wise: u = s ⊙ G (the weight and
		// its update stay full-rank; only the moments above were low-rank).
		var norm float64
		for i := range rows {
			for j := range cols {
				f := s[0]
				if !a.Mini {
					if st.left {
						f = s[j]
					} else {
						f = s[i]
					}
				}
				mat[i][j] *= f
				norm += mat[i][j] * mat[i][j]
			}
		}
		norm = math.Sqrt(norm)

		// Norm-Growth Limiter (eq. 4): cap this step's update norm at γ× the last.
		if a.Gamma > 0 && st.prev > 0 && norm > a.Gamma*st.prev {
			clip := a.Gamma * st.prev / norm
			for i := range rows {
				for j := range cols {
					mat[i][j] *= clip
				}
			}
			norm = a.Gamma * st.prev
		}
		st.prev = norm

		for i := range rows {
			for j := range cols {
				p.SetF64(p.AtF64(i, j)-a.LR*a.Scale*mat[i][j], i, j)
			}
		}
	}
	return nil
}
