package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/internal/linalg"
	"github.com/jxsl13/goai/tensor"
)

// GaLore (Zhao et al. 2024, arXiv:2403.03507, §R81): memory-efficient FULL-parameter
// training by Gradient Low-Rank Projection. The weights stay full-rank; only the
// gradient and the Adam optimizer state are low-rank. For a matrix parameter W[m,n]
// GaLore projects the gradient into a rank-r subspace P (the top-r singular vectors
// of G, recomputed by SVD only every Gap steps), runs Adam THERE — so the moment
// state is O(r·max(m,n)) instead of Adam's O(m·n) — then projects the normalized
// update back and applies W ← W − lr·Scale·P·Adam(PᵀG). Non-matrix parameters use
// plain Adam. This gives near-full-rank training quality at a fraction of the
// optimizer memory (the headline: pre-training LLaMA-7B on a 24 GB GPU).
type GaLore struct {
	Params []*tensor.Tensor // parameters this optimizer updates
	LR     float64          // learning rate
	Rank   int              // projection rank r (default 128, capped at the reduced dim)
	Scale  float64          // update scale α (default 0.25)
	Gap    int              // steps between SVD subspace refreshes (default 200)
	Beta1  float64          // Adam first-moment decay (default 0.9)
	Beta2  float64          // Adam second-moment decay (default 0.999)
	Eps    float64          // Adam denominator epsilon (default 1e-8)

	t  int
	st []*galoreState // per-parameter state (nil until first use)
}

// galoreState holds one parameter's projection and Adam moments.
type galoreState struct {
	proj [][]float64 // top-r singular vectors P (proj[k] is the k-th, length = reduced dim)
	left bool        // true: project rows (m≤n), R=PᵀG; false: project cols, R=GP
	m, v []float64   // Adam moments in the projected (or full, for non-matrix) space
}

// GaLoreOption configures a GaLore optimizer (functional-options idiom, §C12).
type GaLoreOption func(*GaLore)

// WithGaLoreRank sets the projection rank r — the dimension of the low-rank subspace GaLore
// keeps its Adam optimizer state in (the whole point: memory-efficient training).
//
// In plain terms: GaLore compresses the optimizer's memory by working in a small r-dimensional
// slice of each weight matrix instead of the full thing. r is that slice's size. Boundary
// behavior — small r saves the most memory but may miss update directions (underfitting the
// step); large r approaches full-rank Adam and loses the memory savings.
//
// Default 128 (research-grounded: the GaLore paper's reference rank for LLM training, §R81).
func WithGaLoreRank(r int) GaLoreOption { return func(g *GaLore) { g.Rank = r } }

// WithGaLoreScale sets the update scale α applied to the projected-back gradient.
//
// In plain terms: how strongly the low-rank update is scaled before it hits the weights — the
// analogue of a LoRA scaling factor. Boundary behavior — too small underuses the update; too
// large overshoots. Default 0.25 (research-grounded: the GaLore reference scale, §R81).
func WithGaLoreScale(a float64) GaLoreOption { return func(g *GaLore) { g.Scale = a } }

// WithGaLoreGap sets the number of steps between SVD refreshes of the low-rank subspace.
//
// In plain terms: GaLore periodically recomputes (via SVD) which low-rank slice to work in;
// this is how many steps it reuses one before refreshing. Boundary behavior — small gap keeps
// the subspace accurate but pays frequent SVD cost; large gap amortizes the SVD but lets the
// subspace go stale. Default 200 (research-grounded: the GaLore reference update gap, §R81).
func WithGaLoreGap(gap int) GaLoreOption { return func(g *GaLore) { g.Gap = gap } }

// WithGaLoreBetas sets the inner Adam's moment EMA decays β₁, β₂ (applied within the low-rank
// subspace).
//
// In plain terms: same momentum/variance smoothing as Adam, just done in GaLore's compressed
// space. Boundary behavior as in Adam. Defaults 0.9, 0.999 (research-grounded: standard Adam
// values, GaLore paper §R81).
func WithGaLoreBetas(b1, b2 float64) GaLoreOption {
	return func(g *GaLore) { g.Beta1, g.Beta2 = b1, b2 }
}

// NewGaLore builds a GaLore optimizer over params with learning rate lr and the
// paper's defaults (rank 128, scale 0.25, gap 200, Adam β 0.9/0.999).
func NewGaLore(params []*tensor.Tensor, lr float64, opts ...GaLoreOption) *GaLore {
	g := &GaLore{Params: params, LR: lr, Rank: 128, Scale: 0.25, Gap: 200, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
	for _, o := range opts {
		o(g)
	}
	g.st = make([]*galoreState, len(params))
	return g
}

// matAt reads a parameter/gradient tensor into a dense [rows][cols] matrix.
func matAt(t *tensor.Tensor) [][]float64 {
	r, c := t.Shape()[0], t.Shape()[1]
	m := make([][]float64, r)
	for i := range r {
		m[i] = make([]float64, c)
		for j := range c {
			m[i][j] = t.AtF64(i, j)
		}
	}
	return m
}

// galoreProjection returns the top-r singular-vector projection of gradient G[m,n]:
// when m≤n the left vectors U[:, :r] (from the eigenvectors of GGᵀ), else the right
// vectors V[:, :r] (from GᵀG). proj[k] is the k-th vector; left reports the side.
func galoreProjection(g [][]float64, rank int) (proj [][]float64, left bool) {
	m, n := len(g), len(g[0])
	if m <= n {
		// GGᵀ [m,m]; its eigenvectors are the left singular vectors of G.
		gg := make([][]float64, m)
		for i := range m {
			gg[i] = make([]float64, m)
			for j := range m {
				var s float64
				for k := range n {
					s += g[i][k] * g[j][k]
				}
				gg[i][j] = s
			}
		}
		_, vecs := linalg.SymEig(gg)
		r := min(rank, m)
		return vecs[:r], true
	}
	// GᵀG [n,n]; eigenvectors are the right singular vectors of G.
	// k-OUTER rank-1 reblock (PS4009): the natural gtg[i][j]=Σ_k g[k][i]·g[k][j] has
	// k innermost, so g[k][i] strides by n each step — every load a cache miss once
	// n·m exceeds cache (12.85× slower at m=4096,n=1024). Accumulating one contiguous
	// row g[k] into gtg at a time makes both operands and the write stream stride-1 in
	// the inner j loop. BIT-IDENTICAL: each gtg[i][j] still sums over k in ascending
	// order from a freshly-zeroed cell.
	gtg := make([][]float64, n)
	for i := range n {
		gtg[i] = make([]float64, n)
	}
	// GᵀG is symmetric, so accumulate only the UPPER triangle (j≥i) and mirror once — halving the
	// O(m·n²) gram. BIT-IDENTICAL: gtg[j][i] = Σ_k g[k][j]·g[k][i] equals gtg[i][j] = Σ_k g[k][i]·g[k][j]
	// term-for-term (float mult commutes, same ascending-k order), so the mirror reproduces the value
	// the full loop would have computed. (Now worth it — SymEig's 8.9× speedup #890/#891 shrank the
	// eigensolve, so the gram is a much larger fraction of galoreProjection than before.)
	for k := range m {
		gk := g[k]
		for i := range n {
			gki := gk[i]
			gi := gtg[i]
			for j := i; j < n; j++ {
				gi[j] += gki * gk[j]
			}
		}
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			gtg[j][i] = gtg[i][j]
		}
	}
	_, vecs := linalg.SymEig(gtg)
	r := min(rank, n)
	return vecs[:r], false
}

// project maps the full gradient G into the low-rank subspace: R = PᵀG (left) or
// R = GP (right). Returned flat row-major with the reduced shape.
func galoreProjectDown(g, proj [][]float64, left bool) []float64 {
	r := len(proj)
	m, n := len(g), len(g[0])
	if left { // R[a][j] = Σ_i P[a][i] G[i][j]  → [r,n]
		// ikj/axpy: the dot form accumulated each output through one scalar, a serial
		// FMADD chain running at the FMA's latency rather than its throughput (PS4008).
		// BIT-IDENTICAL — each output still sums over i in ascending order from +0, and
		// out is freshly allocated, so it is already zeroed for the += form.
		out := make([]float64, r*n)
		for i := range m {
			gi := g[i]
			for a := range r {
				av := proj[a][i]
				oa := out[a*n : a*n+n]
				for j := range oa {
					oa[j] += av * gi[j]
				}
			}
		}
		return out
	}
	// R[i][a] = Σ_j G[i][j] P[a][j]  → [m,r]
	out := make([]float64, m*r)
	for i := range m {
		gi := g[i]
		for a := range r {
			var s float64
			pa := proj[a][:n]
			// Declined for ALLOCATION, not for speed. Muon's matmulABt is the same
			// A·Bᵀ shape with both operands contiguous in the summation index, and the
			// ikj rewrite won 2.09x there — contiguity is NOT the reason to decline.
			// What rules it out here is that the ikj form needs a transposed copy of
			// proj, an n×r allocation on every call, against a projection whose other
			// three arms allocate nothing.
			//perfscan:ignore PS4008 declined on allocation, not speed — see above
			for j := range pa {
				s += gi[j] * pa[j]
			}
			out[i*r+a] = s
		}
	}
	return out
}

// galoreProjectUp maps a reduced-space update back to full [m,n]: N = P·R (left) or
// N = R·Pᵀ (right).
func galoreProjectUp(red []float64, proj [][]float64, left bool, m, n int) [][]float64 {
	r := len(proj)
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, n)
	}
	// Both branches run ikj/axpy for the reason given on galoreProjectDown, and are
	// bit-identical for the same one: the sum over a stays ascending, and out's rows
	// are freshly allocated (hence zero) so accumulating into them is safe.
	if left { // N[i][j] = Σ_a P[a][i] R[a][j]
		for a := range r {
			pa, ra := proj[a], red[a*n:a*n+n]
			for i := range m {
				av := pa[i]
				oi := out[i][:n]
				for j := range oi {
					oi[j] += av * ra[j]
				}
			}
		}
		return out
	}
	// N[i][j] = Σ_a R[i][a] P[a][j]
	for i := range m {
		oi := out[i][:n]
		for a := range r {
			av := red[i*r+a]
			pa := proj[a][:n]
			for j := range oi {
				oi[j] += av * pa[j]
			}
		}
	}
	return out
}

// Step applies one GaLore update. Parameters with a nil gradient are skipped.
func (g *GaLore) Step(grad GradFn) error {
	g.t++
	b1c := 1 - math.Pow(g.Beta1, float64(g.t))
	b2c := 1 - math.Pow(g.Beta2, float64(g.t))

	for pi, p := range g.Params {
		gt := grad(p)
		if gt == nil {
			continue
		}
		if !gt.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: GaLore grad shape %v != param %v", gt.Shape(), p.Shape())
		}
		st := g.st[pi]

		// Non-matrix parameters: plain Adam on the full gradient (paper applies the
		// projection only to 2-D weights).
		if p.Ndim() != 2 {
			if st == nil {
				st = &galoreState{m: make([]float64, p.Numel()), v: make([]float64, p.Numel())}
				g.st[pi] = st
			}
			for i := range p.Numel() {
				idx := tensor.Unravel(i, p.Shape())
				gv := gt.AtF64(idx...)
				st.m[i] = g.Beta1*st.m[i] + (1-g.Beta1)*gv
				st.v[i] = g.Beta2*st.v[i] + (1-g.Beta2)*gv*gv
				upd := (st.m[i] / b1c) / (math.Sqrt(st.v[i]/b2c) + g.Eps)
				p.SetF64(p.AtF64(idx...)-g.LR*upd, idx...)
			}
			continue
		}

		mat := matAt(gt)
		rows, cols := p.Shape()[0], p.Shape()[1]
		if st == nil {
			proj, left := galoreProjection(mat, g.Rank)
			r := len(proj)
			red := r * cols
			if !left {
				red = rows * r
			}
			st = &galoreState{proj: proj, left: left, m: make([]float64, red), v: make([]float64, red)}
			g.st[pi] = st
		} else if g.Gap > 0 && g.t%g.Gap == 0 {
			st.proj, st.left = galoreProjection(mat, g.Rank) // refresh subspace; keep moments
		}

		// project → Adam in subspace → project back.
		red := galoreProjectDown(mat, st.proj, st.left)
		upd := make([]float64, len(red))
		for i := range red {
			st.m[i] = g.Beta1*st.m[i] + (1-g.Beta1)*red[i]
			st.v[i] = g.Beta2*st.v[i] + (1-g.Beta2)*red[i]*red[i]
			upd[i] = (st.m[i] / b1c) / (math.Sqrt(st.v[i]/b2c) + g.Eps)
		}
		n := galoreProjectUp(upd, st.proj, st.left, rows, cols)
		for i := range rows {
			for j := range cols {
				p.SetF64(p.AtF64(i, j)-g.LR*g.Scale*n[i][j], i, j)
			}
		}
	}
	return nil
}
