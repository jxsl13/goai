package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// QGaLore (Zhang et al. 2024, arXiv:2407.08296, "Q-GaLore: Quantized GaLore with
// INT4 Projection and Layer-Adaptive Low-Rank Gradients"): a strictly more
// memory-efficient GaLore (arXiv:2403.03507, nn.GaLore). GaLore already cuts the
// Adam optimizer state from O(m·n) to O(r·max(m,n)) by running the moments in a
// rank-r SVD subspace of the gradient; Q-GaLore keeps that projection UNCHANGED and
// wins more memory by storing the subspace moments THEMSELVES in low precision.
//
// Concretely, for a matrix parameter W[m,n] Q-GaLore does exactly what GaLore does —
// project the gradient into the top-r singular subspace P, run Adam there, project
// the update back, apply W ← W − lr·Scale·P·Adam(PᵀG), refresh P by SVD every Gap
// steps — but the first- and second-moment vectors m, v live QUANTIZED as INT8
// (block-wise, one absmax scale per BlockSize-element block, on a NON-LINEAR log-
// magnitude grid dense near zero, the bitsandbytes 8-bit-optimizer scheme GaLore
// itself uses; a linear grid would round small second-moments to zero and blow up the
// Adam denominator). Each step dequantizes a block to fp, runs the EMA update, and
// requantizes. That is ~8× less state than GaLore's fp64 moments (1 byte per element +
// a small per-block scale, versus 8), on top of GaLore's already-large win over Adam.
// Non-matrix parameters use plain fp Adam, as in GaLore.
//
// COLLAPSE: with quantization disabled (WithQGaLoreQuantBits(0) — also 64) the
// moments are stored in fp and Q-GaLore reduces EXACTLY to GaLore, bit-identical at
// the same rank/scale/gap/betas. The only thing quantization changes is the moment
// precision.
//
// Shipped here: the INT8-quantized moment state (the core memory win, arXiv:2407.08296
// §3.2). Two further ideas from the paper are documented follow-ups: INT4 storage of
// the projection matrix P itself (§3.2, a smaller secondary win), and the
// convergence-based adaptive subspace refresh that skips SVDs while the gradient
// subspace is stable (§3.1) — here the refresh cadence stays GaLore's fixed Gap.
//
// Further reading: Zhang et al. 2024 (arXiv:2407.08296); the GaLore paper it builds
// on (Zhao et al. 2024, arXiv:2403.03507); Dettmers et al. "8-bit Optimizers via
// Block-wise Quantization" 2022 (arXiv:2110.02861) for the block-wise INT8 state; and
// Goodfellow, Bengio & Courville, Deep Learning (2016), ch. 8, for the Adam family.
type QGaLore struct {
	Params []*tensor.Tensor // parameters this optimizer updates
	LR     float64          // learning rate
	Rank   int              // projection rank r (default 128, capped at the reduced dim)
	Scale  float64          // update scale α (default 0.25)
	Gap    int              // steps between SVD subspace refreshes (default 200)
	Beta1  float64          // Adam first-moment decay (default 0.9)
	Beta2  float64          // Adam second-moment decay (default 0.999)
	Eps    float64          // Adam denominator epsilon (default 1e-8)
	// QuantBits selects the precision of the stored subspace Adam moments. 8 (default)
	// stores m, v as block-wise INT8 — the Q-GaLore memory win. SPECIAL VALUES: 0 and
	// 64 both mean "no quantization", storing the moments in fp64 so the optimizer
	// collapses bit-identically onto nn.GaLore (the §V16 collapse anchor). Only INT8
	// is implemented for the quantized path; any other non-zero, non-64 value is
	// treated as 8. Default 8 (arXiv:2407.08296 §3.2 quantizes the states to 8-bit).
	QuantBits int
	// BlockSize is the block length for the block-wise INT8 moment quantization: the
	// moments are split into contiguous BlockSize-element blocks, each with its own
	// absmax scale, so a few large entries can't wreck the whole vector's resolution.
	// Boundary behavior — small blocks track local magnitude best (finest accuracy)
	// but pay more per-block scale overhead (eroding the memory win); large blocks
	// amortize the scales but let one outlier coarsen a big span. Ignored when
	// quantization is off. Default 256 (the bitsandbytes 8-bit-Adam block size GaLore
	// adopts, arXiv:2110.02861 §3).
	BlockSize int

	t  int
	st []*qgaloreState // per-parameter state (nil until first use)
}

// qgaloreState holds one parameter's projection and its Adam moments. Matrix
// parameters keep the moments either quantized (mq/vq codes + ms/vs block scales,
// when quantization is on) or in fp (mf/vf); non-matrix parameters always use fp.
type qgaloreState struct {
	proj [][]float64 // top-r singular vectors P (proj[k] is the k-th, length = reduced dim)
	left bool        // true: project rows (m≤n), R=PᵀG; false: project cols, R=GP
	n    int         // logical moment length (= reduced size, or Numel for non-matrix)

	mf, vf []float64 // fp moments (used when quantization is off, or for non-matrix params)
	mq, vq []int8    // INT8 moment codes (used when quantization is on)
	ms, vs []float64 // per-block absmax scales for mq, vq
}

// QGaLoreOption configures a QGaLore optimizer (functional-options idiom, §C12).
type QGaLoreOption func(*QGaLore)

// WithQGaLoreRank sets the projection rank r — the dimension of the low-rank subspace
// the (quantized) Adam optimizer state is kept in, exactly as in nn.GaLore.
//
// In plain terms: Q-GaLore, like GaLore, compresses the optimizer's memory by working
// in a small r-dimensional slice of each weight matrix; r is that slice's size, and
// Q-GaLore then also stores the slice's moments in INT8. Boundary behavior — small r
// saves the most memory but may miss update directions (underfitting the step); large
// r approaches full-rank Adam and loses the low-rank savings.
//
// Default 128 (research-grounded: the GaLore reference rank Q-GaLore inherits,
// arXiv:2403.03507 / arXiv:2407.08296).
func WithQGaLoreRank(r int) QGaLoreOption { return func(g *QGaLore) { g.Rank = r } }

// WithQGaLoreScale sets the update scale α applied to the projected-back gradient.
//
// In plain terms: how strongly the low-rank update is scaled before it hits the
// weights — the analogue of a LoRA scaling factor. Boundary behavior — too small
// underuses the update; too large overshoots. Default 0.25 (research-grounded: the
// GaLore reference scale Q-GaLore inherits, arXiv:2403.03507).
func WithQGaLoreScale(a float64) QGaLoreOption { return func(g *QGaLore) { g.Scale = a } }

// WithQGaLoreGap sets the number of steps between SVD refreshes of the low-rank
// subspace.
//
// In plain terms: Q-GaLore periodically recomputes (via SVD) which low-rank slice to
// work in; this is how many steps it reuses one before refreshing. Boundary behavior —
// small gap keeps the subspace accurate but pays frequent SVD cost; large gap
// amortizes the SVD but lets the subspace go stale. Q-GaLore's paper additionally
// SKIPS refreshes adaptively once the subspace stabilizes (a documented follow-up
// here); this knob is the fixed baseline cadence. Default 200 (research-grounded: the
// GaLore reference update gap, arXiv:2403.03507).
func WithQGaLoreGap(gap int) QGaLoreOption { return func(g *QGaLore) { g.Gap = gap } }

// WithQGaLoreBetas sets the inner Adam's moment EMA decays β₁, β₂ (applied within the
// low-rank subspace, on the quantized moments).
//
// In plain terms: same momentum/variance smoothing as Adam, just done in Q-GaLore's
// compressed, quantized space. Boundary behavior as in Adam. Defaults 0.9, 0.999
// (research-grounded: standard Adam values, GaLore/Q-GaLore papers).
func WithQGaLoreBetas(b1, b2 float64) QGaLoreOption {
	return func(g *QGaLore) { g.Beta1, g.Beta2 = b1, b2 }
}

// WithQGaLoreQuantBits sets the precision of the stored subspace Adam moments and is
// the knob behind the §V16 collapse anchor.
//
// In plain terms: 8 (the default) is real Q-GaLore — the moments are stored in one
// byte each (block-wise INT8), the memory win over GaLore. The SPECIAL VALUES 0 and
// 64 turn quantization OFF: the moments are kept in full fp64 precision and the
// optimizer becomes bit-for-bit identical to nn.GaLore, which is how the collapse is
// tested. Boundary behavior — only INT8 is implemented for the quantized path, so any
// other non-zero, non-64 value is treated as 8-bit. Default 8 (arXiv:2407.08296 §3.2
// stores the optimizer states in 8-bit).
func WithQGaLoreQuantBits(bits int) QGaLoreOption { return func(g *QGaLore) { g.QuantBits = bits } }

// WithQGaLoreBlockSize sets the block length for the block-wise INT8 moment
// quantization.
//
// In plain terms: the quantizer picks one scale per block of this many moment
// entries, so an outlier only coarsens its own block, not the whole vector. Boundary
// behavior — small blocks give the finest accuracy but store more per-block scales
// (shrinking the memory win); large blocks store fewer scales but let one big entry
// coarsen a wide span; ignored entirely when quantization is off. Default 256 (the
// bitsandbytes 8-bit-Adam block size, arXiv:2110.02861 §3).
func WithQGaLoreBlockSize(bs int) QGaLoreOption {
	return func(g *QGaLore) { g.BlockSize = bs }
}

// NewQGaLore builds a Q-GaLore optimizer over params with learning rate lr and the
// paper's defaults (rank 128, scale 0.25, gap 200, Adam β 0.9/0.999, INT8 moments in
// 256-element blocks). With WithQGaLoreQuantBits(0) it is exactly nn.GaLore.
func NewQGaLore(params []*tensor.Tensor, lr float64, opts ...QGaLoreOption) *QGaLore {
	g := &QGaLore{
		Params: params, LR: lr, Rank: 128, Scale: 0.25, Gap: 200,
		Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, QuantBits: 8, BlockSize: 256,
	}
	for _, o := range opts {
		o(g)
	}
	g.st = make([]*qgaloreState, len(params))
	return g
}

// quantized reports whether the subspace moments are stored in INT8 (true) or kept in
// fp (false — the nn.GaLore-collapse mode, QuantBits 0 or 64).
func (g *QGaLore) quantized() bool { return g.QuantBits != 0 && g.QuantBits != 64 }

// qgLogRange is the number of binary orders of magnitude below a block's absmax that
// the non-linear INT8 code grid represents. A magnitude smaller than absmax·2^-range
// underflows to code 0 (exact zero); everything above is stored on a logarithmic grid,
// so the reconstruction error is RELATIVE (bounded ≈ 2^(range/252)−1), not absolute.
// This near-zero density is what keeps the quantized second moment from collapsing to
// zero and blowing up the Adam denominator — the reason plain linear INT8 fails on
// optimizer states and 8-bit optimizers use a non-linear/exponent map (Dettmers et al.
// 2022, arXiv:2110.02861; the bitsandbytes dynamic quantization type).
const qgLogRange = 20.0

// quantizeBlockwise encodes x into signed INT8 codes with one absmax scale per
// blockSize-element block, on a NON-LINEAR (log-magnitude) grid: the sign is the code's
// sign, and |code|∈[1,127] is the value's magnitude quantized in the log2 domain over
// [absmax·2^-qgLogRange, absmax]. Code 0 means an exact zero (or an underflow below the
// grid's floor). An all-zero block gets scale 0. Preserving relative precision near
// zero (unlike linear INT8) is what makes the quantized moments usable — the block-wise
// 8-bit scheme GaLore/Q-GaLore adopt from bitsandbytes (arXiv:2110.02861 §3).
func quantizeBlockwise(x []float64, blockSize int, codes []int8, scales []float64) {
	n := len(x)
	for b := 0; b*blockSize < n; b++ {
		lo := b * blockSize
		hi := min(lo+blockSize, n)
		var amax float64
		for i := lo; i < hi; i++ {
			if a := math.Abs(x[i]); a > amax {
				amax = a
			}
		}
		scales[b] = amax
		if amax == 0 {
			for i := lo; i < hi; i++ {
				codes[i] = 0
			}
			continue
		}
		for i := lo; i < hi; i++ {
			u := math.Abs(x[i]) / amax // ∈ [0,1]
			if u == 0 {
				codes[i] = 0
				continue
			}
			e := math.Log2(u) // ∈ (−∞, 0]
			if e < -qgLogRange {
				codes[i] = 0 // underflow → exact zero
				continue
			}
			lvl := int(math.Round((1+e/qgLogRange)*126)) + 1 // ∈ [1,127]
			if lvl < 1 {
				lvl = 1
			} else if lvl > 127 {
				lvl = 127
			}
			if math.Signbit(x[i]) {
				lvl = -lvl
			}
			codes[i] = int8(lvl)
		}
	}
}

// dequantizeBlockwise reconstructs x̂ ≈ x from block-wise log-magnitude INT8 codes:
// code 0 → 0, else x̂ = sign(code)·absmax·2^e with e = ((|code|−1)/126 − 1)·qgLogRange.
// The per-element error is RELATIVE, bounded by ≈ 2^(qgLogRange/252)−1.
func dequantizeBlockwise(codes []int8, scales []float64, blockSize int, out []float64) {
	for i := range codes {
		c := int(codes[i])
		if c == 0 {
			out[i] = 0
			continue
		}
		mag := c
		if mag < 0 {
			mag = -mag
		}
		e := ((float64(mag-1))/126 - 1) * qgLogRange
		v := scales[i/blockSize] * math.Exp2(e)
		if c < 0 {
			v = -v
		}
		out[i] = v
	}
}

// numBlocks is the block count for an n-element vector at the given block size.
func numBlocks(n, blockSize int) int { return (n + blockSize - 1) / blockSize }

// StateBytes reports the total optimizer-state memory (moment storage) in bytes across
// all parameters seen so far. Quantized matrix moments count 1 byte per INT8 code plus
// 8 bytes per block scale; fp moments (non-matrix params, or the quantization-off
// collapse mode) count 8 bytes each. Meaningful only after at least one Step has
// allocated the state; it is the concrete measure of Q-GaLore's memory win — compare a
// QuantBits(8) optimizer against a QuantBits(0) one on the same params.
func (g *QGaLore) StateBytes() int {
	var b int
	for _, st := range g.st {
		if st == nil {
			continue
		}
		if st.mq != nil {
			b += len(st.mq) + len(st.vq)       // INT8 codes, 1 byte each
			b += (len(st.ms) + len(st.vs)) * 8 // fp64 block scales
		} else {
			b += (len(st.mf) + len(st.vf)) * 8 // fp64 moments
		}
	}
	return b
}

// Step applies one Q-GaLore update. Parameters with a nil gradient are skipped.
func (g *QGaLore) Step(grad GradFn) error {
	g.t++
	b1c := 1 - math.Pow(g.Beta1, float64(g.t))
	b2c := 1 - math.Pow(g.Beta2, float64(g.t))
	quant := g.quantized()
	bs := g.BlockSize

	for pi, p := range g.Params {
		gt := grad(p)
		if gt == nil {
			continue
		}
		if !gt.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: QGaLore grad shape %v != param %v", gt.Shape(), p.Shape())
		}
		st := g.st[pi]

		// Non-matrix parameters: plain fp Adam on the full gradient, exactly as GaLore
		// (the projection — and the quantization — apply only to 2-D weights).
		if p.Ndim() != 2 {
			if st == nil {
				st = &qgaloreState{mf: make([]float64, p.Numel()), vf: make([]float64, p.Numel())}
				g.st[pi] = st
			}
			for i := range p.Numel() {
				idx := tensor.Unravel(i, p.Shape())
				gv := gt.AtF64(idx...)
				st.mf[i] = g.Beta1*st.mf[i] + (1-g.Beta1)*gv
				st.vf[i] = g.Beta2*st.vf[i] + (1-g.Beta2)*gv*gv
				upd := (st.mf[i] / b1c) / (math.Sqrt(st.vf[i]/b2c) + g.Eps)
				p.SetF64(p.AtF64(idx...)-g.LR*upd, idx...)
			}
			continue
		}

		mat := matAt(gt)
		rows, cols := p.Shape()[0], p.Shape()[1]
		if st == nil {
			proj, left := galoreProjection(mat, g.Rank)
			r := len(proj)
			n := r * cols
			if !left {
				n = rows * r
			}
			st = &qgaloreState{proj: proj, left: left, n: n}
			if quant {
				nb := numBlocks(n, bs)
				st.mq, st.vq = make([]int8, n), make([]int8, n)
				st.ms, st.vs = make([]float64, nb), make([]float64, nb)
			} else {
				st.mf, st.vf = make([]float64, n), make([]float64, n)
			}
			g.st[pi] = st
		} else if g.Gap > 0 && g.t%g.Gap == 0 {
			st.proj, st.left = galoreProjection(mat, g.Rank) // refresh subspace; keep moments
		}

		// project → dequantize moments → Adam in subspace → requantize → project back.
		red := galoreProjectDown(mat, st.proj, st.left)
		mf, vf := st.mf, st.vf
		if quant {
			mf = make([]float64, st.n)
			vf = make([]float64, st.n)
			dequantizeBlockwise(st.mq, st.ms, bs, mf)
			dequantizeBlockwise(st.vq, st.vs, bs, vf)
		}
		upd := make([]float64, len(red))
		for i := range red {
			mf[i] = g.Beta1*mf[i] + (1-g.Beta1)*red[i]
			vf[i] = g.Beta2*vf[i] + (1-g.Beta2)*red[i]*red[i]
			upd[i] = (mf[i] / b1c) / (math.Sqrt(vf[i]/b2c) + g.Eps)
		}
		if quant {
			quantizeBlockwise(mf, bs, st.mq, st.ms)
			quantizeBlockwise(vf, bs, st.vq, st.vs)
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
