package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/internal/fmath"
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// QGaLore implements Q-GaLore (Zhang et al. 2024, arXiv:2407.08296): GaLore's
// rank-r SVD gradient subspace plus all four defining low-memory mechanisms from
// the paper/reference implementation:
//
//  1. Adam moments in the subspace use block-wise INT8 storage.
//  2. The SVD projection itself uses packed, block-wise affine INT4 storage.
//  3. Each matrix owns an adaptive SVD cadence: after five adjacent first-vector
//     cosine similarities average at least 0.4, its refresh gap doubles.
//  4. Matrix weights are re-quantized block-wise to INT8 after every update using
//     stochastic rounding, which preserves small updates in expectation.
//
// The public tensor remains a dequantized execution mirror because tensor currently
// has no integer dtype; qgaloreState.wq is the authoritative compressed weight state.
// Thus this package validates the paper's numerics and compressed stores, while an
// integer-native Tensor is still required to realize the full end-to-end weight-memory
// saving during a model forward. Non-matrix parameters use plain fp Adam, as GaLore.
//
// COLLAPSE: WithQGaLoreQuantBits(0) (also 64) disables the entire Q-GaLore path —
// moments, projection and weights stay fp and the cadence stays fixed — so updates
// reduce bit-identically to nn.GaLore at the same rank/scale/gap/betas.
//
// Further reading: Zhang et al. 2024 (arXiv:2407.08296); the official
// VITA-Group/Q-GaLore reference; Zhao et al. 2024 (arXiv:2403.03507); and Dettmers
// et al. 2022 (arXiv:2110.02861) for block-wise 8-bit Adam state.
type QGaLore struct {
	Params []*tensor.Tensor // parameters this optimizer updates
	LR     float64          // learning rate
	Rank   int              // projection rank r (default 128, capped at the reduced dim)
	Scale  float64          // update scale α (default 0.25)
	Gap    int              // steps between SVD subspace refreshes (default 200)
	Beta1  float64          // Adam first-moment decay (default 0.9)
	Beta2  float64          // Adam second-moment decay (default 0.999)
	Eps    float64          // Adam denominator epsilon (default 1e-8)
	// QuantBits selects Q-GaLore mode. 8 (default) enables INT8 moments/weights,
	// packed INT4 projections, and adaptive refresh. SPECIAL VALUES 0 and 64 disable
	// the whole paper path for bit-identical GaLore collapse. Other values currently
	// select the same 8-bit path; lower-bit moments are not implemented.
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
	// WeightQuant and ProjectionQuant enable the two low-precision stores that
	// distinguish Q-GaLore from GaLore: INT8 model weights and packed INT4 SVD
	// projections. They default on. QuantBits 0 or 64 disables the complete
	// Q-GaLore path (including these stores) for the bit-identical GaLore anchor.
	WeightQuant bool
	// ProjectionQuant stores P as packed INT4 when paper mode is active.
	ProjectionQuant bool
	// AdaptiveRefresh implements the paper's lazy layer-wise SVD schedule. Each
	// parameter starts at Gap, keeps QueueSize adjacent first-vector cosine
	// similarities, and multiplies its own gap by GapMultiplier when their mean is
	// at least CosThreshold. Paper/reference defaults: 0.4, 2, and 5.
	AdaptiveRefresh bool
	// CosThreshold is the mean-similarity trigger for slowing SVD refreshes.
	CosThreshold float64
	// GapMultiplier scales a stable parameter's current SVD gap.
	GapMultiplier int
	// QueueSize is the number of adjacent projection similarities to average.
	QueueSize int
	// Seed controls stochastic INT8 weight rounding reproducibly.
	Seed uint64

	t   int
	st  []*qgaloreState // per-parameter state (nil until first use)
	rng *rand.Rand
}

// qgaloreState holds one parameter's projection and its Adam moments. Matrix
// parameters keep the moments either quantized (mq/vq codes + ms/vs block scales,
// when quantization is on) or in fp (mf/vf); non-matrix parameters always use fp.
type qgaloreState struct {
	proj [][]float64 // fp projection, used only by the GaLore-collapse path
	// Q-GaLore stores P as packed unsigned INT4 with affine scale/zero per block.
	projQ                  []byte
	projScales, projZeros  []float64
	projRows, projCols     int
	left                   bool // true: project rows (m≤n), R=PᵀG; false: project cols, R=GP
	n                      int  // logical moment length (= reduced size, or Numel for non-matrix)
	gap, svdCount          int
	prevProjectionVector   []float64
	projectionSimilarities []float64
	// Q-GaLore's authoritative INT8 weight store. The public Tensor remains a
	// dequantized execution mirror because tensor currently has no integer dtype.
	wq           []uint8
	weightScales []float64
	weightZeros  []float64

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
// amortizes the SVD but lets the subspace go stale. In paper mode this is the INITIAL
// per-parameter gap; stable subspaces then multiply their own cadence adaptively.
// Default 200 (Q-GaLore paper and official reference implementation).
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

// WithQGaLoreQuantBits sets the moment precision and the §V16 collapse mode.
//
// In plain terms: 8 (the default) is real Q-GaLore — the moments are stored in one
// byte each and the remaining paper mechanisms are enabled. SPECIAL VALUES 0 and 64
// turn the complete Q-GaLore path off: fp moments/projection/weights plus fixed SVD
// cadence become bit-for-bit nn.GaLore. Other non-zero values currently select INT8.
// Default 8 (paper Figure 1 and official reference).
func WithQGaLoreQuantBits(bits int) QGaLoreOption { return func(g *QGaLore) { g.QuantBits = bits } }

// WithQGaLoreBlockSize sets the block length for moment, projection, and weight
// quantization.
//
// In plain terms: the quantizer picks one scale per block of this many moment
// entries, so an outlier only coarsens its own block, not the whole tensor. Boundary
// behavior — small blocks give the finest accuracy but store more per-block scales
// (shrinking the memory win); large blocks store fewer scales but let one big entry
// coarsen a wide span; ignored entirely when quantization is off. Default 256
// (Q-GaLore §3.1 and official reference).
func WithQGaLoreBlockSize(bs int) QGaLoreOption {
	return func(g *QGaLore) { g.BlockSize = bs }
}

// WithQGaLoreWeightQuantization controls the paper's persistent INT8 weight path.
// Enabled by default; false is an ablation that keeps full-precision weights while
// retaining quantized moments/projections. QuantBits 0 or 64 disables it regardless.
func WithQGaLoreWeightQuantization(enabled bool) QGaLoreOption {
	return func(g *QGaLore) { g.WeightQuant = enabled }
}

// WithQGaLoreProjectionQuantization controls packed INT4 storage of the SVD
// projection. Enabled by default; false stores P in fp64 as an ablation.
func WithQGaLoreProjectionQuantization(enabled bool) QGaLoreOption {
	return func(g *QGaLore) { g.ProjectionQuant = enabled }
}

// WithQGaLoreAdaptiveRefresh controls the paper's layer-local lazy SVD cadence.
// Enabled by default. False retains the fixed Gap cadence used by plain GaLore.
func WithQGaLoreAdaptiveRefresh(enabled bool) QGaLoreOption {
	return func(g *QGaLore) { g.AdaptiveRefresh = enabled }
}

// WithQGaLoreAdaptiveConfig sets the lazy-refresh threshold, gap multiplier, and
// similarity-window length. Defaults are 0.4, 2, and 5 from the paper's reference
// implementation. A larger threshold delays slowing the SVD cadence; a multiplier
// <=1 or queue <=0 is normalized back to the defaults.
func WithQGaLoreAdaptiveConfig(threshold float64, multiplier, queue int) QGaLoreOption {
	return func(g *QGaLore) {
		g.CosThreshold, g.GapMultiplier, g.QueueSize = threshold, multiplier, queue
	}
}

// WithQGaLoreSeed sets the deterministic random stream used for unbiased stochastic
// rounding of INT8 weights. Same seed and inputs produce the same trajectory.
func WithQGaLoreSeed(seed uint64) QGaLoreOption { return func(g *QGaLore) { g.Seed = seed } }

// NewQGaLore builds a Q-GaLore optimizer over params with learning rate lr and the
// paper/reference defaults: rank 128, scale 0.25, initial gap 200, Adam β 0.9/0.999,
// block size 256, INT8 moments+weights, INT4 projection, adaptive 0.4/2/5 cadence,
// and deterministic stochastic-rounding seed 1. QuantBits(0) is exactly nn.GaLore.
func NewQGaLore(params []*tensor.Tensor, lr float64, opts ...QGaLoreOption) *QGaLore {
	g := &QGaLore{
		Params: params, LR: lr, Rank: 128, Scale: 0.25, Gap: 200,
		Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, QuantBits: 8, BlockSize: 256,
		WeightQuant: true, ProjectionQuant: true, AdaptiveRefresh: true,
		CosThreshold: 0.4, GapMultiplier: 2, QueueSize: 5, Seed: 1,
	}
	for _, o := range opts {
		o(g)
	}
	if g.BlockSize <= 0 {
		g.BlockSize = 256
	}
	if g.GapMultiplier <= 1 {
		g.GapMultiplier = 2
	}
	if g.QueueSize <= 0 {
		g.QueueSize = 5
	}
	g.st = make([]*qgaloreState, len(params))
	g.rng = rand.New(rand.NewPCG(g.Seed, g.Seed^0x9e3779b97f4a7c15))
	return g
}

// quantized reports whether the complete paper mode is enabled. False is the
// nn.GaLore-collapse mode (QuantBits 0 or 64).
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

// quantizeAffine implements the paper/reference block-wise uniform quantizer. The
// code range is unsigned [0,2^bits-1], with one scale and zero point per block.
// Stochastic mode chooses ceil(y) with probability frac(y), otherwise floor(y), so
// the pre-clamp rounded value is unbiased: E[round(y)] = y.
func quantizeAffine(x []float64, blockSize, bits int, stochastic bool, random func() float64) ([]uint8, []float64, []float64) {
	maxCode := (1 << bits) - 1
	codes := make([]uint8, len(x))
	scales := make([]float64, numBlocks(len(x), blockSize))
	zeros := make([]float64, len(scales))
	for b := range scales {
		lo, hi := b*blockSize, min((b+1)*blockSize, len(x))
		mn, mx := x[lo], x[lo]
		for i := lo + 1; i < hi; i++ {
			mn, mx = fmath.Min(mn, x[i]), fmath.Max(mx, x[i])
		}
		scale := (mx - mn) / float64(maxCode)
		if scale < 1e-5 { // official Q-GaLore quantizer clamp
			scale = 1e-5
		}
		zero := math.Round(-mn / scale)
		//perfscan:ignore PS3077,PS3082 per-block scalar clamp, low trip vs per-element loop | per-block scalar min/max, negligible share
		zero = math.Max(0, math.Min(float64(maxCode), zero))
		scales[b], zeros[b] = scale, zero
		//perfscan:ignore PS5001 divide feeds round/uint8 quantize; memory-bound, not bit-safe
		for i := lo; i < hi; i++ {
			y := x[i] / scale
			var rounded float64
			if stochastic {
				down := math.Floor(y)
				rounded = down
				if random() < y-down {
					rounded = down + 1
				}
			} else {
				rounded = math.Round(y)
			}
			q := fmath.Max(0, fmath.Min(float64(maxCode), rounded+zero))
			codes[i] = uint8(q)
		}
	}
	return codes, scales, zeros
}

func dequantizeAffine(codes []uint8, scales, zeros []float64, blockSize int, out []float64) {
	for i, q := range codes {
		b := i / blockSize
		out[i] = (float64(q) - zeros[b]) * scales[b]
	}
}

func packInt4(codes []uint8) []byte {
	packed := make([]byte, (len(codes)+1)/2)
	for i, q := range codes {
		if i&1 == 0 {
			packed[i/2] = q & 0x0f
		} else {
			packed[i/2] |= (q & 0x0f) << 4
		}
	}
	return packed
}

//perfscan:ignore PS3033 alloc-class unpackInt4 buffer; resource-only, marginal wallclock
func unpackInt4(packed []byte, n int) []uint8 {
	codes := make([]uint8, n)
	for i := range n {
		if i&1 == 0 {
			codes[i] = packed[i/2] & 0x0f
		} else {
			codes[i] = packed[i/2] >> 4
		}
	}
	return codes
}

func flattenMatrix(m [][]float64) []float64 {
	if len(m) == 0 {
		return nil
	}
	out := make([]float64, 0, len(m)*len(m[0]))
	for _, row := range m {
		out = append(out, row...)
	}
	return out
}

func (st *qgaloreState) setProjection(proj [][]float64, blockSize int, quant bool) {
	st.projRows, st.projCols = len(proj), len(proj[0])
	if !quant {
		st.proj = proj
		st.projQ, st.projScales, st.projZeros = nil, nil, nil
		return
	}
	flat := flattenMatrix(proj)
	codes, scales, zeros := quantizeAffine(flat, blockSize, 4, false, nil)
	st.proj = nil
	st.projQ, st.projScales, st.projZeros = packInt4(codes), scales, zeros
}

//perfscan:ignore PS3033 alloc-class projection() reconstruct; resource-only allocs
func (st *qgaloreState) projection(blockSize int) [][]float64 {
	if st.proj != nil {
		return st.proj
	}
	n := st.projRows * st.projCols
	flat := make([]float64, n)
	dequantizeAffine(unpackInt4(st.projQ, n), st.projScales, st.projZeros, blockSize, flat)
	proj := make([][]float64, st.projRows)
	for i := range proj {
		proj[i] = flat[i*st.projCols : (i+1)*st.projCols]
	}
	return proj
}

func projectionCosine(a, b []float64) float64 {
	var dot, aa, bb float64
	//perfscan:ignore PS3010 3-accum cosine reassoc not bit-identical; refresh-only periodic
	for i := range a {
		dot += a[i] * b[i]
		aa += a[i] * a[i]
		bb += b[i] * b[i]
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / math.Sqrt(aa*bb)
}

func (g *QGaLore) refreshProjection(st *qgaloreState, mat [][]float64, paperMode bool) {
	proj, left := galoreProjection(mat, g.Rank)
	st.left = left
	st.svdCount++
	if paperMode && g.AdaptiveRefresh && st.prevProjectionVector != nil {
		sim := projectionCosine(st.prevProjectionVector, proj[0])
		if len(st.projectionSimilarities) == g.QueueSize {
			copy(st.projectionSimilarities, st.projectionSimilarities[1:])
			st.projectionSimilarities = st.projectionSimilarities[:g.QueueSize-1]
		}
		st.projectionSimilarities = append(st.projectionSimilarities, sim)
		if len(st.projectionSimilarities) == g.QueueSize {
			var sum float64
			for _, v := range st.projectionSimilarities {
				sum += v
			}
			if sum/float64(g.QueueSize) >= g.CosThreshold {
				st.gap *= g.GapMultiplier
			}
		}
	}
	st.prevProjectionVector = append(st.prevProjectionVector[:0], proj[0]...)
	st.setProjection(proj, g.BlockSize, paperMode && g.ProjectionQuant)
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
// qgalorePending carries a matrix parameter's not-yet-applied weight update from the parallel
// compute phase to the serial write phase. next holds the fp weights before the (rng-consuming)
// weight quantization; a nil next means the parameter did all its own writing in the compute phase
// (non-matrix Adam) or had no gradient.
type qgalorePending struct {
	p          *tensor.Tensor
	st         *qgaloreState
	next       []float64
	rows, cols int
}

// Step applies one QGaLore update. Parameters with a nil gradient are skipped; matrix parameters
// take the projected low-rank path with quantized optimizer state, and non-matrix parameters fall
// back to plain Adam.
func (g *QGaLore) Step(grad GradFn) error {
	g.t++
	b1c := 1 - math.Pow(g.Beta1, float64(g.t))
	b2c := 1 - math.Pow(g.Beta2, float64(g.t))
	quant := g.quantized()
	paperMode := quant
	bs := g.BlockSize

	// GradFn is the caller's and is not documented thread-safe, so gather every gradient SERIALLY
	// first, validating shapes here where an error can still be returned.
	grads := make([]*tensor.Tensor, len(g.Params))
	work := 0
	for pi, p := range g.Params {
		gt := grad(p)
		if gt == nil {
			continue
		}
		if !gt.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: QGaLore grad shape %v != param %v", gt.Shape(), p.Shape())
		}
		grads[pi] = gt
		work += p.Numel()
	}

	// Phase A (parallel): the rng-free per-parameter compute — the expensive part, in particular the
	// SVD subspace refresh (galoreProjection→SVD) plus project-down / subspace-Adam / moment
	// requant / project-up. Each parameter touches only its own qgaloreState slot, its own p, and
	// self-contained scratch, so it fans out BIT-IDENTICALLY. Matrix parameters defer their
	// rng-CONSUMING weight quantization (the ONLY use of the shared g.rng) to Phase B, so the seeded
	// stochastic-rounding draw ORDER is preserved exactly; non-matrix Adam is rng-free and writes p
	// directly here.
	pend := make([]qgalorePending, len(g.Params))
	parallelRows(len(g.Params), max(work/max(len(g.Params), 1), 1), func(plo, phi int) {
		for pi := plo; pi < phi; pi++ {
			if grads[pi] != nil {
				pend[pi] = g.stepParamCompute(pi, g.Params[pi], grads[pi], b1c, b2c, quant, paperMode, bs)
			}
		}
	})

	// Phase B (serial, in parameter order): apply the rng-dependent weight quantization and write
	// each matrix parameter back. Consuming g.rng here — in the SAME parameter order as the original
	// serial loop — makes the seeded draw sequence, and thus every weight, bit-identical.
	for pi := range g.Params {
		pd := pend[pi]
		if pd.next == nil {
			continue
		}
		if paperMode && g.WeightQuant {
			pd.st.wq, pd.st.weightScales, pd.st.weightZeros = quantizeAffine(pd.next, bs, 8, true, g.rng.Float64)
			dequantizeAffine(pd.st.wq, pd.st.weightScales, pd.st.weightZeros, bs, pd.next)
		} else {
			pd.st.wq, pd.st.weightScales, pd.st.weightZeros = nil, nil, nil
		}
		for i := range pd.rows {
			for j := range pd.cols {
				pd.p.SetF64(pd.next[i*pd.cols+j], i, j)
			}
		}
	}
	return nil
}

// stepParamCompute runs the rng-free part of one parameter's update. Non-matrix parameters do plain
// Adam and write p directly (returning a nil-next pending); matrix parameters run the projection /
// subspace Adam / project-back and return the fp weights in pending.next for Phase B to quantize
// (with the shared rng) and write. It reads/writes only g.st[pi], p, and self-contained scratch.
func (g *QGaLore) stepParamCompute(pi int, p, gt *tensor.Tensor, b1c, b2c float64, quant, paperMode bool, bs int) qgalorePending {
	st := g.st[pi]

	// Non-matrix parameters: plain fp Adam on the full gradient, exactly as GaLore
	// (the projection — and the quantization — apply only to 2-D weights).
	if p.Ndim() != 2 {
		if st == nil {
			st = &qgaloreState{mf: make([]float64, p.Numel()), vf: make([]float64, p.Numel())}
			g.st[pi] = st
		}
		//perfscan:ignore PS5001 b1c divide on memory-bound Adam loop; PS5001 declines
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := gt.AtF64(idx...)
			st.mf[i] = g.Beta1*st.mf[i] + (1-g.Beta1)*gv
			st.vf[i] = g.Beta2*st.vf[i] + (1-g.Beta2)*gv*gv
			upd := (st.mf[i] / b1c) / (math.Sqrt(st.vf[i]/b2c) + g.Eps)
			p.SetF64(p.AtF64(idx...)-g.LR*upd, idx...)
		}
		return qgalorePending{}
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
		st = &qgaloreState{left: left, n: n, gap: g.Gap}
		st.prevProjectionVector = append([]float64(nil), proj[0]...)
		st.svdCount = 1
		st.setProjection(proj, bs, paperMode && g.ProjectionQuant)
		if quant {
			nb := numBlocks(n, bs)
			st.mq, st.vq = make([]int8, n), make([]int8, n)
			st.ms, st.vs = make([]float64, nb), make([]float64, nb)
		} else {
			st.mf, st.vf = make([]float64, n), make([]float64, n)
		}
		g.st[pi] = st
	} else if st.gap > 0 && g.t%st.gap == 0 {
		g.refreshProjection(st, mat, paperMode) // refresh subspace; keep moments
	}

	// project → dequantize moments → Adam in subspace → requantize → project back.
	proj := st.projection(bs)
	red := galoreProjectDown(mat, proj, st.left)
	mf, vf := st.mf, st.vf
	if quant {
		mf = make([]float64, st.n)
		vf = make([]float64, st.n)
		dequantizeBlockwise(st.mq, st.ms, bs, mf)
		dequantizeBlockwise(st.vq, st.vs, bs, vf)
	}
	upd := make([]float64, len(red))
	//perfscan:ignore PS5001 b1c divide memory-bound elementwise; speedup evaporates
	for i := range red {
		mf[i] = g.Beta1*mf[i] + (1-g.Beta1)*red[i]
		vf[i] = g.Beta2*vf[i] + (1-g.Beta2)*red[i]*red[i]
		upd[i] = (mf[i] / b1c) / (math.Sqrt(vf[i]/b2c) + g.Eps)
	}
	if quant {
		quantizeBlockwise(mf, bs, st.mq, st.ms)
		quantizeBlockwise(vf, bs, st.vq, st.vs)
	}
	n := galoreProjectUp(upd, proj, st.left, rows, cols)
	next := make([]float64, 0, rows*cols)
	for i := range rows {
		for j := range cols {
			//perfscan:ignore PS3016 row-hoist dominated by co-located AtF64/SetF64 dispatch
			next = append(next, p.AtF64(i, j)-g.LR*g.Scale*n[i][j])
		}
	}
	return qgalorePending{p: p, st: st, next: next, rows: rows, cols: cols}
}
