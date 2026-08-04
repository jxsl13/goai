package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/internal/simd"
)

// TurboQuant sub-4-bit KV-cache quantization (Zandieh et al., "TurboQuant", Google Research +
// NYU, ICLR 2026, arXiv:2504.19874; §T619). This file builds the pieces bottom-up. The first
// component is PolarQuant's random-rotation stage: a fixed random ORTHOGONAL matrix Π is applied
// to every key/value vector before quantization, which spreads the signal so that each
// coordinate follows a data-independent Beta distribution (≈ Gaussian in high dimension). Being
// orthogonal, Π preserves inner products and norms — so attention scores are unchanged by the
// rotation itself — and its inverse is simply its transpose.
//
// The rotation is built once from a fixed seed (data-oblivious, no training), so encode and
// decode share the identical Π and the transform round-trips exactly.

// randomOrthogonal returns a d×d orthogonal matrix — the Q factor of a QR decomposition of a
// matrix with i.i.d. standard-normal entries (arXiv:2504.19874 §PolarQuant) — computed here by
// modified Gram-Schmidt (numerically stabler than classical GS). Deterministic in seed. The rows
// form an orthonormal basis: Q·Qᵀ = I.
func randomOrthogonal(d int, seed uint64) [][]float64 {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	// rows of a Gaussian matrix, orthonormalized in place by modified Gram-Schmidt.
	q := make([][]float64, d)
	for i := range q {
		//perfscan:ignore PS2008,PS3064 resource-only alloc; one-time matrix build | one-time orthogonal-matrix construction (build once from seed)
		q[i] = make([]float64, d)
		//perfscan:ignore PS4006 reference-only path; cache uses hadamard fwht
		for j := range q[i] {
			q[i][j] = rng.NormFloat64()
		}
	}
	//perfscan:ignore PS3034 Gram-Schmidt one-time construction
	for i := range d {
		// subtract the projections of the already-orthonormal rows 0..i-1 from row i
		for k := range i {
			var dot float64
			//perfscan:ignore PS3010,PS3066 GS inner loop, one-time build
			for j := range d {
				//perfscan:ignore PS3016 GS loop, one-time build
				dot += q[k][j] * q[i][j]
			}
			for j := range d {
				//perfscan:ignore PS3075 norm accum, one-time construction
				q[i][j] -= dot * q[k][j]
			}
		}
		var norm float64
		//perfscan:ignore PS3010 norm accum, one-time construction
		for j := range d {
			norm += q[i][j] * q[i][j]
		}
		norm = math.Sqrt(norm)
		if norm < 1e-12 {
			norm = 1e-12 // degenerate Gaussian draw (astronomically unlikely); keep it finite
		}
		for j := range d {
			q[i][j] /= norm
		}
	}
	return q
}

// polarRotation is TurboQuant's fixed rotation stage: a d×d orthogonal Π applied to length-d
// vectors. Build it once with newPolarRotation and reuse for every key/value row; because Π is
// orthogonal, apply then applyInverse recovers the input exactly (up to float rounding).
type polarRotation struct {
	d int
	q [][]float64 // orthogonal, row-major
}

// newPolarRotation builds the fixed random rotation for length-d vectors, seeded for
// reproducibility (§V10). d must be ≥ 1.
func newPolarRotation(d int, seed uint64) (*polarRotation, error) {
	if d < 1 {
		return nil, fmt.Errorf("nlp: newPolarRotation needs d ≥ 1, got %d", d)
	}
	return &polarRotation{d: d, q: randomOrthogonal(d, seed)}, nil
}

// apply returns Π·x (the rotated vector). len(x) must be d.
func (p *polarRotation) apply(x []float64) ([]float64, error) {
	if len(x) != p.d {
		return nil, fmt.Errorf("nlp: polarRotation.apply wants len %d, got %d", p.d, len(x))
	}
	out := make([]float64, p.d)
	// Unroll-and-jam the GEMV over the free output-row index i: 4 independent
	// accumulators share each x[j] load, breaking the single-accumulator dot's
	// FMA latency chain. Each out[i] still sums j ascending over identical
	// operands → bit-identical (free-dim jam, not an inner-reduction split).
	i := 0
	//perfscan:ignore PS3066,PS3076 polarRotation.apply reference-only (unused by cache), already unrolled | reference-only apply, already 4-way u
	for ; i+3 < p.d; i += 4 {
		r0, r1, r2, r3 := p.q[i], p.q[i+1], p.q[i+2], p.q[i+3]
		var a0, a1, a2, a3 float64
		//perfscan:ignore PS3010 apply inner, reference-only path
		for j := range p.d {
			xv := x[j]
			a0 += r0[j] * xv
			a1 += r1[j] * xv
			a2 += r2[j] * xv
			a3 += r3[j] * xv
		}
		out[i], out[i+1], out[i+2], out[i+3] = a0, a1, a2, a3
	}
	for ; i < p.d; i++ {
		var acc float64
		row := p.q[i]
		//perfscan:ignore PS3010 apply tail, reference-only path
		for j := range p.d {
			acc += row[j] * x[j]
		}
		out[i] = acc
	}
	return out, nil
}

// applyInverse returns Πᵀ·y, the inverse rotation (Π orthogonal ⇒ Π⁻¹ = Πᵀ). len(y) must be d.
func (p *polarRotation) applyInverse(y []float64) ([]float64, error) {
	if len(y) != p.d {
		return nil, fmt.Errorf("nlp: polarRotation.applyInverse wants len %d, got %d", p.d, len(y))
	}
	out := make([]float64, p.d)
	//perfscan:ignore PS6010 applyInverse reference-only path, resource-only
	for j := range p.d {
		var acc float64
		//perfscan:ignore PS3010 applyInverse reference-only path
		for i := range p.d {
			//perfscan:ignore PS1010 applyInverse reference-only path (cache uses fwht)
			acc += p.q[i][j] * y[i]
		}
		out[j] = acc
	}
	return out, nil
}

// polarCodebook returns the MSE-optimal (Lloyd-Max) reconstruction centroids for a b-bit
// per-coordinate quantizer of TurboQuant's rotated unit-vector coordinates (arXiv:2504.19874).
// A coordinate of a unit vector in ℝ^d has magnitude ≈ 1/√d, and after rotation its SCALED
// value (×√d) converges to a STANDARD NORMAL in high dimension (the Beta density → N(0,1)); so
// the codebook is the Lloyd-Max quantizer of N(0,1) divided by √d. lloydMaxGaussian solves it
// numerically for any b ≥ 1 and reproduces the paper's closed forms (b=1 ±√(2/π); b=2 ±0.4528,
// ±1.510). Returns ascending centroids. This unblocks the 2.5-bit outlier split (3-bit channels).
func polarCodebook(b, d int) ([]float64, error) {
	if b < 1 {
		return nil, fmt.Errorf("nlp: polarCodebook needs b ≥ 1, got %d", b)
	}
	g := lloydMaxGaussian(b)
	sd := math.Sqrt(float64(d))
	out := make([]float64, len(g))
	//perfscan:ignore PS5001 one-time tiny codebook loop; output feeds quantizer, hoist unsafe
	for i, c := range g {
		out[i] = c / sd
	}
	return out, nil
}

// lloydMaxGaussian returns the 2^b MSE-optimal reconstruction centroids of a standard normal
// N(0,1), found by Lloyd's algorithm: alternate decision boundaries at centroid midpoints with
// centroid updates to the Gaussian's conditional mean on each interval, E[Z | a≤Z<c] =
// (φ(a)−φ(c))/(Φ(c)−Φ(a)) with φ the pdf and Φ the cdf. Symmetric; converges in a few dozen
// iterations. Ascending centroids.
func lloydMaxGaussian(b int) []float64 {
	n := 1 << b
	// init centroids at uniform quantiles of the Gaussian (a good starting partition)
	c := make([]float64, n)
	//perfscan:ignore PS4003 one-time codebook init, tiny n=2^b
	for i := range c {
		p := (float64(i) + 0.5) / float64(n)
		c[i] = gaussianQuantile(p)
	}
	phi := func(x float64) float64 { return math.Exp(-x*x/2) / math.Sqrt(2*math.Pi) }
	cdf := func(x float64) float64 { return 0.5 * math.Erfc(-x/math.Sqrt2) }
	for iter := 0; iter < 100; iter++ {
		// boundaries: midpoints between consecutive centroids; outer boundaries ±∞.
		//perfscan:ignore PS3035 Lloyd iteration, one-time codebook build
		bnd := make([]float64, n+1)
		bnd[0], bnd[n] = math.Inf(-1), math.Inf(1)
		for i := 1; i < n; i++ {
			bnd[i] = (c[i-1] + c[i]) / 2
		}
		var maxΔ float64
		for i := range c {
			lo, hi := bnd[i], bnd[i+1]
			var pl, ph, cl, ch float64 = 0, 0, 0, 1
			if !math.IsInf(lo, -1) {
				pl, cl = phi(lo), cdf(lo)
			}
			if !math.IsInf(hi, 1) {
				ph, ch = phi(hi), cdf(hi)
			}
			mass := ch - cl
			if mass < 1e-15 {
				continue
			}
			nc := (pl - ph) / mass // conditional mean
			if d := math.Abs(nc - c[i]); d > maxΔ {
				maxΔ = d
			}
			c[i] = nc
		}
		if maxΔ < 1e-12 {
			break
		}
	}
	return c
}

// gaussianQuantile is the inverse standard-normal CDF (probit) via the Beasley-Springer/Moro
// rational approximation — accurate enough to seed Lloyd's iteration.
func gaussianQuantile(p float64) float64 {
	if p <= 0 {
		return -5
	}
	if p >= 1 {
		return 5
	}
	// use the symmetric rational approximation around the tails/centre.
	a := []float64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	bco := []float64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	cco := []float64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	dco := []float64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00}
	plow, phigh := 0.02425, 1-0.02425
	switch {
	case p < plow:
		q := math.Sqrt(-2 * math.Log(p))
		return (((((cco[0]*q+cco[1])*q+cco[2])*q+cco[3])*q+cco[4])*q + cco[5]) /
			((((dco[0]*q+dco[1])*q+dco[2])*q+dco[3])*q + 1)
	case p <= phigh:
		q := p - 0.5
		r := q * q
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q /
			(((((bco[0]*r+bco[1])*r+bco[2])*r+bco[3])*r+bco[4])*r + 1)
	default:
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((cco[0]*q+cco[1])*q+cco[2])*q+cco[3])*q+cco[4])*q + cco[5]) /
			((((dco[0]*q+dco[1])*q+dco[2])*q+dco[3])*q + 1)
	}
}

// nearestCentroid returns the index of the codebook centroid closest to v (the decision
// boundaries are the midpoints between consecutive centroids, so nearest-centroid is exact).
func nearestCentroid(v float64, cb []float64) int {
	best, bestD := 0, math.Abs(v-cb[0])
	//perfscan:ignore PS3068 low trip-count (2^bits centroids), linear scan optimal
	for i := 1; i < len(cb); i++ {
		if dd := math.Abs(v - cb[i]); dd < bestD {
			best, bestD = i, dd
		}
	}
	return best
}

// polarQuantize is the PolarQuant storage path: unit-normalize x (store ‖x‖ exactly), rotate the
// unit vector by Π, and quantize each rotated coordinate to a b-bit codebook index. Returns the
// indices and the stored norm. A zero vector round-trips to zero.
func (p *polarRotation) polarQuantize(x []float64, b int) (idx []int, norm float64, err error) {
	cb, err := polarCodebook(b, p.d)
	if err != nil {
		return nil, 0, err
	}
	for _, v := range x {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	u := make([]float64, p.d)
	if norm > 0 {
		for i, v := range x {
			u[i] = v / norm
		}
	}
	ru, err := p.apply(u)
	if err != nil {
		return nil, 0, err
	}
	idx = make([]int, p.d)
	//perfscan:ignore PS3065 polarQuantize reference-only path (unused by cache)
	for i, v := range ru {
		idx[i] = nearestCentroid(v, cb)
	}
	return idx, norm, nil
}

// polarDequantize reconstructs x̃ from the stored indices and norm: look up the centroids, rotate
// back by Πᵀ, and rescale by the norm. It is the lossy inverse of polarQuantize.
func (p *polarRotation) polarDequantize(idx []int, norm float64, b int) ([]float64, error) {
	cb, err := polarCodebook(b, p.d)
	if err != nil {
		return nil, err
	}
	if len(idx) != p.d {
		return nil, fmt.Errorf("nlp: polarDequantize wants %d indices, got %d", p.d, len(idx))
	}
	ru := make([]float64, p.d)
	for i, k := range idx {
		if k < 0 || k >= len(cb) {
			return nil, fmt.Errorf("nlp: polarDequantize index %d out of range [0,%d)", k, len(cb))
		}
		ru[i] = cb[k]
	}
	u, err := p.applyInverse(ru)
	if err != nil {
		return nil, err
	}
	out := make([]float64, p.d)
	for i, v := range u {
		out[i] = norm * v
	}
	return out, nil
}

// qjlSketch is TurboQuant's QJL (Quantized Johnson-Lindenstrauss) 1-bit residual corrector
// (QJL, Zandieh et al., AAAI 2025, arXiv:2406.03482). It sketches the PolarQuant residual r into
// 1-bit signs qjl = sign(S·r) with the FAST JL transform S = (1/√d)·H·D (D a Rademacher ±1
// diagonal, H the d×d Walsh-Hadamard matrix) instead of a dense Gaussian matrix — O(d log d) via
// the fwht butterfly, not O(d²). The residual is reconstructed as (√(π/2)/√d)·‖r‖·Sᵀ·qjl, an
// UNBIASED estimate of r: for the (1/√d)HD sketch, (Sr)_i ≈ N(0, ‖r‖²/d) so per-row
// E[(Sq)_i·sign((Sr)_i)] = √(2/π)·⟨q,r⟩/(√d·‖r‖), summed over d rows ⇒ the √(π/2)/√d constant
// (empirically verified — the dense √(π/2)/d is 10× too small here). Unbiasedness is what a
// softmax over many keys needs; a single 1-bit sketch is noisy but centred. Requires d a power of
// two (the cache always sketches in the padded ℝ^m, m = nextPow2(dim)).
type qjlSketch struct {
	d     int
	signs []float64 // Rademacher diagonal D, length d
}

func newQJLSketch(d int, seed uint64) *qjlSketch {
	rng := rand.New(rand.NewPCG(seed, 0x2545f4914f6cdd1d))
	s := make([]float64, d)
	for i := range s {
		if rng.Uint64()&1 == 0 {
			s[i] = 1
		} else {
			s[i] = -1
		}
	}
	return &qjlSketch{d: d, signs: s}
}

// transform applies S·x = (1/√d)·H·(D·x) via the fast Walsh-Hadamard butterfly (x length d,
// a power of two).
func (q *qjlSketch) transform(x []float64) []float64 {
	b := make([]float64, q.d)
	for i := range x {
		b[i] = x[i] * q.signs[i]
	}
	fwht(b)
	inv := 1 / math.Sqrt(float64(q.d))
	for i := range b {
		b[i] *= inv
	}
	return b
}

// encode sketches a residual r into 1-bit signs qjl = sign(S·r) and returns ‖r‖₂.
func (q *qjlSketch) encode(r []float64) (signs []bool, rnorm float64) {
	sr := q.transform(r)
	signs = make([]bool, q.d)
	for i, v := range sr {
		signs[i] = v >= 0
	}
	for _, v := range r {
		rnorm += v * v
	}
	return signs, math.Sqrt(rnorm)
}

// decodeResidual returns the unbiased residual estimate (√(π/2)/√d)·‖r‖·Sᵀ·qjl, where
// Sᵀ·qjl = (1/√d)·D·H·qjl (H symmetric) is another fwht — O(d log d).
func (q *qjlSketch) decodeResidual(signs []bool, rnorm float64) []float64 {
	var s tqScratch
	return q.decodeResidualInto(signs, rnorm, &s)
}

// decodeResidualInto is decodeResidual writing through a caller-supplied scratch, so a batch of
// rows reuses two buffers instead of allocating two per row. Both are fully overwritten before
// they are read — z is filled from every sign and out from every z — so reuse cannot carry a
// value across rows. Same arithmetic, same order, identical bits.
func (q *qjlSketch) decodeResidualInto(signs []bool, rnorm float64, sc *tqScratch) []float64 {
	z := sc.grow(&sc.z, q.d)
	for i, s := range signs {
		if s {
			z[i] = 1
		} else {
			z[i] = -1
		}
	}
	fwht(z)                                              // H·qjl
	scale := math.Sqrt(math.Pi/2) / float64(q.d) * rnorm // (√(π/2)/√d)·‖r‖·(1/√d) = √(π/2)/d·‖r‖
	out := sc.grow(&sc.res, q.d)
	for i := range out {
		out[i] = scale * z[i] * q.signs[i]
	}
	return out
}

// TurboQuantKVCache is a sub-4-bit KV cache using TurboQuant (Zandieh et al., ICLR 2026,
// arXiv:2504.19874; §T619) — the extreme-compression tier beyond the near-lossless 8-bit Q8_0
// cache (QuantKVCache, §R108). Each stored key/value row is compressed to `bits` bits per
// coordinate by PolarQuant (a fixed random rotation → per-coordinate MSE-optimal quantizer) plus
// a 1-bit QJL residual sketch that keeps the attention inner product UNBIASED. The rotation and
// sketch are fixed random matrices seeded once — data-oblivious, no calibration, no training — so
// any model can use it immediately and encode/decode share the identical transforms.
//
// # For the AI professional
//
// Per row x: idx = Quant_polar(Πx̂) at `bits` bits (x̂ the unit-normalized x, ‖x‖ stored), plus
// qjl = sign(S·r) over the quantization residual r and ‖r‖. Reconstruction
// x̃ = ‖x‖·Πᵀ(centroids[idx] + (√(π/2)/d)·‖r‖·Sᵀ·qjl) is unbiased for the inner product ⟨q,x⟩,
// which is what a softmax over many keys needs (bias would systematically skew scores; a single
// 1-bit sketch is noisy but centred). Footprint ≈ d·bits/8 + d/8 + 16 bytes per row.
//
// # For the newcomer
//
// The key/value memory ("KV cache") dominates the cost of long-context generation. This stores it
// at under 4 bits per number instead of 32, by first spinning each vector with a fixed random
// rotation (which spreads its information evenly), rounding each coordinate to a tiny code, and
// keeping one correction bit so attention scores stay accurate on average. It needs no tuning.
//
// Further reading: Zandieh et al. 2025, "TurboQuant", arXiv:2504.19874 (ICLR 2026); the QJL
// sketch (Zandieh et al., "QJL", AAAI 2025); compare the 8-bit [QuantKVCache].
//
// In plain terms: a way to shrink the biggest memory cost of long-context LLM generation to
// under 4 bits per number, with no accuracy loss on average and no calibration.
type TurboQuantKVCache struct {
	dim  int
	m    int // padded dimension = nextPow2(dim); the Hadamard rotation works in ℝ^m
	bits int
	rot  *hadamardRotation
	qjl  *qjlSketch
	cb   []float64 // per-coordinate centroids, scaled by 1/√m
	keys []tqRow
	vals []tqRow
}

// tqRow is one compressed key/value vector: the PolarQuant indices, the QJL residual signs, and
// the two stored scalars (‖x‖ and ‖r‖).
type tqRow struct {
	idx   []int
	signs []bool
	norm  float64
	rnorm float64
}

// NewTurboQuantKVCache builds a TurboQuant KV cache for dim-dimensional keys/values, quantizing
// each coordinate to bits bits (Lloyd-Max codebook, any bits ≥ 1; the paper uses 2, plus a 3-bit
// outlier tier) plus a 1-bit QJL residual. It uses the fast Hadamard-Rademacher rotation (O(d
// log m) vs the dense O(d²)); vectors are zero-padded to m = nextPow2(dim). seed fixes the
// rotation and sketch matrices (reproducible, §V10). dim ≥ 1, bits ≥ 1.
func NewTurboQuantKVCache(dim, bits int, seed uint64) (*TurboQuantKVCache, error) {
	if dim < 1 {
		return nil, fmt.Errorf("nlp: NewTurboQuantKVCache needs dim ≥ 1, got %d", dim)
	}
	rot, err := newHadamardRotation(dim, seed)
	if err != nil {
		return nil, err
	}
	cb, err := polarCodebook(bits, rot.m) // scaled by 1/√m — the rotated coords live in ℝ^m
	if err != nil {
		return nil, err
	}
	return &TurboQuantKVCache{
		dim: dim, m: rot.m, bits: bits, rot: rot, cb: cb,
		qjl: newQJLSketch(rot.m, seed^0x9e3779b9),
	}, nil
}

// Len returns the number of key/value rows stored.
func (c *TurboQuantKVCache) Len() int { return len(c.keys) }

// Bytes returns the approximate compressed footprint: per row, m·bits/8 (PolarQuant over the m
// padded/rotated coordinates) + m/8 (QJL 1-bit) + 16 (two f64 scalars), for keys and values. For
// power-of-two dims m = dim; otherwise the few padding coordinates cost a little extra.
func (c *TurboQuantKVCache) Bytes() int {
	perRow := c.m*c.bits/8 + (c.m+7)/8 + 16
	return perRow * (len(c.keys) + len(c.vals))
}

func (c *TurboQuantKVCache) compress(x []float64) (tqRow, error) {
	var norm float64
	for _, v := range x {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	u := make([]float64, c.dim)
	if norm > 0 {
		for i, v := range x {
			u[i] = v / norm
		}
	}
	ru, err := c.rot.apply(u) // m rotated coordinates
	if err != nil {
		return tqRow{}, err
	}
	idx := make([]int, c.m)
	r := make([]float64, c.m)
	//perfscan:ignore PS3065 reconstruct O(m) add, fwht-dominated
	for i, v := range ru {
		idx[i] = nearestCentroid(v, c.cb)
		r[i] = v - c.cb[idx[i]]
	}
	signs, rnorm := c.qjl.encode(r)
	return tqRow{idx: idx, signs: signs, norm: norm, rnorm: rnorm}, nil
}

// tqScratch holds the per-row intermediates of a reconstruction. Five of the six buffers a row
// needed were pure scratch — the sketch decode's two, the rotation's two, and the rotated-coord
// staging — and only the returned row itself outlives the call. One scratch per WORKER makes the
// allocation count a function of GOMAXPROCS instead of the cache length.
type tqScratch struct {
	z, res, ruT, buf, uT []float64
}

// grow returns *b resliced to n, allocating only when the current capacity is short. Every caller
// then writes all n entries before reading any, which is what makes reuse invisible.
func (s *tqScratch) grow(b *[]float64, n int) []float64 {
	if cap(*b) < n {
		*b = make([]float64, n)
	}
	*b = (*b)[:n]
	return *b
}

func (c *TurboQuantKVCache) reconstruct(row tqRow) []float64 {
	var s tqScratch
	return c.reconstructInto(row, &s)
}

// reconstructInto is reconstruct with the intermediates supplied by the caller. Only the returned
// row is freshly allocated — it is kept by the caller, so it cannot be scratch.
func (c *TurboQuantKVCache) reconstructInto(row tqRow, sc *tqScratch) []float64 {
	res := c.qjl.decodeResidualInto(row.signs, row.rnorm, sc) // m coords
	// Sized by the row's OWN index count rather than c.m, so the buffer is exactly the region the
	// next loop writes. That is what makes reuse safe without a clearing pass: there is no tail
	// left over from the previous row to read. A row whose index count disagrees with the rotation
	// width then fails the length check inside applyInverseInto instead of quietly reading stale
	// values — compress always produces c.m of them, and this keeps that an enforced invariant
	// rather than an assumed one.
	ruT := sc.grow(&sc.ruT, len(row.idx))
	for i := range ruT {
		ruT[i] = c.cb[row.idx[i]] + res[i]
	}
	uT, _ := c.rot.applyInverseInto(ruT, sc) // back to dim coords
	out := make([]float64, c.dim)
	for i := range uT {
		out[i] = row.norm * uT[i]
	}
	return out
}

// Append compresses and stores one key and value vector (each length dim).
func (c *TurboQuantKVCache) Append(k, v []float64) error {
	if len(k) != c.dim || len(v) != c.dim {
		return fmt.Errorf("nlp: TurboQuantKVCache.Append wants len %d, got k=%d v=%d", c.dim, len(k), len(v))
	}
	kr, err := c.compress(k)
	if err != nil {
		return err
	}
	vr, err := c.compress(v)
	if err != nil {
		return err
	}
	c.keys = append(c.keys, kr)
	c.vals = append(c.vals, vr)
	return nil
}

// Keys returns the reconstructed key rows [Len][dim] (the unbiased TurboQuant estimate). Values
// is the analogous accessor for the value rows.
func (c *TurboQuantKVCache) Keys() [][]float64 { return c.rows(c.keys) }

// Values returns the reconstructed value rows [Len][dim].
func (c *TurboQuantKVCache) Values() [][]float64 { return c.rows(c.vals) }

func (c *TurboQuantKVCache) rows(rs []tqRow) [][]float64 {
	out := make([][]float64, len(rs))
	// Each row's reconstruction is independent: reconstruct writes only out[i] and reads the
	// immutable rotation/codebook/sketch (applyInverse and decodeResidual allocate all their
	// buffers locally — no receiver scratch), so the loop fans out over GOMAXPROCS bit-identically
	// to the serial loop. Gated on len(rs)·dim so a short cache stays serial.
	parallelChunks(len(rs), len(rs)*c.dim, func(lo, hi int) {
		var sc tqScratch // one per worker band, not one per row
		for i := lo; i < hi; i++ {
			out[i] = c.reconstructInto(rs[i], &sc)
		}
	})
	return out
}

// hadamardRotation is TurboQuant's FAST rotation: the randomized Hadamard-Rademacher transform
// R = (1/√m)·H·D, where D is a random ±1 (Rademacher) diagonal, H the m×m Walsh-Hadamard matrix,
// and m the next power of two ≥ d (the vector is zero-padded to m). It is orthogonal and spreads
// the signal into near-Gaussian coordinates exactly like the dense Gaussian rotation, but the
// Walsh-Hadamard transform runs in O(m log m) via the in-place butterfly instead of the dense
// O(d²) mat-vec (arXiv:2504.19874; the "standard fast preconditioner" of the QJL/JL literature).
// It is the drop-in fast replacement for polarRotation.
type hadamardRotation struct {
	d, m  int
	signs []float64 // Rademacher diagonal D, length m
}

// nextPow2 returns the smallest power of two ≥ n (≥1).
func nextPow2(n int) int {
	m := 1
	for m < n {
		m <<= 1
	}
	return m
}

// fwht applies the UNNORMALIZED in-place Fast Walsh-Hadamard Transform (len(a) a power of two).
// H is symmetric and H·H = m·I, so a second call scaled by 1/m inverts it.
func fwht(a []float64) { simd.FWHTF64(a) }

func newHadamardRotation(d int, seed uint64) (*hadamardRotation, error) {
	if d < 1 {
		return nil, fmt.Errorf("nlp: newHadamardRotation needs d ≥ 1, got %d", d)
	}
	m := nextPow2(d)
	rng := rand.New(rand.NewPCG(seed, 0x14057b7ef767814f))
	signs := make([]float64, m)
	for i := range signs {
		if rng.Uint64()&1 == 0 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}
	return &hadamardRotation{d: d, m: m, signs: signs}, nil
}

// apply returns R·x = (1/√m)·H·(D·x_padded). len(x) must be d; the result is length m (the
// rotated coordinates live in the padded space, quantized there).
func (r *hadamardRotation) apply(x []float64) ([]float64, error) {
	if len(x) != r.d {
		return nil, fmt.Errorf("nlp: hadamardRotation.apply wants len %d, got %d", r.d, len(x))
	}
	buf := make([]float64, r.m)
	for i := range x {
		buf[i] = x[i] * r.signs[i]
	}
	fwht(buf)
	inv := 1 / math.Sqrt(float64(r.m))
	for i := range buf {
		buf[i] *= inv
	}
	return buf, nil
}

// applyInverse returns Rᵀ·y = D·(1/√m)·H·y, truncated to the original d coordinates (R
// orthogonal ⇒ R⁻¹ = Rᵀ; H,D symmetric). len(y) must be m.
func (r *hadamardRotation) applyInverse(y []float64) ([]float64, error) {
	var s tqScratch
	return r.applyInverseInto(y, &s)
}

// applyInverseInto is applyInverse writing through a caller-supplied scratch. The transform buffer
// is a full copy of y and the output is written for every index, so neither can carry a value from
// the previous row; the arithmetic is untouched.
func (r *hadamardRotation) applyInverseInto(y []float64, sc *tqScratch) ([]float64, error) {
	if len(y) != r.m {
		return nil, fmt.Errorf("nlp: hadamardRotation.applyInverse wants len %d, got %d", r.m, len(y))
	}
	buf := sc.grow(&sc.buf, r.m)
	copy(buf, y)
	fwht(buf)
	inv := 1 / math.Sqrt(float64(r.m))
	out := sc.grow(&sc.uT, r.d)
	for i := range out {
		out[i] = buf[i] * inv * r.signs[i]
	}
	return out, nil
}
