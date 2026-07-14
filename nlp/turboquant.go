package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"
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
		q[i] = make([]float64, d)
		for j := range q[i] {
			q[i][j] = rng.NormFloat64()
		}
	}
	for i := range d {
		// subtract the projections of the already-orthonormal rows 0..i-1 from row i
		for k := range i {
			var dot float64
			for j := range d {
				dot += q[k][j] * q[i][j]
			}
			for j := range d {
				q[i][j] -= dot * q[k][j]
			}
		}
		var norm float64
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
	for i := range p.d {
		var acc float64
		row := p.q[i]
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
	for j := range p.d {
		var acc float64
		for i := range p.d {
			acc += p.q[i][j] * y[i]
		}
		out[j] = acc
	}
	return out, nil
}

// polarCodebook returns the MSE-optimal (Lloyd-Max) reconstruction centroids for a b-bit
// per-coordinate quantizer of TurboQuant's rotated unit-vector coordinates, whose marginal is the
// Beta density f_X(x)=Γ(d/2)/(π·Γ((d−1)/2))·(1−x²)^((d−3)/2) (arXiv:2504.19874). The paper gives
// the closed-form codebooks for b=1 (±√(2/(πd))) and b=2 (±0.453/√d, ±1.51/√d), scaled by 1/√d
// because a coordinate of a unit vector in ℝ^d has magnitude ≈ 1/√d. Higher b (numerically-solved
// centroids) is a follow-up. Returns ascending centroids.
func polarCodebook(b, d int) ([]float64, error) {
	sd := math.Sqrt(float64(d))
	switch b {
	case 1:
		c := math.Sqrt(2 / (math.Pi * float64(d)))
		return []float64{-c, c}, nil
	case 2:
		return []float64{-1.51 / sd, -0.453 / sd, 0.453 / sd, 1.51 / sd}, nil
	default:
		return nil, fmt.Errorf("nlp: polarCodebook supports b=1,2 (paper closed-forms), got b=%d", b)
	}
}

// nearestCentroid returns the index of the codebook centroid closest to v (the decision
// boundaries are the midpoints between consecutive centroids, so nearest-centroid is exact).
func nearestCentroid(v float64, cb []float64) int {
	best, bestD := 0, math.Abs(v-cb[0])
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
// (arXiv:2504.19874, Algorithm 2). A fixed random S∈ℝ^{d×d}, i.i.d N(0,1), sketches the PolarQuant
// residual r = ru − ruHat (rotated-unit space) into 1-bit signs qjl=sign(S·r). The residual is
// reconstructed as (√(π/2)/d)·‖r‖·Sᵀ·qjl, giving an UNBIASED estimate of r (E_S[·]=r, since per
// row E[S_i·sign(⟨S_i,r⟩)]=√(2/π)·r/‖r‖ and √(π/2)·√(2/π)=1) — so ⟨q, key⟩ becomes unbiased. This
// is variance-for-bias: a single 1-bit sketch is noisy, but the estimate is centred on the true
// value, which is what a softmax over many keys needs (bias would systematically distort scores).
type qjlSketch struct {
	d int
	s [][]float64 // [d,d] i.i.d N(0,1), fixed/seeded
}

func newQJLSketch(d int, seed uint64) *qjlSketch {
	rng := rand.New(rand.NewPCG(seed, 0x2545f4914f6cdd1d))
	s := make([][]float64, d)
	for i := range s {
		s[i] = make([]float64, d)
		for j := range s[i] {
			s[i][j] = rng.NormFloat64()
		}
	}
	return &qjlSketch{d: d, s: s}
}

// encode sketches a residual r into 1-bit signs qjl=sign(S·r) and returns ‖r‖₂.
func (q *qjlSketch) encode(r []float64) (signs []bool, rnorm float64) {
	signs = make([]bool, q.d)
	for i := range q.d {
		var dot float64
		row := q.s[i]
		for j := range q.d {
			dot += row[j] * r[j]
		}
		signs[i] = dot >= 0
	}
	for _, v := range r {
		rnorm += v * v
	}
	return signs, math.Sqrt(rnorm)
}

// decodeResidual returns the unbiased residual estimate (√(π/2)/d)·‖r‖·Sᵀ·qjl (rotated-unit
// space), to add to the PolarQuant reconstruction before rotating back.
func (q *qjlSketch) decodeResidual(signs []bool, rnorm float64) []float64 {
	scale := math.Sqrt(math.Pi/2) / float64(q.d) * rnorm
	out := make([]float64, q.d)
	for j := range q.d {
		var acc float64
		for i := range q.d {
			if signs[i] {
				acc += q.s[i][j]
			} else {
				acc -= q.s[i][j]
			}
		}
		out[j] = scale * acc
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
	bits int
	rot  *polarRotation
	qjl  *qjlSketch
	cb   []float64
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
// each coordinate to bits bits (1 or 2 — the paper's closed-form codebooks) plus a 1-bit QJL
// residual. seed fixes the rotation and sketch matrices (reproducible, §V10). dim ≥ 1, bits ∈ {1,2}.
func NewTurboQuantKVCache(dim, bits int, seed uint64) (*TurboQuantKVCache, error) {
	if dim < 1 {
		return nil, fmt.Errorf("nlp: NewTurboQuantKVCache needs dim ≥ 1, got %d", dim)
	}
	cb, err := polarCodebook(bits, dim)
	if err != nil {
		return nil, err
	}
	rot, err := newPolarRotation(dim, seed)
	if err != nil {
		return nil, err
	}
	return &TurboQuantKVCache{
		dim: dim, bits: bits, rot: rot, cb: cb,
		qjl: newQJLSketch(dim, seed^0x9e3779b9),
	}, nil
}

// Len returns the number of key/value rows stored.
func (c *TurboQuantKVCache) Len() int { return len(c.keys) }

// Bytes returns the approximate compressed footprint: per row, dim·bits/8 (PolarQuant) + dim/8
// (QJL 1-bit) + 16 (two f64 scalars), for keys and values.
func (c *TurboQuantKVCache) Bytes() int {
	perRow := c.dim*c.bits/8 + (c.dim+7)/8 + 16
	return perRow * (len(c.keys) + len(c.vals))
}

func (c *TurboQuantKVCache) compress(x []float64) (tqRow, error) {
	idx, norm, err := c.rot.polarQuantize(x, c.bits)
	if err != nil {
		return tqRow{}, err
	}
	// residual in rotated-unit space
	u := make([]float64, c.dim)
	if norm > 0 {
		for i, v := range x {
			u[i] = v / norm
		}
	}
	ru, err := c.rot.apply(u)
	if err != nil {
		return tqRow{}, err
	}
	r := make([]float64, c.dim)
	for i := range ru {
		r[i] = ru[i] - c.cb[idx[i]]
	}
	signs, rnorm := c.qjl.encode(r)
	return tqRow{idx: idx, signs: signs, norm: norm, rnorm: rnorm}, nil
}

func (c *TurboQuantKVCache) reconstruct(row tqRow) []float64 {
	res := c.qjl.decodeResidual(row.signs, row.rnorm)
	ruT := make([]float64, c.dim)
	for i := range row.idx {
		ruT[i] = c.cb[row.idx[i]] + res[i]
	}
	uT, _ := c.rot.applyInverse(ruT)
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
	for i, r := range rs {
		out[i] = c.reconstruct(r)
	}
	return out
}
