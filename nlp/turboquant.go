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
