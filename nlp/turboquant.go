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
