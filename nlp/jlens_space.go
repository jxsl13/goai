package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// J-space decomposition (§T813, §R250). The J-space paper (Anthropic 2026,
// "Verbalizable Representations Form a Global Workspace in Language Models")
// defines J-space as the set of activations expressible as SPARSE NONNEGATIVE
// combinations of at most k (≤ 25) J-lens vocabulary directions. The lens
// readout at layer l is JL_l(h) = W_U · norm(J_l · h); linearising it (norm
// preserves direction up to positive scaling), the logit of token v is
//
//	logit_v ∝ w_v · (J_l h) = (J_lᵀ w_v) · h,
//
// so the J-lens DIRECTION of token v in layer-l residual space is
//
//	d_v = J_lᵀ w_v / ‖J_lᵀ w_v‖    (w_v = column v of the model's W_U).
//
// [JSpaceDecompose] greedily writes an activation as h ≈ Σ c_i d_{v_i} with
// c_i > 0 (nonnegative matching pursuit) and reports the empirical OCCUPANCY:
// directions are added only while each one's reconstruction gain beats the
// gain a random unit direction achieves on the same residual — the §R250
// random-direction control. The public reference implementation
// (github.com/anthropics/jacobian-lens) ships no decomposition code, so this
// is a faithful transcription of the §R250 recipe, verified by property tests
// (tier-2 §V16): exact recovery of planted sparse combinations, near-zero
// occupancy on isotropic noise, and monotone reconstruction on real residuals
// of the §T812 golden model.

// jspaceOpts collects the JSpaceDecompose knobs (functional options).
type jspaceOpts struct {
	ctrlDirs int
	seed     uint64
}

// JSpaceOption configures [JSpaceDecompose].
type JSpaceOption func(*jspaceOpts)

// WithJSpaceControlDirections sets how many random unit directions the
// occupancy control samples per pursuit step (default: one per dictionary
// atom, i.e. the vocabulary size — the control argmax then faces exactly the
// same selection bias as the dictionary argmax, so on isotropic noise the
// two gains are identically distributed and occupancy collapses to ≈ 0).
// Larger values make the control stricter.
func WithJSpaceControlDirections(n int) JSpaceOption {
	return func(o *jspaceOpts) { o.ctrlDirs = n }
}

// WithJSpaceSeed sets the PCG seed of the control's random directions
// (default 1). The decomposition is fully deterministic for a fixed seed.
func WithJSpaceSeed(seed uint64) JSpaceOption {
	return func(o *jspaceOpts) { o.seed = seed }
}

// JSpaceComponent is one direction of a J-space decomposition: the vocabulary
// token whose J-lens direction was selected, its accumulated nonnegative
// coefficient, and the total squared-residual reduction it contributed.
type JSpaceComponent struct {
	Token int     // vocabulary id of the selected direction
	Coef  float64 // nonnegative pursuit coefficient (> 0)
	Gain  float64 // reduction of ‖residual‖² attributed to this direction
}

// JSpaceDecomposition is the result of [JSpaceDecompose]: the sparse
// nonnegative expansion h ≈ Σ Components[i].Coef · d_{Components[i].Token}
// plus the occupancy estimate and reconstruction diagnostics.
type JSpaceDecomposition struct {
	// Components lists the selected directions in greedy selection order
	// (re-selections of an already-chosen direction are merged in place, so
	// entries are distinct tokens).
	Components []JSpaceComponent
	// Occupancy is the empirical J-space occupancy (§R250): the number of
	// distinct directions whose reconstruction gain beat the
	// random-direction control before the pursuit stopped. Equal to
	// len(Components); when the pursuit ran into the k cap the true
	// occupancy may be larger.
	Occupancy int
	// InputNorm is ‖h‖, ResidualNorm is ‖h − Σ c_i d_i‖ after the final
	// accepted component (ResidualNorm == InputNorm when nothing was
	// accepted).
	InputNorm    float64
	ResidualNorm float64 // ‖residual‖ after the final accepted component (== InputNorm if none accepted)
	// ResidualCurve holds ‖residual‖ after each accepted pursuit step
	// (strictly decreasing; one entry per accepted step, which can exceed
	// len(Components) when a direction was re-selected and merged).
	ResidualCurve []float64
	// Reconstruction is the dense reconstruction Σ c_i d_i in layer-l
	// residual space (length Dim) — h minus the final residual.
	Reconstruction []float64
}

// JSpaceDecompose expands a layer-l residual activation h in the J-lens
// vocabulary directions of lens at that layer (§T813, §R250): greedy
// nonnegative matching pursuit over the dictionary d_v = normalize(J_lᵀ w_v),
// with w_v the model's unembedding column for token v.
//
// Each step selects the direction with the largest positive correlation
// c = ⟨d_v, r⟩ to the current residual r, subtracts the projection c·d_v
// (the plain matching-pursuit update — atoms are unit vectors, so the step
// removes exactly c² of ‖r‖²), and repeats. The pursuit stops at k steps
// (k ≤ 0 selects the paper's default 25), when no direction correlates
// positively, or when the step's gain c² no longer beats the best gain among
// random unit directions on the same residual (the §R250 occupancy control;
// see [WithJSpaceControlDirections] and [WithJSpaceSeed]).
//
// h holds one residual-stream activation: shape [dim] or [1, dim] (a row as
// captured by a [ResidualCapture] or sliced from one). layer follows the GoAI
// residual convention of [JLens.Apply]; unfitted layers (nil J) are rejected.
// model supplies the unembedding matrix W_U [dim, vocab] — [GPT] (Head) and
// [Llama] (Out) are recognised directly, and any other [LensReadoutModel] can
// participate by exposing an `UnembeddingMatrix() *tensor.Tensor` method.
//
// The decomposition runs in pure f64 Go (dictionary vocab × dim, pursuit
// k · vocab · dim) — no backend dispatch and no external solver.
func JSpaceDecompose(lens *JLens, model LensReadoutModel, h *tensor.Tensor, layer, k int, opts ...JSpaceOption) (*JSpaceDecomposition, error) {
	o := jspaceOpts{seed: 1}
	for _, opt := range opts {
		opt(&o)
	}
	if k <= 0 {
		k = 25 // the paper's default sparsity cap (§R250: k ≤ 25)
	}
	if layer < 0 || layer >= len(lens.J) {
		return nil, fmt.Errorf("nlp: JSpaceDecompose: layer %d outside 0..%d", layer, len(lens.J)-1)
	}
	J := lens.J[layer]
	if J == nil {
		return nil, fmt.Errorf("nlp: JSpaceDecompose: layer %d not fitted in this lens", layer)
	}
	dim := lens.Dim
	r, err := jspaceVector(h, dim)
	if err != nil {
		return nil, err
	}
	W, err := unembeddingMatrix(model)
	if err != nil {
		return nil, err
	}
	ws := W.Shape()
	if len(ws) != 2 || ws[0] != dim {
		return nil, fmt.Errorf("nlp: JSpaceDecompose: unembedding shape %v, want [%d vocab]", ws, dim)
	}
	vocab := ws[1]

	// Dictionary: d_v[b] = Σ_a J[a,b]·W[a,v], ℓ2-normalised. Zero directions
	// (a token the transported readout cannot express) are skipped.
	atoms := make([][]float64, 0, vocab)
	atomTok := make([]int, 0, vocab)
	for v := 0; v < vocab; v++ {
		d := make([]float64, dim)
		var n2 float64
		for b := 0; b < dim; b++ {
			var acc float64
			for a := 0; a < dim; a++ {
				acc += J.AtF64(a, b) * W.AtF64(a, v)
			}
			d[b] = acc
			n2 += acc * acc
		}
		if n2 == 0 {
			continue
		}
		inv := 1 / math.Sqrt(n2)
		for b := range d {
			d[b] *= inv
		}
		atoms = append(atoms, d)
		atomTok = append(atomTok, v)
	}
	if len(atoms) == 0 {
		return nil, fmt.Errorf("nlp: JSpaceDecompose: dictionary empty (J_l·W_U ≡ 0 at layer %d)", layer)
	}
	ctrl := o.ctrlDirs
	if ctrl <= 0 {
		ctrl = len(atoms) // matched control: same selection bias as the dictionary argmax
	}
	rng := rand.New(rand.NewPCG(o.seed, 0x6a2d7370616365)) // "j-space"

	dec := &JSpaceDecomposition{InputNorm: norm2(r)}
	compIdx := map[int]int{} // token → index into dec.Components
	g := make([]float64, dim)
	for step := 0; step < k; step++ {
		// Greedy atom: max positive correlation with the residual.
		best, bestC := -1, 0.0
		for i, d := range atoms {
			var c float64
			for b, rb := range r {
				c += d[b] * rb
			}
			if c > bestC {
				best, bestC = i, c
			}
		}
		if best < 0 {
			break // no direction correlates positively — nonnegativity exhausted
		}
		gain := bestC * bestC

		// Random-direction control (§R250): the best gain random unit
		// directions achieve on the same residual. A direction below this
		// bar explains no more than chance — stop, occupancy reached.
		var ctrlGain float64
		for i := 0; i < ctrl; i++ {
			var n2, c float64
			for b := range g {
				g[b] = rng.NormFloat64()
				n2 += g[b] * g[b]
			}
			if n2 == 0 {
				continue
			}
			for b, rb := range r {
				c += g[b] * rb
			}
			if c > 0 {
				if cg := c * c / n2; cg > ctrlGain {
					ctrlGain = cg
				}
			}
		}
		if gain <= ctrlGain {
			break
		}

		// Accept: subtract the projection, merge re-selections.
		for b, db := range atoms[best] {
			r[b] -= bestC * db
		}
		tok := atomTok[best]
		if i, ok := compIdx[tok]; ok {
			dec.Components[i].Coef += bestC
			dec.Components[i].Gain += gain
		} else {
			compIdx[tok] = len(dec.Components)
			dec.Components = append(dec.Components, JSpaceComponent{Token: tok, Coef: bestC, Gain: gain})
		}
		dec.ResidualCurve = append(dec.ResidualCurve, norm2(r))
	}
	dec.Occupancy = len(dec.Components)
	dec.ResidualNorm = dec.InputNorm
	if n := len(dec.ResidualCurve); n > 0 {
		dec.ResidualNorm = dec.ResidualCurve[n-1]
	}
	dec.Reconstruction = make([]float64, dim)
	for b := range dec.Reconstruction {
		dec.Reconstruction[b] = h1AtF64(h, b) - r[b]
	}
	return dec, nil
}

// jspaceVector copies a [dim] or [1, dim] activation into a fresh f64 slice.
func jspaceVector(h *tensor.Tensor, dim int) ([]float64, error) {
	s := h.Shape()
	ok := (len(s) == 1 && s[0] == dim) || (len(s) == 2 && s[0] == 1 && s[1] == dim)
	if !ok {
		return nil, fmt.Errorf("nlp: JSpaceDecompose: activation shape %v, want [%d] or [1 %d]", s, dim, dim)
	}
	r := make([]float64, dim)
	for b := range r {
		r[b] = h1AtF64(h, b)
	}
	return r, nil
}

// h1AtF64 reads element b of a [dim] or [1, dim] tensor.
func h1AtF64(h *tensor.Tensor, b int) float64 {
	if len(h.Shape()) == 1 {
		return h.AtF64(b)
	}
	return h.AtF64(0, b)
}

// norm2 returns the euclidean norm of v.
func norm2(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

// unembeddingMatrix returns a model's raw output projection W_U [dim, vocab]
// — the matrix whose columns are the vocabulary read-out directions in the
// final residual basis. [GPT] and [Llama] are recognised directly; any other
// model can opt in by exposing UnembeddingMatrix() *tensor.Tensor.
func unembeddingMatrix(model LensReadoutModel) (*tensor.Tensor, error) {
	if u, ok := model.(interface{ UnembeddingMatrix() *tensor.Tensor }); ok {
		return u.UnembeddingMatrix(), nil
	}
	switch m := model.(type) {
	case *GPT:
		return m.Head, nil
	case *Llama:
		return m.Out, nil
	}
	return nil, fmt.Errorf("nlp: JSpaceDecompose: %T exposes no unembedding matrix (add UnembeddingMatrix() *tensor.Tensor)", model)
}
