package nlp

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/pytorch"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// ResidualCapture observes the residual stream of a decoder forward pass
// (§T810). It is invoked once per residual snapshot with the layer index —
// 0 for the post-embedding input h_0, l ∈ 1..Layers for the residual AFTER
// block l (PRE final norm) — and the LIVE [seq, dim] tensor the forward
// consumes. Observers must not mutate h (test-only injectors may, on an eager
// backend, at their own risk).
type ResidualCapture func(layer int, h *tensor.Tensor)

// ResidualModel is any decoder exposing per-layer residual-stream taps
// (§T810): GPT and Llama (and therefore every Llama-family config: Mistral,
// Qwen2/3, Granite). The returned tensor is the post-final-norm hidden state;
// a nil capture must be byte-identical to the plain forward.
type ResidualModel interface {
	ForwardResiduals(ctx *backend.Context, tokens []int, capture ResidualCapture) (*tensor.Tensor, error)
}

// JLens is a Jacobian lens (Anthropic 2026, "Verbalizable Representations Form
// a Global Workspace in Language Models"; ref-impl github.com/anthropics/
// jacobian-lens, §R250): one matrix per residual-stream layer,
//
//	J_l = E[ ∂h_L(t') / ∂h_l(t) ],   summed over targets t' ≥ t,
//	                                 averaged over sources t and sequences,
//
// where h_L is the FINAL residual stream (after the last block, before the
// final norm) and h_l the stream at layer l (index 0 = post-embedding). J_l
// transports a layer-l activation into the final-layer basis, where the
// model's own final norm + unembedding read it out as ranked vocabulary
// (the transported logit lens, §T812). By construction J[Layers] is the
// identity. Fit with [FitJLens], combine disjoint fits with [Merge], persist
// with [Save]/[LoadJLens], or import a reference-implementation artifact with
// [JLensFromPT].
type JLens struct {
	// J holds Layers+1 matrices, J[l] of shape [dim, dim] (f64), with
	// J[l][a][b] = E[∂h_L[a]/∂h_l[b]] — rows index the output (final-layer)
	// basis, columns the layer-l basis, so the transported readout is the
	// ordinary matrix-vector product J[l]·h.
	J []*tensor.Tensor

	Dim    int     // residual-stream width
	Layers int     // transformer block count L; len(J) == Layers+1
	Arch   string  // model architecture tag ("gpt2", "llama", "" if unknown)
	Weight float64 // total fit weight (sequences accumulated); Merge's mixing mass
}

// fitJLensOpts collects the FitJLens knobs (functional options).
type fitJLensOpts struct {
	maxSequences     int
	maxTargetsPerSeq int
}

// FitJLensOption configures FitJLens.
type FitJLensOption func(*fitJLensOpts)

// WithJLensMaxSequences caps the number of corpus sequences consumed (0 = all).
// The paper observes lens quality saturating after a few hundred generic
// sequences (§R250), so a cap is the cheap knob for large corpora.
func WithJLensMaxSequences(n int) FitJLensOption {
	return func(o *fitJLensOpts) { o.maxSequences = n }
}

// WithJLensMaxTargetsPerSeq caps the number of target positions t' per sequence
// (0 = all). When capped, targets are an evenly spaced deterministic subset
// (always including the first and last position), which approximates the
// full target sum at proportionally reduced cost — the backward count per
// sequence is targets·dim, so this is the dominant cost knob.
func WithJLensMaxTargetsPerSeq(n int) FitJLensOption {
	return func(o *fitJLensOpts) { o.maxTargetsPerSeq = n }
}

// FitJLens fits a Jacobian lens to model on a token corpus (§T811, §R250).
//
// For every sequence it runs one forward pass through an autograd tape with
// residual capture (§T810), then extracts exact rows of every per-layer
// Jacobian by reverse-mode VJPs: seeding the cotangent e_j at target row t'
// of the final residual h_L and reading the tape gradient at each captured
// h_l gives row j of ∂h_L(t')/∂h_l(t) for ALL source positions t ≤ t' at
// once. Contributions are summed over targets t' ≥ t, averaged over source
// positions within a sequence, then averaged over sequences — exactly the
// §R250 estimator, with no finite-difference or sampling approximation.
//
// Cost: the exact version performs targets·dim backward passes per sequence
// (each a full tape walk), which is affordable for the tiny models this is
// verified on (dim ≤ 64, layers ≤ 4, seq ≤ 32) and boundable via
// [WithJLensMaxSequences] / [WithJLensMaxTargetsPerSeq] for anything larger.
//
// The cotangent seed uses the tape's ones-seeded Backward with the
// inner-product trick: recording m = h_L ⊙ c for a one-hot c makes
// Backward(m)'s gradient at h_L exactly c, i.e. an arbitrary-cotangent VJP
// on a scalar-loss tape API.
func FitJLens(model ResidualModel, corpus [][]int, opts ...FitJLensOption) (*JLens, error) {
	var o fitJLensOpts
	for _, opt := range opts {
		opt(&o)
	}
	if len(corpus) == 0 {
		return nil, fmt.Errorf("nlp: FitJLens: empty corpus")
	}
	seqs := corpus
	if o.maxSequences > 0 && len(seqs) > o.maxSequences {
		seqs = seqs[:o.maxSequences]
	}

	var (
		acc    [][]float64 // [layers+1][dim*dim] running sum of per-sequence means
		dim    int
		layers int
		nSeq   float64
	)
	for si, tokens := range seqs {
		if len(tokens) == 0 {
			return nil, fmt.Errorf("nlp: FitJLens: corpus sequence %d is empty", si)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()

		var resids []*tensor.Tensor
		_, err := model.ForwardResiduals(ctx, tokens, func(layer int, h *tensor.Tensor) {
			if layer != len(resids) {
				panic(fmt.Sprintf("nlp: FitJLens: capture layer %d out of order (want %d)", layer, len(resids)))
			}
			resids = append(resids, h)
		})
		if err != nil {
			return nil, fmt.Errorf("nlp: FitJLens: sequence %d: %w", si, err)
		}
		if len(resids) < 2 {
			return nil, fmt.Errorf("nlp: FitJLens: capture yielded %d residual snapshots (need ≥ 2)", len(resids))
		}
		L := len(resids) - 1
		hL := resids[L]
		seq, d := hL.Shape()[0], hL.Shape()[1]
		if acc == nil {
			dim, layers = d, L
			acc = make([][]float64, L+1)
			for l := range acc {
				acc[l] = make([]float64, d*d)
			}
		} else if d != dim || L != layers {
			return nil, fmt.Errorf("nlp: FitJLens: sequence %d geometry [%d layers, dim %d] != [%d, %d]", si, L, d, layers, dim)
		}

		// Per-sequence accumulator: sum over targets t' ≥ t of the exact VJP rows.
		seqAcc := make([][]float64, L+1)
		for l := range seqAcc {
			seqAcc[l] = make([]float64, d*d)
		}
		for _, tp := range jlensTargets(seq, o.maxTargetsPerSeq) {
			for j := 0; j < d; j++ {
				// Cotangent c = e_{t',j}: Backward(h_L ⊙ c) seeds grad(h_L) = c.
				c := tensor.New(tensor.F64, tensor.Shape{seq, d})
				c.SetF64(1, tp, j)
				m, err := exec1(ctx, backend.OpMul, nil, hL, c)
				if err != nil {
					return nil, fmt.Errorf("nlp: FitJLens: cotangent mul: %w", err)
				}
				if err := tape.Backward(m); err != nil {
					return nil, fmt.Errorf("nlp: FitJLens: backward (seq %d, target %d, dir %d): %w", si, tp, j, err)
				}
				for l := 0; l <= L; l++ {
					G := tape.Grad(resids[l])
					if G == nil {
						continue // unreachable layer: zero contribution
					}
					g := G.Storage().F64()
					// Row j of ∂h_L(t')/∂h_l(t) for every source t ≤ t'
					// (rows t > t' are exactly zero under causal attention).
					row := seqAcc[l][j*d : (j+1)*d]
					for t := 0; t <= tp; t++ {
						gt := g[t*d : (t+1)*d]
						for i, v := range gt {
							row[i] += v
						}
					}
				}
			}
		}
		// Average over source positions, then accumulate the per-sequence mean.
		inv := 1 / float64(seq)
		for l := range seqAcc {
			for i, v := range seqAcc[l] {
				acc[l][i] += v * inv
			}
		}
		nSeq++
	}

	jl := &JLens{Dim: dim, Layers: layers, Arch: archTag(model), Weight: nSeq}
	for l := range acc {
		t := tensor.New(tensor.F64, tensor.Shape{dim, dim})
		dst := t.Storage().F64()
		for i, v := range acc[l] {
			dst[i] = v / nSeq
		}
		jl.J = append(jl.J, t)
	}
	return jl, nil
}

// jlensTargets returns the target positions t' for a sequence of length seq:
// all positions when max ≤ 0 or seq ≤ max, else an evenly spaced deterministic
// subset including position 0 and seq-1.
func jlensTargets(seq, max int) []int {
	if max <= 0 || seq <= max {
		ts := make([]int, seq)
		for i := range ts {
			ts[i] = i
		}
		return ts
	}
	if max == 1 {
		return []int{seq - 1}
	}
	ts := make([]int, 0, max)
	prev := -1
	for k := 0; k < max; k++ {
		t := k * (seq - 1) / (max - 1)
		if t != prev {
			ts = append(ts, t)
			prev = t
		}
	}
	return ts
}

// archTag maps a known model type to its artifact architecture tag.
func archTag(model ResidualModel) string {
	switch model.(type) {
	case *GPT:
		return "gpt2"
	case *Llama:
		return "llama"
	default:
		return ""
	}
}

// Merge folds another lens fitted on a DISJOINT corpus slice into this one by
// weighted averaging (§R250 merge(): the estimator is a plain expectation, so
// slice-wise fits combine exactly): J ← (Weight·J + weight·other.J)/(Weight+weight),
// then Weight += weight. Pass weight = other.Weight (its sequence count) to make
// merging half-corpus fits identical to the full-corpus fit. Geometry must match.
func (jl *JLens) Merge(other *JLens, weight float64) error {
	if other.Dim != jl.Dim || other.Layers != jl.Layers {
		return fmt.Errorf("nlp: JLens.Merge: geometry [%d layers, dim %d] != [%d, %d]", other.Layers, other.Dim, jl.Layers, jl.Dim)
	}
	if weight <= 0 || jl.Weight <= 0 {
		return fmt.Errorf("nlp: JLens.Merge: non-positive weight (self %g, other %g)", jl.Weight, weight)
	}
	total := jl.Weight + weight
	for l := range jl.J {
		a, b := jl.J[l].Storage().F64(), other.J[l].Storage().F64()
		for i := range a {
			a[i] = (jl.Weight*a[i] + weight*b[i]) / total
		}
	}
	jl.Weight = total
	return nil
}

// Save writes the lens to a GoAI-native safetensors file: one f64 tensor per
// layer named "jlens.layer.N" plus metadata (format, dim, layers, arch,
// weight). LoadJLens round-trips it bit-exactly.
func (jl *JLens) Save(path string) error {
	ts := make(map[string]*tensor.Tensor, len(jl.J))
	for l, t := range jl.J {
		ts[fmt.Sprintf("jlens.layer.%d", l)] = t
	}
	meta := map[string]string{
		"format": "goai-jlens",
		"dim":    strconv.Itoa(jl.Dim),
		"layers": strconv.Itoa(jl.Layers),
		"arch":   jl.Arch,
		"weight": strconv.FormatFloat(jl.Weight, 'g', -1, 64),
	}
	return safetensors.SaveFile(path, ts, meta)
}

// LoadJLens reads a lens saved by [JLens.Save].
func LoadJLens(path string) (*JLens, error) {
	ts, meta, err := safetensors.LoadFile(path)
	if err != nil {
		return nil, fmt.Errorf("nlp: LoadJLens: %w", err)
	}
	if f := meta["format"]; f != "goai-jlens" {
		return nil, fmt.Errorf("nlp: LoadJLens: metadata format %q != \"goai-jlens\"", f)
	}
	layers, err := strconv.Atoi(meta["layers"])
	if err != nil {
		return nil, fmt.Errorf("nlp: LoadJLens: bad layers metadata %q", meta["layers"])
	}
	dim, err := strconv.Atoi(meta["dim"])
	if err != nil {
		return nil, fmt.Errorf("nlp: LoadJLens: bad dim metadata %q", meta["dim"])
	}
	weight, err := strconv.ParseFloat(meta["weight"], 64)
	if err != nil {
		return nil, fmt.Errorf("nlp: LoadJLens: bad weight metadata %q", meta["weight"])
	}
	jl := &JLens{Dim: dim, Layers: layers, Arch: meta["arch"], Weight: weight}
	for l := 0; l <= layers; l++ {
		t := ts[fmt.Sprintf("jlens.layer.%d", l)]
		if t == nil {
			return nil, fmt.Errorf("nlp: LoadJLens: missing tensor jlens.layer.%d", l)
		}
		if len(t.Shape()) != 2 || t.Shape()[0] != dim || t.Shape()[1] != dim {
			return nil, fmt.Errorf("nlp: LoadJLens: jlens.layer.%d shape %v != [%d %d]", l, t.Shape(), dim, dim)
		}
		jl.J = append(jl.J, t)
	}
	return jl, nil
}

// JLensFromPT imports a reference-implementation lens artifact (a torch.save
// .pt file, loaded through the safe format/pytorch unpickler) so
// Anthropic/Neuronpedia-fitted lenses are directly usable (§T811, §R250).
//
// NAMING ASSUMPTION (documented, to be pinned by the §T812 golden fixtures):
// the ref-impl artifact is assumed to serialize the per-layer Jacobians as a
// dict/list whose flattened keys carry the layer index as their final
// '.'-separated component — e.g. "jacobians.0", "jacobians.1", … — each a
// square [dim, dim] matrix. The importer therefore collects EVERY square
// tensor whose key ends in an integer, requires them to form a contiguous
// 0..N index run of one consistent dim, and ignores everything else
// (unembedding refs, model metadata). Matrices are taken VERBATIM (no
// transpose); if the reference stores J^T, the §T812 golden verification
// will flip this mapping. Arch is left "" and Weight set to 1 (the artifact
// carries no fit-mass information this importer understands yet).
func JLensFromPT(path string) (*JLens, error) {
	ts, err := pytorch.LoadFile(path)
	if err != nil {
		return nil, fmt.Errorf("nlp: JLensFromPT: %w", err)
	}
	jl, err := jlensFromTensors(ts)
	if err != nil {
		return nil, fmt.Errorf("nlp: JLensFromPT %s: %w", path, err)
	}
	return jl, nil
}

// jlensLayerKey extracts a trailing integer layer index from a flattened
// state-dict key ("jacobians.3" → 3, "3" → 3).
var jlensLayerKey = regexp.MustCompile(`(?:^|\.)(\d+)$`)

// jlensFromTensors assembles a JLens from a loaded .pt tensor map under the
// JLensFromPT naming assumption (see there). Split out for unit-testing the
// mapping without a torch fixture.
func jlensFromTensors(ts map[string]*tensor.Tensor) (*JLens, error) {
	byIdx := map[int]*tensor.Tensor{}
	dim := 0
	for name, t := range ts {
		m := jlensLayerKey.FindStringSubmatch(name)
		if m == nil {
			continue
		}
		s := t.Shape()
		if len(s) != 2 || s[0] != s[1] {
			continue // not a square Jacobian (unembed ref, meta, …)
		}
		idx, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		if prev, dup := byIdx[idx]; dup {
			return nil, fmt.Errorf("layer index %d claimed by two square tensors (%v and %v)", idx, prev.Shape(), s)
		}
		if dim == 0 {
			dim = s[0]
		} else if s[0] != dim {
			return nil, fmt.Errorf("inconsistent Jacobian dims %d and %d", dim, s[0])
		}
		byIdx[idx] = t
	}
	if len(byIdx) == 0 {
		return nil, fmt.Errorf("no square integer-indexed Jacobian tensors found (naming assumption: keys like \"jacobians.N\")")
	}
	idxs := make([]int, 0, len(byIdx))
	for i := range byIdx {
		idxs = append(idxs, i)
	}
	sort.Ints(idxs)
	for k, i := range idxs {
		if i != k {
			return nil, fmt.Errorf("layer indices %v not a contiguous 0..N run", idxs)
		}
	}
	jl := &JLens{Dim: dim, Layers: len(idxs) - 1, Weight: 1}
	for _, i := range idxs {
		t := byIdx[i]
		if t.Dtype() != tensor.F64 {
			f := tensor.New(tensor.F64, t.Shape())
			dst := f.Storage().F64()
			for j := range dst {
				dst[j] = t.AtF64(j/dim, j%dim)
			}
			t = f
		}
		jl.J = append(jl.J, t)
	}
	return jl, nil
}
