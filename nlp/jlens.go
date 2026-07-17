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
//	J_l = E[ ∂h_L(t') / ∂h_l(t) ],   summed over valid targets t' ≥ t,
//	                                 averaged over valid sources t and sequences,
//
// where h_L is the FINAL residual stream (after the last block, before the
// final norm), h_l the stream at layer l (index 0 = post-embedding), and the
// valid positions exclude the final position and an optional attention-sink
// prefix (see [FitJLens]). J_l transports a layer-l activation into the
// final-layer basis, where the model's own final norm + unembedding read it
// out as ranked vocabulary (the transported logit lens [JLens.Apply], §T812).
// By construction J[Layers] is the identity. Fit with [FitJLens], combine
// disjoint fits with [Merge], persist with [Save]/[LoadJLens], or import a
// reference-implementation artifact with [JLensFromPT].
type JLens struct {
	// J holds Layers+1 matrices, J[l] of shape [dim, dim] (f64), with
	// J[l][a][b] = E[∂h_L[a]/∂h_l[b]] — rows index the output (final-layer)
	// basis, columns the layer-l basis, so the transported readout is the
	// ordinary matrix-vector product J[l]·h. An entry may be nil for a layer
	// the lens was not fitted at (a [JLensFromPT] import has J[0] == nil:
	// the reference format never fits the post-embedding stream).
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
	skipFirst        int
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
// (0 = all). When capped, targets are an evenly spaced deterministic subset of
// the valid positions (always including the first and last valid one), which
// approximates the full target sum at proportionally reduced cost — the
// backward count per sequence is targets·dim, so this is the dominant cost
// knob. (The reference implementation instead batches all targets into one
// cotangent; the estimator is identical when uncapped, by linearity.)
func WithJLensMaxTargetsPerSeq(n int) FitJLensOption {
	return func(o *fitJLensOpts) { o.maxTargetsPerSeq = n }
}

// WithJLensSkipFirst excludes the first n positions of every sequence from the
// Jacobian average (both as targets and as sources), mirroring the reference
// implementation's skip_first: early positions act as attention sinks and have
// atypical residual statistics. The reference default is 16 (tuned for real
// models on 128-token web text); GoAI defaults to 0 so tiny-model fits keep
// every usable position — pass 16 to reproduce a reference fit exactly.
func WithJLensSkipFirst(n int) FitJLensOption {
	return func(o *fitJLensOpts) { o.skipFirst = n }
}

// FitJLens fits a Jacobian lens to model on a token corpus (§T811, §R250).
//
// For every sequence it runs one forward pass through an autograd tape with
// residual capture (§T810), then extracts exact rows of every per-layer
// Jacobian by reverse-mode VJPs: seeding the cotangent e_j at target row t'
// of the final residual h_L and reading the tape gradient at each captured
// h_l gives row j of ∂h_L(t')/∂h_l(t) for ALL source positions t ≤ t' at
// once. Contributions are summed over targets t' ≥ t, averaged over source
// positions within a sequence, then averaged over sequences — the §R250
// estimator, with no finite-difference or sampling approximation.
//
// Position mask (reference-implementation parity, pinned by the §T812 golden
// fixtures): both the target sum and the source average run over the VALID
// positions only — the final position is always excluded (it has no
// next-token target), and [WithJLensSkipFirst] drops leading attention-sink
// positions exactly like the reference's skip_first. A sequence with no valid
// positions (len ≤ skipFirst+1) is an error; the reference logs and skips
// such prompts instead — GoAI is explicit so a silently empty fit cannot
// happen.
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
	if o.skipFirst < 0 {
		return nil, fmt.Errorf("nlp: FitJLens: negative skipFirst %d", o.skipFirst)
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
		if len(tokens) <= o.skipFirst+1 {
			return nil, fmt.Errorf("nlp: FitJLens: corpus sequence %d too short (%d tokens, need > skipFirst+1 = %d)", si, len(tokens), o.skipFirst+1)
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

		// Valid positions (reference mask): skipFirst … seq-2 inclusive. The
		// same set serves as targets AND as the source-average support.
		valid := make([]int, 0, seq-1-o.skipFirst)
		for p := o.skipFirst; p < seq-1; p++ {
			valid = append(valid, p)
		}

		// Per-sequence accumulator: sum over valid targets t' ≥ t of the exact
		// VJP rows, read at valid source positions t.
		seqAcc := make([][]float64, L+1)
		for l := range seqAcc {
			seqAcc[l] = make([]float64, d*d)
		}
		for _, ti := range jlensTargets(len(valid), o.maxTargetsPerSeq) {
			tp := valid[ti]
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
					// Row j of ∂h_L(t')/∂h_l(t) for every valid source t ≤ t'
					// (rows t > t' are exactly zero under causal attention).
					row := seqAcc[l][j*d : (j+1)*d]
					for _, t := range valid {
						if t > tp {
							break
						}
						gt := g[t*d : (t+1)*d]
						for i, v := range gt {
							row[i] += v
						}
					}
				}
			}
		}
		// Average over valid source positions, then accumulate the per-sequence mean.
		inv := 1 / float64(len(valid))
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

// jlensTargets returns the indices (into the valid-position list of length n)
// to use as targets: all of them when max ≤ 0 or n ≤ max, else an evenly
// spaced deterministic subset including the first and last index.
func jlensTargets(n, max int) []int {
	if max <= 0 || n <= max {
		ts := make([]int, n)
		for i := range ts {
			ts[i] = i
		}
		return ts
	}
	if max == 1 {
		return []int{n - 1}
	}
	ts := make([]int, 0, max)
	prev := -1
	for k := 0; k < max; k++ {
		t := k * (n - 1) / (max - 1)
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
		if jl.J[l] == nil && other.J[l] == nil {
			continue // layer unfitted in both (e.g. J[0] of imported lenses)
		}
		if jl.J[l] == nil || other.J[l] == nil {
			return fmt.Errorf("nlp: JLens.Merge: layer %d fitted in only one lens", l)
		}
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
// weight). LoadJLens round-trips it bit-exactly. A lens with unfitted (nil)
// layers — a [JLensFromPT] import — cannot be saved in the GoAI-native
// format, which requires every layer; keep the original .pt artifact instead.
func (jl *JLens) Save(path string) error {
	ts := make(map[string]*tensor.Tensor, len(jl.J))
	for l, t := range jl.J {
		if t == nil {
			return fmt.Errorf("nlp: JLens.Save: layer %d unfitted (imported reference lens?) — the native format requires every layer", l)
		}
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

// JLensFromPT imports a reference-implementation lens artifact (a
// JacobianLens.save torch .pt file, loaded through the safe format/pytorch
// unpickler) so Anthropic/Neuronpedia-fitted lenses are directly usable
// (§T811, §R250).
//
// ACTUAL ARTIFACT FORMAT (pinned by the §T812 golden fixture
// testdata/jlens_ref.pt, correcting the preliminary §T811 naming assumption):
// the reference saves
//
//	{"J": {layer: Tensor[d, d], …}, "n_prompts": int,
//	 "source_layers": [ints], "d_model": int}
//
// with INTEGER dict keys in "J" (flattened to "J.0", "J.1", … by the loader)
// and fp16 matrices by default. Orientation is row-major J[out, in] with the
// readout J·h — identical to GoAI's convention, taken verbatim (no
// transpose; the reference's own tiny-model test pins J_l = I + W for a
// last-block h + W·h, which the golden fixture reconfirms).
//
// LAYER INDEXING: the reference indexes by BLOCK OUTPUT — its layer l is the
// residual after block l, i.e. GoAI layer l+1 (GoAI layer 0 is the
// post-embedding stream, which the reference never fits). Its default fit
// covers source layers 0..L-2 against target layer L-1 (the last block), so
// an artifact holding K contiguous matrices maps to a GoAI lens with
// Layers = K+1 where J[0] is nil (unfitted in the reference format — Apply
// rejects it), J[l] for l ∈ 1..K is the artifact's J_{l-1}, and J[Layers] is
// the identity (synthesized; exact by definition, the target layer's Jacobian
// to itself).
//
// n_prompts and source_layers are non-tensor pickle values the safe loader
// does not surface: Weight is set to 1, so callers merging imported lenses
// must supply real weights themselves, and non-contiguous source_layers
// selections (a non-default reference fit) are rejected via the contiguity
// check rather than silently misaligned.
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
// artifact key ("J.3" → 3, "3" → 3).
var jlensLayerKey = regexp.MustCompile(`(?:^|\.)(\d+)$`)

// jlensFromTensors assembles a JLens from a loaded .pt tensor map under the
// JLensFromPT format (see there): square [d,d] tensors with trailing integer
// indices are the reference's block-output-indexed Jacobians; everything else
// is ignored. Split out for unit-testing the mapping without a torch fixture.
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
		return nil, fmt.Errorf("no square integer-indexed Jacobian tensors found (reference format: {\"J\": {0: …, 1: …}} → keys \"J.N\")")
	}
	idxs := make([]int, 0, len(byIdx))
	for i := range byIdx {
		idxs = append(idxs, i)
	}
	sort.Ints(idxs)
	for k, i := range idxs {
		if i != k {
			return nil, fmt.Errorf("layer indices %v not a contiguous 0..N run (non-default source_layers fit?)", idxs)
		}
	}
	// K reference matrices at block outputs 0..K-1, target = block K → GoAI
	// Layers = K+1: J[0] unfitted (nil), J[l] = ref J_{l-1}, J[Layers] = I.
	k := len(idxs)
	jl := &JLens{Dim: dim, Layers: k + 1, Weight: 1}
	jl.J = make([]*tensor.Tensor, k+2)
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
		jl.J[i+1] = t
	}
	eye := tensor.New(tensor.F64, tensor.Shape{dim, dim})
	for i := 0; i < dim; i++ {
		eye.SetF64(1, i, i)
	}
	jl.J[k+1] = eye
	return jl, nil
}
