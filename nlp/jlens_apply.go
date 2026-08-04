package nlp

import (
	"fmt"
	"slices"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LensReadoutModel is the model access a transported readout needs beyond the
// residual taps: the model's OWN final norm + unembedding, applied to an
// arbitrary residual-stream tensor. Implemented by [Llama] (RMSNorm + Out,
// including the Granite logits scaling, so Unembed(ForwardHidden-residual)
// reproduces Forward's logits exactly) and [GPT] (LayerNorm + Head). This
// mirrors the reference implementation's LensModel.unembed seam: the lens
// never owns a decoder head, it borrows the model's.
type LensReadoutModel interface {
	ResidualModel

	// Unembed maps residual-stream rows h [n, dim] to logits [n, vocab] via
	// the model's final norm and output projection.
	Unembed(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error)
}

// Apply reads a layer-l residual out through the lens (§T812, §R250): the
// transported logit lens
//
//	JL_l(h) = W_U · norm(J_l · h)
//
// — h is transported into the final-layer basis by J_l, then decoded by the
// model's own final norm + unembedding ([LensReadoutModel.Unembed]), exactly
// like the reference implementation's lens.apply readout. h holds residual
// rows [n, dim] (one row per position, as handed to a [ResidualCapture]);
// the result is logits [n, vocab]. layer is the GoAI residual index: 0 =
// post-embedding, l ∈ 1..Layers = after block l; layer == Layers is the
// identity transport, i.e. the model's real output head on h. Unfitted
// layers (nil J, see [JLensFromPT]) are rejected.
//
// The transport itself runs in pure f64 Go (J is f64 and dim-sized — tiny
// next to the forward pass); only the norm + unembedding go through the
// backend, so Apply works on any backend the model runs on.
func (jl *JLens) Apply(model LensReadoutModel, ctx *backend.Context, h *tensor.Tensor, layer int) (*tensor.Tensor, error) {
	if layer < 0 || layer >= len(jl.J) {
		return nil, fmt.Errorf("nlp: JLens.Apply: layer %d outside 0..%d", layer, len(jl.J)-1)
	}
	J := jl.J[layer]
	if J == nil {
		return nil, fmt.Errorf("nlp: JLens.Apply: layer %d not fitted in this lens (imported reference artifacts carry block outputs only)", layer)
	}
	s := h.Shape()
	if len(s) != 2 || s[1] != jl.Dim {
		return nil, fmt.Errorf("nlp: JLens.Apply: residual shape %v, want [n %d]", s, jl.Dim)
	}
	n := s[0]
	jm := J.Storage().F64()
	out := tensor.New(tensor.F64, tensor.Shape{n, jl.Dim})
	dst := out.Storage().F64()
	for t := 0; t < n; t++ {
		row := dst[t*jl.Dim : (t+1)*jl.Dim]
		//perfscan:ignore PS3067 JLens.Apply interpretability probe, not hot inference
		for a := 0; a < jl.Dim; a++ {
			ja := jm[a*jl.Dim : (a+1)*jl.Dim]
			var acc float64
			//perfscan:ignore PS1005 lens transport AtF64, offline probe tiny vs forward, not hot
			for b, w := range ja {
				acc += w * h.AtF64(t, b)
			}
			row[a] = acc
		}
	}
	return model.Unembed(ctx, out)
}

// JLensToken is one (token, score) entry of a ranked lens readout.
type JLensToken struct {
	Token int     // vocabulary id
	Score float64 // readout logit
}

// JLensReadout is a single-position lens readout: the full logit vector plus
// the vocabulary ranked by it. Produced by [JLens.ApplyAt].
type JLensReadout struct {
	// Logits holds the readout logit per vocabulary id.
	Logits []float64
	// Ranked lists every vocabulary token by descending score, ties broken by
	// ascending id (deterministic).
	Ranked []JLensToken

	rank []int // rank per vocabulary id (same order as Logits)
}

// RankOf returns the 0-based rank of token in the readout (0 = top token);
// -1 if token is outside the vocabulary.
func (r *JLensReadout) RankOf(token int) int {
	if token < 0 || token >= len(r.rank) {
		return -1
	}
	return r.rank[token]
}

// newJLensReadout ranks one row of a [n, vocab] logits tensor.
func newJLensReadout(logits *tensor.Tensor, row int) *JLensReadout {
	vocab := logits.Shape()[1]
	r := &JLensReadout{
		Logits: make([]float64, vocab),
		Ranked: make([]JLensToken, vocab),
		rank:   make([]int, vocab),
	}
	// Devirtualize the per-element AtF64 dispatch: for a contiguous F64 tensor
	// read the row straight from the typed backing slice. AtF64(row,j) on a
	// row-major tensor is exactly storage[row*vocab+j], so bit-identical.
	if lc := logits.Contiguous(); lc.Dtype() == tensor.F64 {
		src := lc.Storage().F64()[row*vocab : row*vocab+vocab]
		for j := 0; j < vocab; j++ {
			v := src[j]
			r.Logits[j] = v
			r.Ranked[j] = JLensToken{Token: j, Score: v}
		}
	} else {
		for j := 0; j < vocab; j++ {
			v := logits.AtF64(row, j)
			r.Logits[j] = v
			r.Ranked[j] = JLensToken{Token: j, Score: v}
		}
	}
	// (Score desc, Token asc) is already a TOTAL order — Token is unique per element —
	// so stability adds nothing; unstable sort.Slice (pdqsort) gives the identical full-
	// vocab ranking at a fraction of symMerge's cost.
	//perfscan:ignore PS3002 already SliceStable to SortFunc; lens readout not hot path
	slices.SortFunc(r.Ranked, func(a, b JLensToken) int {
		if a.Score != b.Score {
			if a.Score > b.Score {
				return -1
			}
			return 1
		}
		if a.Token < b.Token {
			return -1
		}
		if a.Token > b.Token {
			return 1
		}
		return 0
	})
	for rk, e := range r.Ranked {
		r.rank[e.Token] = rk
	}
	return r
}

// ApplyAt is the one-call probe (§T812): it runs model.ForwardResiduals on
// tokens, reads the layer-l residual at one position through [JLens.Apply],
// and returns the ranked readout. position follows Python indexing — negative
// counts from the end (-1 = last position), mirroring the reference
// implementation's positions argument. layer uses the GoAI residual
// convention (see [JLens.Apply]).
func (jl *JLens) ApplyAt(model LensReadoutModel, ctx *backend.Context, tokens []int, layer, position int) (*JLensReadout, error) {
	if position < 0 {
		position += len(tokens)
	}
	if position < 0 || position >= len(tokens) {
		return nil, fmt.Errorf("nlp: JLens.ApplyAt: position outside the %d-token sequence", len(tokens))
	}
	var hl *tensor.Tensor
	if _, err := model.ForwardResiduals(ctx, tokens, func(l int, h *tensor.Tensor) {
		if l == layer {
			hl = h
		}
	}); err != nil {
		return nil, fmt.Errorf("nlp: JLens.ApplyAt: %w", err)
	}
	if hl == nil {
		return nil, fmt.Errorf("nlp: JLens.ApplyAt: layer %d not captured (model has %d+1 residual layers?)", layer, jl.Layers)
	}
	row := tensor.New(tensor.F64, tensor.Shape{1, jl.Dim})
	dst := row.Storage().F64()
	for b := 0; b < jl.Dim; b++ {
		dst[b] = hl.AtF64(position, b)
	}
	logits, err := jl.Apply(model, ctx, row, layer)
	if err != nil {
		return nil, err
	}
	return newJLensReadout(logits, 0), nil
}

// JLensSliceRow is one layer row of a [JLensSlice] grid.
type JLensSliceRow struct {
	// Layer is the residual layer index (GoAI convention: 0 = post-embedding,
	// l ∈ 1..Layers = after block l).
	Layer int
	// Top holds, per position, the top-1 token of this layer's transported
	// readout.
	Top []int
	// Rank holds, per position, the 0-based rank of the model's ACTUAL top-1
	// next-token prediction under this layer's readout — 0 means this row
	// already agrees with the model's final output.
	Rank []int
}

// JLensSlice is the layer × position readout grid behind the paper's slice
// visualisation (§R250, rendered by §T814): every fitted layer read out at
// every position. Produced by [JLens.Slice].
type JLensSlice struct {
	// Tokens is the prompt the grid was computed on.
	Tokens []int
	// Output holds, per position, the model's actual top-1 next-token
	// prediction (the grid's bottom row: the layer == Layers readout is the
	// identity transport, so its Top equals Output and its Rank is 0).
	Output []int
	// Rows lists the per-layer grid rows in ascending Layer order. Unfitted
	// layers (nil J, e.g. layer 0 of an imported reference artifact) are
	// skipped; the last row is always layer == Layers.
	Rows []JLensSliceRow
}

// Slice computes the full layer × position lens grid for a prompt (§T812):
// one forward pass with residual capture, then every fitted layer's
// transported readout at every position. The returned data — top-1 token per
// cell plus the rank of the model's real next-token prediction under each
// cell's readout — is exactly what the §T814 slice page renders.
func (jl *JLens) Slice(model LensReadoutModel, ctx *backend.Context, tokens []int) (*JLensSlice, error) {
	resids := make([]*tensor.Tensor, 0, jl.Layers+1)
	if _, err := model.ForwardResiduals(ctx, tokens, func(l int, h *tensor.Tensor) {
		resids = append(resids, h)
	}); err != nil {
		return nil, fmt.Errorf("nlp: JLens.Slice: %w", err)
	}
	if len(resids) != jl.Layers+1 {
		return nil, fmt.Errorf("nlp: JLens.Slice: model has %d residual layers, lens %d", len(resids)-1, jl.Layers+1)
	}
	seq := len(tokens)

	// Bottom row first: the identity transport at layer == Layers IS the
	// model's own readout; its argmax defines Output.
	sl := &JLensSlice{Tokens: tokens, Output: make([]int, seq)}
	final, err := jl.Apply(model, ctx, resids[jl.Layers], jl.Layers)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3065 lens Slice per-position loop, offline diagnostic
	for p := 0; p < seq; p++ {
		sl.Output[p] = argmaxRow(final, p)
	}
	for l := 0; l <= jl.Layers; l++ {
		if jl.J[l] == nil {
			continue
		}
		logits := final
		if l != jl.Layers {
			if logits, err = jl.Apply(model, ctx, resids[l], l); err != nil {
				return nil, err
			}
		}
		row := JLensSliceRow{Layer: l, Top: make([]int, seq), Rank: make([]int, seq)}
		//perfscan:ignore PS3065 lens Slice per-position loop, offline diagnostic
		for p := 0; p < seq; p++ {
			row.Top[p] = argmaxRow(logits, p)
			row.Rank[p] = rankInRow(logits, p, sl.Output[p])
		}
		sl.Rows = append(sl.Rows, row)
	}
	return sl, nil
}

// argmaxRow returns the column with the highest value in row p (first hit on
// ties — deterministic).
func argmaxRow(logits *tensor.Tensor, p int) int {
	vocab := logits.Shape()[1]
	best, bestV := 0, logits.AtF64(p, 0)
	//perfscan:ignore PS1001,PS3068 argmaxRow AtF64 in lens diagnostic, not hot | argmaxRow lens diagnostic, not hot
	for j := 1; j < vocab; j++ {
		if v := logits.AtF64(p, j); v > bestV {
			best, bestV = j, v
		}
	}
	return best
}

// rankInRow returns the 0-based rank of token in row p: the number of tokens
// scoring strictly higher, plus earlier-id ties (consistent with
// [JLensReadout.Ranked]'s ordering).
func rankInRow(logits *tensor.Tensor, p, token int) int {
	v := logits.AtF64(p, token)
	rank := 0
	for j := 0; j < logits.Shape()[1]; j++ {
		if w := logits.AtF64(p, j); w > v || (w == v && j < token) {
			rank++
		}
	}
	return rank
}
