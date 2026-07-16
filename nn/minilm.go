package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MiniLMRelation selects one pairwise self-attention relation to transfer in
// MiniLM deep self-attention distillation (Wang, Wei, Dong, Bao, Yang & Zhou
// 2020, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression
// of Pre-Trained Transformers", arXiv:2002.10957; and Wang, Bao, Huang, Dong &
// Wei 2020, "MiniLMv2: Multi-Head Self-Attention Relation Distillation",
// arXiv:2012.15828). A relation (a,b) is the per-head row-softmax of the scaled
// Gram matrix softmax(a·bᵀ/√dₕ) ∈ [seq,seq] between two of the layer's Q/K/V
// projections. In plain terms: instead of copying the teacher's internal
// vectors (which have the wrong size for a smaller student), MiniLM copies how
// strongly each position relates to every other position — a picture whose
// size depends only on the sequence, never on the model width.
type MiniLMRelation int

const (
	// MiniLMRelationQK is the attention distribution softmax(Q·Kᵀ/√dₕ) — the
	// v1 paper's attention-transfer term (arXiv:2002.10957 §3.1, Eq. 5).
	MiniLMRelationQK MiniLMRelation = iota + 1
	// MiniLMRelationQQ is the query self-relation softmax(Q·Qᵀ/√dₕ) — part of
	// MiniLMv2's multi-relation transfer (arXiv:2012.15828 §3.2).
	MiniLMRelationQQ
	// MiniLMRelationKK is the key self-relation softmax(K·Kᵀ/√dₕ) — part of
	// MiniLMv2's multi-relation transfer (arXiv:2012.15828 §3.2).
	MiniLMRelationKK
	// MiniLMRelationVV is the value self-relation softmax(V·Vᵀ/√dₕ) — v1's
	// value-relation term (arXiv:2002.10957 §3.1, Eq. 6) and the key MiniLM
	// idea: it is width-independent, so teacher and student head dims may differ.
	MiniLMRelationVV
)

// miniLMConfig carries the MiniLMLoss knobs set via MiniLMOption.
type miniLMConfig struct {
	relations []MiniLMRelation
}

// MiniLMOption configures MiniLMLoss (functional options; see
// WithMiniLMRelations).
type MiniLMOption func(*miniLMConfig)

// WithMiniLMRelations selects which self-attention relations MiniLMLoss
// transfers from teacher to student. The default (no option) is MiniLM v1:
// {MiniLMRelationQK, MiniLMRelationVV} — attention-distribution transfer plus
// value-relation transfer (arXiv:2002.10957 §3.1). Passing
// {MiniLMRelationQQ, MiniLMRelationKK, MiniLMRelationVV} gives MiniLMv2's
// multi-head self-attention relation distillation (arXiv:2012.15828 §3.2).
// A single {MiniLMRelationQK} is the attention-only ablation from the v1
// paper. Each listed relation contributes one KL term to the summed loss, so
// listing a relation twice doubles its weight; the list must be non-empty and
// every entry must be one of the four MiniLMRelation constants.
func WithMiniLMRelations(relations ...MiniLMRelation) MiniLMOption {
	return func(c *miniLMConfig) { c.relations = relations }
}

// MiniLMRelationMaps computes the per-head MiniLM relation maps between two
// same-width projections x, y ∈ [seq, dim] of one attention layer, re-split
// into the given number of relation heads (dim must be divisible by heads,
// heads ≥ 1). For each head h with slice width dₕ = dim/heads it returns
//
//	Rₕ = softmax( xₕ·yₕᵀ / √dₕ )  ∈  [seq, seq]
//
// — a row-stochastic matrix whose row i is a probability distribution over
// positions. With (x,y) = (Q,K) this is the ordinary attention distribution;
// with x = y it is the self-relation of MiniLM (V·Vᵀ, arXiv:2002.10957 §3.1)
// and MiniLMv2 (Q·Qᵀ/K·Kᵀ/V·Vᵀ, arXiv:2012.15828 §3). In plain terms: each map
// answers "how similar is position i's vector to position j's?", normalized so
// every row sums to one. heads is MiniLMv2's relation-head count: it need not
// equal the model's attention-head count (re-splitting decouples teacher and
// student head counts); heads=1 yields a single whole-width relation map, and
// heads equal to the layer's attention-head count reproduces the v1 per-head
// relations exactly. The result is differentiable w.r.t. x and y.
func MiniLMRelationMaps(ctx *backend.Context, x, y *tensor.Tensor, heads int) ([]*tensor.Tensor, error) {
	if x.Ndim() != 2 || y.Ndim() != 2 || !x.Shape().Equal(y.Shape()) {
		return nil, fmt.Errorf("nn: MiniLMRelationMaps wants equal-shaped rank-2 [seq,dim] inputs, got %v and %v", x.Shape(), y.Shape())
	}
	if x.Dtype() != y.Dtype() {
		return nil, fmt.Errorf("nn: MiniLMRelationMaps dtype mismatch: %v vs %v", x.Dtype(), y.Dtype())
	}
	d := x.Shape()[1]
	if heads < 1 || d%heads != 0 {
		return nil, fmt.Errorf("nn: MiniLMRelationMaps heads must be ≥1 and divide dim %d, got %d", d, heads)
	}
	dh := d / heads
	invSqrt := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(dh)))
	maps := make([]*tensor.Tensor, 0, heads)
	for h := range heads {
		attrs := backend.SliceAttrs{Axis: 1, Start: h * dh, End: (h + 1) * dh}
		xh, err := rdropExec(ctx, backend.OpSlice, attrs, x) // [seq, dh]
		if err != nil {
			return nil, err
		}
		yh, err := rdropExec(ctx, backend.OpSlice, attrs, y) // [seq, dh]
		if err != nil {
			return nil, err
		}
		yt, err := rdropExec(ctx, backend.OpTranspose, nil, yh) // [dh, seq]
		if err != nil {
			return nil, err
		}
		s, err := rdropExec(ctx, backend.OpMatMul, nil, xh, yt) // [seq, seq]
		if err != nil {
			return nil, err
		}
		if s, err = rdropExec(ctx, backend.OpMul, nil, s, invSqrt); err != nil { // broadcast [seq,seq]·[1,1]
			return nil, err
		}
		m, err := rdropExec(ctx, backend.OpSoftmax, nil, s) // row softmax
		if err != nil {
			return nil, err
		}
		maps = append(maps, m)
	}
	return maps, nil
}

// MiniLMLoss is the MiniLM deep self-attention distillation loss (Wang et al.
// 2020, arXiv:2002.10957; MiniLMv2: Wang et al. 2020, arXiv:2012.15828). It
// distills a TEACHER transformer's self-attention BEHAVIOR — not its output
// logits (see GKDLoss/DistillLoss for those) — into a smaller STUDENT, by
// matching per-head relation maps computed from one layer's Q, K, V
// projections. For each selected relation (a,b) (see WithMiniLMRelations) and
// each of the given relation heads, with R = softmax(a·bᵀ/√dₕ) ∈ [seq,seq]:
//
//	L = Σ_relations (1/(heads·seq)) Σ_h Σ_i KL( Rᵀᵉᵃᶜʰᵉʳ_{h,i} ‖ Rˢᵗᵘᵈᵉⁿᵗ_{h,i} )
//
// i.e. the KL divergence between the teacher's and the student's relation
// distributions, averaged over heads and query positions and summed over the
// selected relations (v1 Eq. 4–6; v2 Eq. 2–4). In plain terms: the student is
// trained so that its positions attend to (and resemble) each other in the
// same proportions as the teacher's — a behavior target whose [seq,seq] shape
// is independent of either model's width, which is why a thin student can
// learn from a wide teacher.
//
// Inputs are one layer's projections, rank-2 [seq, width]: studentQ/K/V share
// a width, teacherQ/K/V share a width, and the two widths MAY DIFFER — only
// the sequence length, dtype and relation-head count must match. The paper
// transfers from the teacher's LAST transformer layer to the student's last
// layer (v1 §3.1); for deep (24-layer) teachers MiniLMv2 found an upper-middle
// teacher layer transfers better than the last (v2 §4) — the layer choice is
// the caller's, since this function just consumes that layer's Q/K/V.
//
// Knobs: heads is the number of relation heads both sides are re-split into
// (must be ≥1 and divide both widths; heads=1 → one whole-width relation;
// heads = the shared attention-head count → exactly MiniLM v1; a count larger
// than the attention-head count gives MiniLMv2's fine-grained relations and
// decouples mismatched teacher/student head counts, v2 §3.2). Relations
// default to v1's {Q-K attention, V-V value relation}; see WithMiniLMRelations
// for the v2 multi-relation setting.
//
// The teacher tensors are detached internally (stop-gradient), so the loss is
// differentiable w.r.t. the student's Q, K, V only and the teacher receives
// zero gradient. The returned scalar is ≥ 0 and is exactly 0 when student and
// teacher produce identical relation maps.
func MiniLMLoss(ctx *backend.Context, studentQ, studentK, studentV, teacherQ, teacherK, teacherV *tensor.Tensor, heads int, opts ...MiniLMOption) (*tensor.Tensor, error) {
	cfg := miniLMConfig{relations: []MiniLMRelation{MiniLMRelationQK, MiniLMRelationVV}} // v1 default
	for _, opt := range opts {
		opt(&cfg)
	}
	if len(cfg.relations) == 0 {
		return nil, fmt.Errorf("nn: MiniLMLoss needs at least one relation")
	}
	for _, r := range cfg.relations {
		if r < MiniLMRelationQK || r > MiniLMRelationVV {
			return nil, fmt.Errorf("nn: MiniLMLoss unknown relation %d", r)
		}
	}
	student := [3]*tensor.Tensor{studentQ, studentK, studentV}
	teacher := [3]*tensor.Tensor{teacherQ, teacherK, teacherV}
	for _, trio := range [2][3]*tensor.Tensor{student, teacher} {
		for _, t := range trio {
			if t.Ndim() != 2 {
				return nil, fmt.Errorf("nn: MiniLMLoss wants rank-2 [seq,width] projections, got %v", t.Shape())
			}
			if t.Dtype() != studentQ.Dtype() {
				return nil, fmt.Errorf("nn: MiniLMLoss dtype mismatch: %v vs %v", t.Dtype(), studentQ.Dtype())
			}
			if !t.Shape().Equal(trio[0].Shape()) {
				return nil, fmt.Errorf("nn: MiniLMLoss Q/K/V of one model must share a shape, got %v and %v", trio[0].Shape(), t.Shape())
			}
		}
	}
	seq := studentQ.Shape()[0]
	if teacherQ.Shape()[0] != seq {
		return nil, fmt.Errorf("nn: MiniLMLoss student/teacher seq mismatch: %d vs %d", seq, teacherQ.Shape()[0])
	}
	if heads < 1 || studentQ.Shape()[1]%heads != 0 || teacherQ.Shape()[1]%heads != 0 {
		return nil, fmt.Errorf("nn: MiniLMLoss heads must be ≥1 and divide student width %d and teacher width %d, got %d",
			studentQ.Shape()[1], teacherQ.Shape()[1], heads)
	}

	// Detach the teacher: it is a frozen target, no gradient may flow to it.
	for i, t := range teacher {
		d, err := rdropExec(ctx, backend.OpStopGradient, nil, t)
		if err != nil {
			return nil, err
		}
		teacher[i] = d
	}

	// pair maps a relation to its (x,y) operands within one model's Q/K/V.
	pair := func(qkv [3]*tensor.Tensor, r MiniLMRelation) (x, y *tensor.Tensor) {
		switch r {
		case MiniLMRelationQK:
			return qkv[0], qkv[1]
		case MiniLMRelationQQ:
			return qkv[0], qkv[0]
		case MiniLMRelationKK:
			return qkv[1], qkv[1]
		default: // MiniLMRelationVV
			return qkv[2], qkv[2]
		}
	}

	var total *tensor.Tensor // running Σ of KL terms over relations, heads, rows
	for _, rel := range cfg.relations {
		sx, sy := pair(student, rel)
		tx, ty := pair(teacher, rel)
		sMaps, err := MiniLMRelationMaps(ctx, sx, sy, heads)
		if err != nil {
			return nil, err
		}
		tMaps, err := MiniLMRelationMaps(ctx, tx, ty, heads)
		if err != nil {
			return nil, err
		}
		for h := range heads {
			logT, err := rdropExec(ctx, backend.OpLog, nil, tMaps[h])
			if err != nil {
				return nil, err
			}
			logS, err := rdropExec(ctx, backend.OpLog, nil, sMaps[h])
			if err != nil {
				return nil, err
			}
			terms, err := klTerms(ctx, tMaps[h], logT, logS) // p·(log p − log q), p = teacher
			if err != nil {
				return nil, err
			}
			s, err := rdropExec(ctx, backend.OpSum, nil, terms) // Σ over rows and columns
			if err != nil {
				return nil, err
			}
			if total == nil {
				total = s
			} else if total, err = rdropExec(ctx, backend.OpAdd, nil, total, s); err != nil {
				return nil, err
			}
		}
	}
	// mean over heads and query positions (v1 Eq. 5–6): ÷ (heads·seq)
	return rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: 1 / float64(heads*seq)},
		total, tensor.New(studentQ.Dtype(), tensor.Shape{}))
}
