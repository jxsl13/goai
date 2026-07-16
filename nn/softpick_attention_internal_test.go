package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// spExec runs a single-output op through a fresh context (internal test helper).
func spExec(t *testing.T, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	o, err := backend.Execute(backend.NewContext(), op, in, at)
	if err != nil {
		t.Fatal(err)
	}
	return o[0]
}

func spScores(vals [][]float64) *tensor.Tensor {
	m := tensor.New(tensor.F64, tensor.Shape{len(vals), len(vals[0])})
	for i := range vals {
		for j := range vals[i] {
			m.SetF64(vals[i][j], i, j)
		}
	}
	return m
}

// §V16 (a) COLLAPSE — the anchor. Driving the appended sink logit to −∞ (≈−1e30)
// zeroes the sink column, so softpickWeights (append-[s,sink] → softmax → drop
// sink) reduces EXACTLY to an ordinary row-softmax over the real keys. Verified
// both at the weight level and end-to-end through the full layer, at 1e-10.
func TestSoftpickCollapseToSoftmax(t *testing.T) {
	scores := spScores([][]float64{
		{0.5, -1.0, 2.0, 0.0},
		{-3.0, 1.5, 0.25, -0.5},
		{2.0, 2.0, 2.0, 2.0},
	})
	sp := softpickWeightsMust(t, scores, -1e30) // sink logit → −∞
	std := spExec(t, backend.OpSoftmax, nil, scores)
	var maxDiff float64
	for i := range scores.Shape()[0] {
		for j := range scores.Shape()[1] {
			if d := math.Abs(sp.AtF64(i, j) - std.AtF64(i, j)); d > maxDiff {
				maxDiff = d
			}
		}
	}
	if maxDiff > 1e-10 {
		t.Errorf("weight-level collapse: softpick(sink=−∞) vs softmax max diff %.3g > 1e-10", maxDiff)
	}

	// End-to-end: the whole layer with sink→−∞ ≡ standard-softmax multi-head
	// attention over the same projections (built here from the block's pieces).
	const dim, heads, seq = 8, 2, 5
	s, err := NewSoftpickAttention(tensor.F64, dim, heads, 21, WithSoftpickCausal())
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	for i := range seq {
		for j := range dim {
			x.SetF64(math.Sin(float64(i*dim+j)), i, j)
		}
	}
	collapsed, err := s.forward(backend.NewContext(), x, -1e30)
	if err != nil {
		t.Fatal(err)
	}
	ref := standardSoftmaxRef(t, s, x)
	var maxE2E float64
	for i := range seq {
		for j := range dim {
			if d := math.Abs(collapsed.AtF64(i, j) - ref.AtF64(i, j)); d > maxE2E {
				maxE2E = d
			}
		}
	}
	if maxE2E > 1e-10 {
		t.Errorf("end-to-end collapse: layer(sink=−∞) vs standard softmax attention max diff %.3g > 1e-10", maxE2E)
	}
	t.Logf("collapse: weight-level %.3g, end-to-end %.3g (both < 1e-10)", maxDiff, maxE2E)
}

// §V16 (a') HAND-COMPUTED softmax_1 PARITY. softpickWeights with sink logit 0
// must equal the closed-form softmax_1(s)_i = e^{s_i}/(1+Σ_j e^{s_j}) to 1e-10,
// and the per-row deficit 1−Σ_i weight_i must equal the sink mass 1/(1+Σe^{s_j}).
func TestSoftpickHandSoftmax1Parity(t *testing.T) {
	rows := [][]float64{
		{0.5, -1.0, 2.0},
		{-2.0, 0.0, 0.75},
	}
	scores := spScores(rows)
	w := softpickWeightsMust(t, scores, 0)
	for i, r := range rows {
		denom := 1.0
		for _, s := range r {
			denom += math.Exp(s)
		}
		var wsum float64
		for j, s := range r {
			want := math.Exp(s) / denom
			got := w.AtF64(i, j)
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("row %d col %d: softmax_1 %.12g, want %.12g", i, j, got, want)
			}
			wsum += got
		}
		sinkMass := 1.0 / denom
		if d := math.Abs((1 - wsum) - sinkMass); d > 1e-10 {
			t.Errorf("row %d: weight deficit %.12g != sink mass %.12g (diff %.3g)", i, 1-wsum, sinkMass, d)
		}
		if wsum >= 1 {
			t.Errorf("row %d: non-sink weights sum %.12g must be < 1", i, wsum)
		}
	}
}

// §V16 (c) THE VALUE — attend-to-nothing / no forced normalization. On a query
// whose scores against every key are very negative (nothing relevant), ordinary
// softmax still forces the weights to sum to 1 and produces an O(‖V‖) output;
// softmax_1 lets the mass drain into the sink, so the non-sink weights sum to ≈0
// and the attention OUTPUT norm collapses toward zero — the massive-activation /
// sink-suppression property Softpick targets.
func TestSoftpickAttendToNothing(t *testing.T) {
	// One "nothing relevant" row (all very negative) and one ordinary row.
	scores := spScores([][]float64{
		{-30, -30, -30, -30},   // nothing worth attending to
		{3.0, -1.0, 0.5, -2.0}, // a normal row for contrast
	})
	sp := softpickWeightsMust(t, scores, 0)
	std := spExec(t, backend.OpSoftmax, nil, scores)

	// Non-sink weight sum on the all-negative row: ~0 for softpick, exactly 1
	// for standard softmax; the deficit is the sink mass.
	var spSum, stdSum float64
	for j := range 4 {
		spSum += sp.AtF64(0, j)
		stdSum += std.AtF64(0, j)
	}
	sinkMass := 1 - spSum
	if spSum >= 1 || spSum > 1e-10 {
		t.Errorf("softpick non-sink weight sum %.3g on all-negative row: want ≪1 (near 0)", spSum)
	}
	if math.Abs(stdSum-1) > 1e-10 {
		t.Errorf("standard softmax weight sum %.3g, want exactly 1 (forced normalization)", stdSum)
	}
	if sinkMass < 0.99 {
		t.Errorf("sink mass %.6g on all-negative row: want ≈1 (query attends to nothing)", sinkMass)
	}

	// Attention output = weights·V. Softpick output norm ≪ standard output norm.
	v := spScores([][]float64{
		{1.0, -2.0},
		{0.5, 3.0},
		{-1.5, 1.0},
		{2.0, -0.5},
	})
	spOut := spExec(t, backend.OpMatMul, nil, sp, v)   // [2,2]
	stdOut := spExec(t, backend.OpMatMul, nil, std, v) // [2,2]
	spNorm := math.Hypot(spOut.AtF64(0, 0), spOut.AtF64(0, 1))
	stdNorm := math.Hypot(stdOut.AtF64(0, 0), stdOut.AtF64(0, 1))
	if spNorm > 1e-9 {
		t.Errorf("softpick output norm %.3g on all-negative row: want ≈0 (attends to nothing)", spNorm)
	}
	if stdNorm < 0.5 {
		t.Errorf("standard softmax output norm %.3g: expected O(‖V‖), softmax forced a pick", stdNorm)
	}
	if stdNorm <= spNorm*1e6 {
		t.Errorf("softpick norm %.3g not ≪ standard norm %.3g — sink did not suppress the output", spNorm, stdNorm)
	}
	t.Logf("attend-to-nothing: softpick non-sink sum %.3g, sink mass %.6g; output norms softpick %.3g vs standard %.3g",
		spSum, sinkMass, spNorm, stdNorm)
}

// softpickWeightsMust runs softpickWeights or fails the test.
func softpickWeightsMust(t *testing.T, scores *tensor.Tensor, sinkLogit float64) *tensor.Tensor {
	t.Helper()
	w, err := softpickWeights(backend.NewContext(), scores, sinkLogit)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// standardSoftmaxRef recomputes s.forward with ORDINARY softmax over the real
// keys (no sink augmentation) — the standard multi-head attention baseline the
// softmax_1 layer must collapse to when the sink logit → −∞.
func standardSoftmaxRef(t *testing.T, s *SoftpickAttention, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			t.Fatal(err)
		}
		return o[0]
	}
	tlen := x.Shape()[0]
	proj := func(l *Linear) *tensor.Tensor {
		o, err := l.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	q, k, v := proj(s.QProj), proj(s.KProj), proj(s.VProj)
	invSqrtD := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(s.HeadDim)))
	var mask *tensor.Tensor
	if s.causal {
		mask = qkCausalMask(x.Dtype(), tlen, tlen)
	}
	heads := make([]*tensor.Tensor, s.Heads)
	for h := range s.Heads {
		lo, hi := h*s.HeadDim, (h+1)*s.HeadDim
		sl := func(m *tensor.Tensor) *tensor.Tensor {
			return ex(backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, m)
		}
		qh, kh, vh := sl(q), sl(k), sl(v)
		sc := ex(backend.OpMul, nil, ex(backend.OpMatMul, nil, qh, ex(backend.OpTranspose, nil, kh)), invSqrtD)
		if s.causal {
			sc = ex(backend.OpAdd, nil, sc, mask)
		}
		w := ex(backend.OpSoftmax, nil, sc) // ordinary softmax, no sink
		heads[h] = ex(backend.OpMatMul, nil, w, vh)
	}
	concat := ex(backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	o, err := s.OProj.Forward(ctx, concat)
	if err != nil {
		t.Fatal(err)
	}
	return o
}
