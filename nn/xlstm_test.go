package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// mlstmRandCol returns an [r,1] column of N(0,1) gate pre-activations.
func mlstmRandCol(rng *rand.Rand, r int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, 1})
	for i := range r {
		t.SetF64(rng.NormFloat64(), i, 0)
	}
	return t
}

// mlstmRecurNaive is the UNSTABILIZED sequential mLSTM recurrence (no running-max
// m_t): i_t=exp(ĩ_t), f_t=σ(f̃_t) or exp(f̃_t), C_t=f_t·C_{t-1}+i_t·v_t k_tᵀ,
// n_t likewise, h_t=C_t q_t/max(|n_tᵀ q_t|,1). Overflows for large ĩ/f̃ — the foil
// for the stabilization-invariance anchor.
func mlstmRecurNaive(q, k, v, ipre, fpre *tensor.Tensor, expForget bool) *tensor.Tensor {
	seq, dk := q.Shape()[0], q.Shape()[1]
	dv := v.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{seq, dv})
	c := make([]float64, dv*dk)
	n := make([]float64, dk)
	for t := range seq {
		it := math.Exp(ipre.AtF64(t, 0))
		ft := 1 / (1 + math.Exp(-fpre.AtF64(t, 0)))
		if expForget {
			ft = math.Exp(fpre.AtF64(t, 0))
		}
		for a := range dv {
			va := v.AtF64(t, a)
			base := a * dk
			for i := range dk {
				c[base+i] = ft*c[base+i] + it*va*k.AtF64(t, i)
			}
		}
		for i := range dk {
			n[i] = ft*n[i] + it*k.AtF64(t, i)
		}
		var nq float64
		for i := range dk {
			nq += n[i] * q.AtF64(t, i)
		}
		denom := math.Max(math.Abs(nq), 1)
		for a := range dv {
			var cq float64
			base := a * dk
			for i := range dk {
				cq += c[base+i] * q.AtF64(t, i)
			}
			out.SetF64(cq/denom, t, a)
		}
	}
	return out
}

func mlstmMaxAbsDiff(a, b *tensor.Tensor) float64 {
	var m float64
	for i := range a.Numel() {
		coord := tensor.Unravel(i, a.Shape())
		if d := math.Abs(a.AtF64(coord...) - b.AtF64(coord...)); d > m {
			m = d
		}
	}
	return m
}

// §V16.1 (THE anchor, paper-CONFIRMED, arXiv:2405.04517 App. A): the parallel
// training form MLSTMParallel and the step-by-step recurrence MLSTMRecurrent
// produce identical output at f64 1e-10, for both forget-gate variants.
func TestMLSTMDuality(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, exp := range []bool{false, true} {
		for trial := range 5 {
			const seq, dk, dv = 7, 4, 3
			q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
			ipre, fpre := mlstmRandCol(rng, seq), mlstmRandCol(rng, seq)
			par, err := nn.MLSTMParallel(backend.NewContext(), q, k, v, ipre, fpre, exp)
			if err != nil {
				t.Fatal(err)
			}
			rec, err := nn.MLSTMRecurrent(q, k, v, ipre, fpre, exp)
			if err != nil {
				t.Fatal(err)
			}
			if d := mlstmMaxAbsDiff(par, rec); d > 1e-10 {
				t.Errorf("expForget=%v trial %d: parallel≠recurrent, maxAbsDiff=%.3g", exp, trial, d)
			}
		}
	}
}

// §V16.2 (COLLAPSE, defining property): with the input gate i_t≡1 (ĩ_t=0) and
// forget f_t≡1 (exp forget, f̃_t=0) the mLSTM memory is C_t=Σ_{j≤t} v_j k_jᵀ, so
// h_t = (Σ_{j≤t}(q_t·k_j)v_j)/max(|Σ_{j≤t} q_t·k_j|,1): unnormalized causal linear
// attention divided by the max(·,1) floor. Matched at f64 1e-10 against the
// attention-form oracle (linAttnRef, shared with the GLA suite).
func TestMLSTMCollapseToLinearAttention(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const seq, dk, dv = 6, 3, 4
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	zeros := tensor.New(tensor.F64, tensor.Shape{seq, 1}) // ĩ=f̃=0 ⇒ i=1, f=exp(0)=1
	got, err := nn.MLSTMParallel(backend.NewContext(), q, k, v, zeros, zeros, true)
	if err != nil {
		t.Fatal(err)
	}
	lin := linAttnRef(toRows(q), toRows(k), toRows(v)) // numerator Σ_{j≤t}(q·k_j)v_j
	for i := range seq {
		var rowsum float64 // Σ_{j≤t} q_i·k_j
		for j := 0; j <= i; j++ {
			for c := range dk {
				rowsum += q.AtF64(i, c) * k.AtF64(j, c)
			}
		}
		denom := math.Max(math.Abs(rowsum), 1)
		for c := range dv {
			want := lin[i][c] / denom
			if d := math.Abs(got.AtF64(i, c) - want); d > 1e-10 {
				t.Errorf("out[%d,%d]=%.10g want %.10g", i, c, got.AtF64(i, c), want)
			}
		}
	}
}

// §V16.3 (STABILIZATION invariance + overflow safety): the App. A stabilized
// recurrence equals the naive unstabilized one at f64 1e-8 on moderate gate
// pre-activations, AND stays finite on large pre-activations where the naive form
// overflows to Inf/NaN.
func TestMLSTMStabilizationInvariance(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const seq, dk, dv = 6, 3, 3
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	ipre, fpre := mlstmRandCol(rng, seq), mlstmRandCol(rng, seq)
	stab, err := nn.MLSTMRecurrent(q, k, v, ipre, fpre, false)
	if err != nil {
		t.Fatal(err)
	}
	naive := mlstmRecurNaive(q, k, v, ipre, fpre, false)
	if d := mlstmMaxAbsDiff(stab, naive); d > 1e-8 {
		t.Errorf("stabilized≠naive on moderate inputs, maxAbsDiff=%.3g", d)
	}
	// Large input-gate pre-activations: exp(ĩ) overflows the naive recurrence.
	big := tensor.New(tensor.F64, tensor.Shape{seq, 1})
	for i := range seq {
		big.SetF64(800, i, 0) // exp(800)=+Inf in f64
	}
	naiveBig := mlstmRecurNaive(q, k, v, big, fpre, false)
	var sawBad bool
	for i := range naiveBig.Numel() {
		if val := naiveBig.AtF64(tensor.Unravel(i, naiveBig.Shape())...); math.IsNaN(val) || math.IsInf(val, 0) {
			sawBad = true
		}
	}
	if !sawBad {
		t.Error("expected the naive recurrence to overflow at ĩ=800")
	}
	stabBig, err := nn.MLSTMRecurrent(q, k, v, big, fpre, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := range stabBig.Numel() {
		coord := tensor.Unravel(i, stabBig.Shape())
		if val := stabBig.AtF64(coord...); math.IsNaN(val) || math.IsInf(val, 0) {
			t.Errorf("stabilized recurrence not finite at ĩ=800: out%v=%v", coord, val)
		}
	}
	// The stabilized parallel form must likewise stay finite at ĩ=800.
	parBig, err := nn.MLSTMParallel(backend.NewContext(), q, k, v, big, fpre, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := range parBig.Numel() {
		coord := tensor.Unravel(i, parBig.Shape())
		if val := parBig.AtF64(coord...); math.IsNaN(val) || math.IsInf(val, 0) {
			t.Errorf("stabilized parallel form not finite at ĩ=800: out%v=%v", coord, val)
		}
	}
}

// §V16.4 (GRADCHECK): finite-difference gradient check of MLSTMParallel through
// q, k, v and both gate pre-activations ĩ, f̃; max rel err ≤ 1e-4.
func TestMLSTMGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const seq, dk, dv = 5, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	ipre, fpre := mlstmRandCol(rng, seq), mlstmRandCol(rng, seq)
	tape := autograd.NewTape()
	out, err := nn.MLSTMParallel(tape.Context(), q, k, v, ipre, fpre, false)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		o, _ := nn.MLSTMParallel(backend.NewContext(), q, k, v, ipre, fpre, false)
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := fwd()
			p.SetF64(o-h, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("q", q)
	check("k", k)
	check("v", v)
	check("ipre", ipre)
	check("fpre", fpre)
}

// §V16.6 (sanity): the MLSTM layer forward is shape-correct, deterministic, and
// its parameter Forward path trains (gradients reach all five projections).
func TestMLSTMLayer(t *testing.T) {
	const dModel, seq = 8, 5
	m, err := nn.NewMLSTM(tensor.F64, dModel, 42, nn.WithMLSTMHeads(2))
	if err != nil {
		t.Fatal(err)
	}
	if m.HeadDim != 4 {
		t.Fatalf("HeadDim=%d want 4", m.HeadDim)
	}
	x := randMat(rand.New(rand.NewPCG(9, 10)), seq, dModel)
	y1, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !y1.Shape().Equal(tensor.Shape{seq, dModel}) {
		t.Fatalf("forward shape %v want [%d,%d]", y1.Shape(), seq, dModel)
	}
	y2, _ := m.Forward(backend.NewContext(), x)
	if d := mlstmMaxAbsDiff(y1, y2); d != 0 {
		t.Errorf("non-deterministic forward, maxAbsDiff=%.3g", d)
	}
	// Gradients reach every projection.
	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	for i, p := range m.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Errorf("param %d: nil grad", i)
			continue
		}
		var nz bool
		for j := range g.Numel() {
			if g.AtF64(tensor.Unravel(j, g.Shape())...) != 0 {
				nz = true
				break
			}
		}
		if !nz {
			t.Errorf("param %d: all-zero grad", i)
		}
	}
}

// §V16.6 (sanity): dModel must divide by heads; shape contract enforced.
func TestMLSTMErrors(t *testing.T) {
	if _, err := nn.NewMLSTM(tensor.F64, 8, 0, nn.WithMLSTMHeads(3)); err == nil {
		t.Error("heads not dividing dModel: want error")
	}
	if _, err := nn.NewMLSTM(tensor.F64, 0, 0); err == nil {
		t.Error("non-positive dModel: want error")
	}
	m, _ := nn.NewMLSTM(tensor.F64, 4, 0)
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("wrong width x: want error")
	}
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.MLSTMParallel(backend.NewContext(), r(2, 3), r(3, 3), r(2, 3), r(2, 1), r(2, 1), false); err == nil {
		t.Error("seq mismatch: want error")
	}
	if _, err := nn.MLSTMParallel(backend.NewContext(), r(2, 3), r(2, 3), r(2, 3), r(2, 2), r(2, 1), false); err == nil {
		t.Error("ĩ not [L,1]: want error")
	}
}

// §V16.5 (THE VALUE): a delayed-copy long-memory task. Each sequence's target at
// the LAST position is its FIRST-token vector — the mLSTM must carry x[0] through
// the whole sequence in its matrix memory and the query at the last step must read
// it back. Train W_q/W_k/W_v/W_i/W_f plus a linear readout with Adam; assert the
// loss falls by ≥5× and report initial/final. Also trains the exponential-forget
// variant to show both gate types learn.
func TestMLSTMLongMemoryTask(t *testing.T) {
	train := func(expForget bool) (float64, float64) {
		const dModel, seq, nSeq, steps = 6, 8, 6, 500
		rng := rand.New(rand.NewPCG(11, 12))
		opts := []nn.MLSTMOption{}
		if expForget {
			opts = append(opts, nn.WithMLSTMExpForget())
		}
		m, _ := nn.NewMLSTM(tensor.F64, dModel, 100, opts...)
		readout := nn.NewLinear(tensor.F64, dModel, dModel, 200)
		xs := make([]*tensor.Tensor, nSeq)
		ys := make([]*tensor.Tensor, nSeq) // target = x[0]
		for s := range nSeq {
			xs[s] = randMat(rng, seq, dModel)
			y := tensor.New(tensor.F64, tensor.Shape{1, dModel})
			for c := range dModel {
				y.SetF64(xs[s].AtF64(0, c), 0, c)
			}
			ys[s] = y
		}
		params := append(m.Params(), readout.Params()...)
		opt := nn.NewAdam(params, 0.02)
		lossAt := func() float64 {
			var tot float64
			for s := range nSeq {
				o, _ := m.Forward(backend.NewContext(), xs[s])
				last, _ := backend.Execute(backend.NewContext(), backend.OpSlice, []*tensor.Tensor{o}, backend.SliceAttrs{Axis: 0, Start: seq - 1, End: seq})
				pred, _ := readout.Forward(backend.NewContext(), last[0])
				for c := range dModel {
					d := pred.AtF64(0, c) - ys[s].AtF64(0, c)
					tot += d * d
				}
			}
			return tot / float64(nSeq*dModel)
		}
		first := lossAt()
		for range steps {
			tape := autograd.NewTape()
			ctx := tape.Context()
			var loss *tensor.Tensor
			for s := range nSeq {
				o, err := m.Forward(ctx, xs[s])
				if err != nil {
					t.Fatal(err)
				}
				last, _ := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{o}, backend.SliceAttrs{Axis: 0, Start: seq - 1, End: seq})
				pred, err := readout.Forward(ctx, last[0])
				if err != nil {
					t.Fatal(err)
				}
				diff, _ := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{pred, ys[s]}, nil)
				sq, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
				sm, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{sq[0]}, backend.ReduceAttrs{})
				if loss == nil {
					loss = sm[0]
				} else {
					add, _ := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{loss, sm[0]}, nil)
					loss = add[0]
				}
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(func(p *tensor.Tensor) *tensor.Tensor { return tape.Grad(p) }); err != nil {
				t.Fatal(err)
			}
		}
		return first, lossAt()
	}
	for _, exp := range []bool{false, true} {
		first, last := train(exp)
		name := "sigmoid-forget"
		if exp {
			name = "exp-forget"
		}
		t.Logf("delayed-copy [%s]: loss %.5f → %.5f (%.1f× lower)", name, first, last, first/last)
		if last > first/5 {
			t.Errorf("%s: loss did not fall ≥5× (%.5f → %.5f)", name, first, last)
		}
	}
}

func ExampleMLSTM() {
	// One head, d=1. With ĩ=0 (i=1) and exp-forget f̃=0 (f=1) the cell is
	// unnormalized causal linear attention: state accumulates k·v, read by q,
	// divided by max(|Σ q·k|,1). q=k=1 at both steps, v=(2,3):
	//   t=0: num=1·2=2, denom=max(|1|,1)=1 → 2
	//   t=1: num=1·2+1·3=5, denom=max(|2|,1)=2 → 2.5
	q := toT([][]float64{{1}, {1}})
	k := toT([][]float64{{1}, {1}})
	v := toT([][]float64{{2}, {3}})
	g := toT([][]float64{{0}, {0}}) // ĩ=f̃=0
	out, _ := nn.MLSTMParallel(backend.NewContext(), q, k, v, g, g, true)
	fmt.Printf("%.4g %.4g\n", out.AtF64(0, 0), out.AtF64(1, 0))
	// Output:
	// 2 2.5
}
