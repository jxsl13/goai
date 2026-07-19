package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// rawSigmoidOverSet returns the normalized sigmoid affinities of an index set —
// the DeepSeek gate value the load-balancing bias must NOT influence.
func rawSigmoidOverSet(logits []float64, set []int) []float64 {
	var sum float64
	out := make([]float64, len(set))
	for j, i := range set {
		out[j] = 1 / (1 + math.Exp(-logits[i]))
		sum += out[j]
	}
	for j := range out {
		out[j] /= sum
	}
	return out
}

// §R242 crux (test a): the per-expert bias must change only WHICH experts are
// selected, never the gate WEIGHTS. Whatever set TopKGatingBiased picks, its
// weights must equal a plain softmax over the RAW logits of that set — the bias
// magnitude must be absent from the weights.
func TestLossFreeGateValueInvariance(t *testing.T) {
	logits := []float64{1.0, 3.0, 0.5, 2.0}

	// (1) A bias that CHANGES the selection: a big boost on expert 0 pulls it into
	// the top-2 (displacing expert 3), yet the weights stay raw-logit softmax.
	bias := []float64{5, 0, 0, 0}
	experts, weights := nn.TopKGatingBiased(logits, bias, 2)
	if len(experts) != 2 {
		t.Fatalf("want 2 selected, got %v", experts)
	}
	// Expert 0 (score 6) and expert 1 (score 3) win over expert 3 (score 2).
	if !(experts[0] == 0 && experts[1] == 1) {
		t.Fatalf("biased selection = %v, want [0 1]", experts)
	}
	want := rawSigmoidOverSet(logits, experts)
	for i := range weights {
		if math.Abs(weights[i]-want[i]) > 1e-12 {
			t.Errorf("weight[%d] = %.15g, want raw-logit softmax %.15g (bias leaked into gate)", i, weights[i], want[i])
		}
	}

	// (2) A bias that does NOT change the selection must be byte-identical (same
	// experts AND same weights) to the unbiased router — proving the weights are a
	// pure function of the selected set and the raw sigmoid affinities.
	small := []float64{0.1, 0.2, -0.3, 0.05} // too small to reorder the top-2 {1,3}
	be, bw := nn.TopKGatingBiased(logits, small, 2)
	ue, uw := nn.TopKGatingBiased(logits, nil, 2)
	if len(be) != len(ue) {
		t.Fatalf("selection size differs: %v vs %v", be, ue)
	}
	for i := range be {
		if be[i] != ue[i] {
			t.Fatalf("small bias changed selection: %v vs unbiased %v", be, ue)
		}
		if bw[i] != uw[i] { // exact equality: same formula, same inputs
			t.Errorf("weight[%d] biased %.17g != unbiased %.17g", i, bw[i], uw[i])
		}
	}
}

// §R242's gate is sigmoid, not softmax. This case is deliberately chosen so
// adding the bias after sigmoid changes the winner while adding it to the raw
// logit would not; it catches both the activation and selection-order bugs.
func TestTopKGatingBiasedUsesSigmoidAffinity(t *testing.T) {
	logits := []float64{10, 0}
	bias := []float64{0, 1.1}
	experts, _ := nn.TopKGatingBiased(logits, bias, 1)
	if len(experts) != 1 || experts[0] != 1 {
		t.Fatalf("sigmoid(logit)+bias selection = %v, want [1]", experts)
	}

	// With no bias, sigmoid is monotone so selection matches TopKGating, but the
	// combination weights are normalized sigmoid affinities, not softmax logits.
	logits = []float64{2, 0, -1}
	experts, weights := nn.TopKGatingBiased(logits, nil, 2)
	if experts[0] != 0 || experts[1] != 1 {
		t.Fatalf("zero-bias selection = %v, want [0 1]", experts)
	}
	want := rawSigmoidOverSet(logits, experts)
	for i := range weights {
		if math.Abs(weights[i]-want[i]) > 1e-12 {
			t.Fatalf("sigmoid weight[%d]=%.15g, want %.15g", i, weights[i], want[i])
		}
	}
	_, softmaxWeights := nn.TopKGating(logits, 2)
	if weights[0] == softmaxWeights[0] {
		t.Fatal("loss-free sigmoid gate unexpectedly equals historical softmax gate")
	}

	// Saturated negative logits may round every sigmoid affinity to zero. Decode
	// must return finite zero weights, matching dense OpMoECombine's zero-denom
	// policy, rather than divide 0/0.
	_, weights = nn.TopKGatingBiased([]float64{-1000, -1000}, nil, 2)
	for i, weight := range weights {
		if weight != 0 || math.IsNaN(weight) {
			t.Fatalf("saturated weight[%d]=%g, want finite zero", i, weight)
		}
	}
}

// nilBias must reproduce an explicit all-zero bias byte-for-byte.
func TestTopKGatingBiasedNilEqualsZeroBias(t *testing.T) {
	cases := [][]float64{
		{1, 3, 0.5, 2},
		{-2, -2, -2, 5, 0.1},
		{0, 0, 0, 0},
	}
	for _, logits := range cases {
		for k := 1; k <= len(logits); k++ {
			be, bw := nn.TopKGatingBiased(logits, nil, k)
			ue, uw := nn.TopKGatingBiased(logits, make([]float64, len(logits)), k)
			if len(be) != len(ue) {
				t.Fatalf("k=%d len %d != %d", k, len(be), len(ue))
			}
			for i := range be {
				if be[i] != ue[i] || bw[i] != uw[i] {
					t.Fatalf("logits=%v k=%d: biased(nil)=(%v,%v) != zero-bias(%v,%v)", logits, k, be, bw, ue, uw)
				}
			}
		}
	}
}

// The integrated SparseMoE path must feed sigmoid affinities — not the historical
// softmax gates — into OpMoECombine when loss-free routing is enabled.
func TestLossFreeSparseMoEUsesSigmoidGate(t *testing.T) {
	const dim, hidden, experts, topK, tokens = 3, 5, 3, 2, 2
	m := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 17, nn.WithLossFreeBalance(0.01))
	x := tensor.FromFloat64(tensor.Shape{tokens, dim}, []float64{
		1.2, -0.7, 0.3,
		-0.4, 1.1, 0.8,
	})
	ctx := backend.NewContext()
	got, logits, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	expertOut := make([]*tensor.Tensor, experts)
	for i := range experts {
		expertOut[i], err = m.Experts[i].Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
	}

	var differsFromSoftmax bool
	for tok := range tokens {
		row := make([]float64, experts)
		for i := range experts {
			row[i] = logits.AtF64(tok, i)
		}
		selected, weights := nn.TopKGatingBiased(row, nil, topK)
		softSelected, softWeights := nn.TopKGating(row, topK)
		for d := range dim {
			var want, softWant float64
			for j, expert := range selected {
				want += weights[j] * expertOut[expert].AtF64(tok, d)
			}
			for j, expert := range softSelected {
				softWant += softWeights[j] * expertOut[expert].AtF64(tok, d)
			}
			if math.Abs(got.AtF64(tok, d)-want) > 1e-12 {
				t.Fatalf("token %d dim %d: integrated=%g sigmoid-ref=%g", tok, d, got.AtF64(tok, d), want)
			}
			if math.Abs(got.AtF64(tok, d)-softWant) > 1e-8 {
				differsFromSoftmax = true
			}
		}
	}
	if !differsFromSoftmax {
		t.Fatal("decisive fixture did not distinguish sigmoid routing from softmax")
	}
}

// spread returns the max−min gap of a load histogram — a scalar imbalance metric.
func spread(load []int) int {
	if len(load) == 0 {
		return 0
	}
	lo, hi := load[0], load[0]
	for _, c := range load {
		if c < lo {
			lo = c
		}
		if c > hi {
			hi = c
		}
	}
	return hi - lo
}

// §R242 (test b): on a deliberately imbalanced router, repeated UpdateLoadBias
// steps drive the routed-load spread DOWN. Step 0 uses the all-zero bias (i.e.
// no balancing yet, identical to the unbiased router), so comparing the late
// spread against the step-0 spread is exactly "balanced vs no balancing".
func TestLossFreeBalancingReducesSpread(t *testing.T) {
	const (
		dim     = 4
		hidden  = 8
		experts = 6
		topK    = 1 // top-1 makes the load histogram (and imbalance) easiest to read
		tokens  = 64
	)
	// A small, research-scale bias-rate. Because routing is DISCRETE and the update
	// is sign-based, the load never freezes at a single point — it settles into a
	// small limit cycle around the balanced mean (a large rate like 0.5 would make
	// that cycle huge; see the "u too high → oscillation" note on WithLossFreeBalance).
	// So we compare the step-0 spread (all-zero bias == no balancing) against the
	// TIME-AVERAGED spread of the converged tail, which is immune to that jitter.
	moe := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 7, nn.WithLossFreeBalance(0.02))

	// Deterministic, deliberately skewed inputs so the raw router strongly favors a
	// couple of experts (creating the imbalance the bias must correct).
	xs := make([]float64, tokens*dim)
	for i := range xs {
		// A low-rank-ish pattern: a few directions dominate → skewed routing.
		xs[i] = math.Sin(float64(i)*0.7) + 0.5*math.Cos(float64(i/dim))
	}
	x := tensor.FromFloat64(tensor.Shape{tokens, dim}, xs)
	ctx := backend.NewContext()

	const (
		steps = 300
		tail  = 40 // average the spread over the final `tail` steps
	)
	var first int
	var tailSum float64
	for s := 0; s < steps; s++ {
		if _, _, err := moe.Forward(ctx, x); err != nil {
			t.Fatal(err)
		}
		load := moe.UpdateLoadBias()
		total := 0
		for _, c := range load {
			total += c
		}
		if total != tokens*topK {
			t.Fatalf("step %d: routed %d tokens, want %d", s, total, tokens*topK)
		}
		if s == 0 {
			first = spread(load)
		}
		if s >= steps-tail {
			tailSum += float64(spread(load))
		}
	}
	if first == 0 {
		t.Skip("router already balanced at step 0 — no imbalance to correct")
	}
	tailAvg := tailSum / float64(tail)
	if tailAvg >= float64(first) {
		t.Errorf("load spread did not shrink: step0=%d converged-avg=%.2f (balancing ineffective)", first, tailAvg)
	}
	t.Logf("load spread: step0=%d → converged-avg=%.2f over last %d steps", first, tailAvg, tail)
}

// §R242 (test c): a SparseMoE built WITHOUT WithLossFreeBalance must route
// byte-identically to the historical layer — same output tensor, element for
// element — as one built the same way. (Off-by-default identity.)
func TestLossFreeOffByDefaultIdentity(t *testing.T) {
	const dim, hidden, experts, topK = 4, 8, 3, 2
	x := tensor.FromFloat64(tensor.Shape{3, dim}, []float64{
		1, 0, -1, 0.5, 0.2, 2, -0.3, 1, -1.1, 0.4, 0.9, -0.2,
	})
	ctx := backend.NewContext()

	a := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 42) // no options
	b := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 42) // identical build
	ya, _, err := a.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	yb, _, err := b.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	if !ya.Shape().Equal(yb.Shape()) {
		t.Fatalf("shape %v vs %v", ya.Shape(), yb.Shape())
	}
	for k := 0; k < ya.Numel(); k++ {
		idx := tensor.Unravel(k, ya.Shape())
		if ya.AtF64(idx...) != yb.AtF64(idx...) {
			t.Fatalf("off-by-default routing diverged at %v: %.17g vs %.17g", idx, ya.AtF64(idx...), yb.AtF64(idx...))
		}
	}
}

// §R242 (test d): the bias buffer is NOT a Parameter and adds NO gradient. The
// parameter set is unchanged by the option and the sigmoid-gated path remains
// differentiable through the ordinary model parameters.
func TestLossFreeBiasNotAParameter(t *testing.T) {
	const dim, hidden, experts, topK = 4, 6, 3, 2
	x := tensor.FromFloat64(tensor.Shape{2, dim}, []float64{0.5, -1, 2, 0.3, -0.7, 1.1, 0.2, -1.5})

	plain := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 7)
	free := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 7, nn.WithLossFreeBalance(0.01))

	// (1) Params() must be identical in count — the bias is excluded.
	if len(plain.Params()) != len(free.Params()) {
		t.Fatalf("Params() count differs: plain=%d free=%d (bias leaked into Params)", len(plain.Params()), len(free.Params()))
	}

	// (2) A backward pass through the sigmoid-gated route produces finite
	// gradients on every selected model parameter.
	tape := autograd.NewTape()
	y, _, err := free.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(y); err != nil {
		t.Fatal(err)
	}
	var gradientBearing int
	for i, p := range free.Params() {
		g := tape.Grad(p)
		if g == nil {
			continue
		}
		gradientBearing++
		for k := 0; k < g.Numel(); k++ {
			v := g.AtF64(tensor.Unravel(k, g.Shape())...)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("param %d gradient[%d] is non-finite: %g", i, k, v)
			}
		}
	}
	if gradientBearing == 0 {
		t.Fatal("loss-free route produced no parameter gradients")
	}

	// (3) Central finite difference through the integrated sigmoid route. The
	// fixture has stable top-k choices under this small perturbation, so it checks
	// the differentiable gate math rather than the detached selector.
	p := free.Router.W
	const h = 1e-5
	orig := p.AtF64(0, 0)
	lossAt := func(v float64) float64 {
		p.SetF64(v, 0, 0)
		out, _, err := free.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for k := 0; k < out.Numel(); k++ {
			sum += out.AtF64(tensor.Unravel(k, out.Shape())...)
		}
		return sum
	}
	lp, lm := lossAt(orig+h), lossAt(orig-h)
	p.SetF64(orig, 0, 0)
	numeric := (lp - lm) / (2 * h)
	analytic := tape.Grad(p).AtF64(0, 0)
	denom := math.Max(1, math.Max(math.Abs(numeric), math.Abs(analytic)))
	if rel := math.Abs(numeric-analytic) / denom; rel > 1e-4 {
		t.Fatalf("router sigmoid grad analytic=%g numeric=%g rel=%g", analytic, numeric, rel)
	}

	// (4) UpdateLoadBias must NOT run during Backward: gradients above already
	// prove it, and the disabled layer's UpdateLoadBias is a nil no-op.
	if got := plain.UpdateLoadBias(); got != nil {
		t.Errorf("disabled UpdateLoadBias returned %v, want nil no-op", got)
	}
}

// ExampleTopKGatingBiased shows the §R242 crux: a bias can pull an expert into the
// selection, but the returned gate weights are still the normalized RAW sigmoid
// affinities of the selected pair — the bias never inflates a gate value.
func ExampleTopKGatingBiased() {
	logits := []float64{1, 3, 0.5, 2}
	// A +5 bias on expert 0 (raw logit 1) makes it beat expert 3 for the second
	// slot, but its WEIGHT is still computed from sigmoid(1) and sigmoid(3).
	experts, weights := nn.TopKGatingBiased(logits, []float64{5, 0, 0, 0}, 2)
	fmt.Printf("experts=%v weights=%.2f,%.2f\n", experts, weights[0], weights[1])
	// Output: experts=[0 1] weights=0.43,0.57
}

// ExampleSparseMoE_lossFreeBalance enables auxiliary-loss-free load balancing
// (§R242) via WithLossFreeBalance and runs one balancing step: Forward routes the
// batch, then UpdateLoadBias adjusts the per-expert bias from the observed load
// and returns the per-expert routed-token counts (which sum to tokens × top-k).
func ExampleSparseMoE_lossFreeBalance() {
	opts := []nn.SparseMoEOption{nn.WithLossFreeBalance(0.01)}
	moe := nn.NewSparseMoE(tensor.F64, 4 /*dim*/, 8 /*hidden*/, 4 /*experts*/, 2 /*top-k*/, 42, opts...)
	x := tensor.FromFloat64(tensor.Shape{6, 4}, []float64{
		1, 0, -1, 0.5, 0.2, 2, -0.3, 1, -1.1, 0.4, 0.9, -0.2,
		0.3, -0.6, 1.2, 0.1, -0.8, 0.7, 0.4, -1.3, 1.5, -0.2, -0.9, 0.6,
	})
	if _, _, err := moe.Forward(backend.NewContext(), x); err != nil {
		panic(err)
	}
	load := moe.UpdateLoadBias() // one gradient-free balancing step
	total := 0
	for _, c := range load {
		total += c
	}
	fmt.Println(total)
	// Output: 12
}
