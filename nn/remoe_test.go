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

// §V2 + §V16 (THE anchor): ReMoE's whole point is that the ROUTING is
// differentiable — central finite differences of (masked output + L1 penalty)
// w.r.t. EVERY parameter (router W and B, all three matrices of every expert)
// and the input match the tape gradients, with NO straight-through estimator.
// The router weights must receive a non-zero gradient (a top-k router cannot —
// its selection is detached).
func TestReMoEGradcheck(t *testing.T) {
	const tks, d, e, hidden = 3, 4, 3, 5
	rng := rand.New(rand.NewPCG(11, 13))
	m := nn.NewReMoE(tensor.F64, d, hidden, e, 2, 21, nn.WithReMoELambda(0.05), nn.WithReMoEAdaptRate(0))
	x := randMat(rng, tks, d)
	mask := randMat(rng, tks, d)

	// FD needs the ReLU kinks away from 0: perturbing a weight by h=1e-6 moves a
	// logit by O(h); any logit within 1e-4 of 0 would make the FD invalid.
	logits, err := m.Router.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for tk := range tks {
		for i := range e {
			if v := math.Abs(logits.AtF64(tk, i)); v < 1e-4 {
				t.Fatalf("seed places router logit [%d,%d]=%g too close to the ReLU kink; pick another seed", tk, i, v)
			}
		}
	}

	lossOf := func() float64 {
		y, _, pen, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		s := pen.AtF64()
		for i := range tks {
			for k := range d {
				s += mask.AtF64(i, k) * y.AtF64(i, k)
			}
		}
		return s
	}

	tape := autograd.NewTape()
	y, _, pen, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	pr, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{y, mask}, nil)
	sum, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pr[0]}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpAdd, []*tensor.Tensor{sum[0], pen}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}

	const h = 1e-6
	entries, maxRel := 0, 0.0
	check := func(name string, param *tensor.Tensor) {
		g := tape.Grad(param)
		if g == nil {
			t.Fatalf("%s: nil gradient — routing not differentiable", name)
		}
		for i := range param.Numel() {
			idx := tensor.Unravel(i, param.Shape())
			o := param.AtF64(idx...)
			param.SetF64(o+h, idx...)
			lp := lossOf()
			param.SetF64(o-h, idx...)
			lm := lossOf()
			param.SetF64(o, idx...)
			num, ana := (lp-lm)/(2*h), g.AtF64(idx...)
			rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
			maxRel = math.Max(maxRel, rel)
			entries++
			if rel > 1e-4 {
				t.Errorf("%s grad[%v]: numeric %.8g vs analytic %.8g", name, idx, num, ana)
			}
		}
	}
	check("x", x)
	check("Router.W", m.Router.W)
	check("Router.B", m.Router.B)
	for i, ex := range m.Experts {
		check(fmt.Sprintf("expert%d.Wgate", i), ex.Wgate)
		check(fmt.Sprintf("expert%d.Wup", i), ex.Wup)
		check(fmt.Sprintf("expert%d.Wdown", i), ex.Wdown)
	}
	// the differentiable-routing claim: non-zero gradient REACHES the router
	gw := tape.Grad(m.Router.W)
	var gmax float64
	for i := range gw.Numel() {
		idx := tensor.Unravel(i, gw.Shape())
		gmax = math.Max(gmax, math.Abs(gw.AtF64(idx...)))
	}
	if gmax == 0 {
		t.Fatal("router weights received an all-zero gradient — ReLU routing must carry gradient")
	}
	t.Logf("gradcheck: %d entries, max rel err %.3g, max |dL/dRouter.W| = %.3g", entries, maxRel, gmax)
}

// The ReLU gate is EXACTLY zero wherever the router logit is ≤ 0 and equals the
// logit where it is positive — the mechanism that makes the per-token expert
// count variable and the unused experts' contribution exactly nil.
func TestReMoEGateExactZeros(t *testing.T) {
	const tks, d, e, hidden = 5, 3, 4, 4
	rng := rand.New(rand.NewPCG(2, 7))
	m := nn.NewReMoE(tensor.F64, d, hidden, e, 2, 5)
	x := randMat(rng, tks, d)

	logits, err := m.Router.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	_, gates, _, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	pos, neg := 0, 0
	for tk := range tks {
		for i := range e {
			l, g := logits.AtF64(tk, i), gates.AtF64(tk, i)
			if l <= 0 {
				neg++
				if g != 0 {
					t.Errorf("gate[%d,%d] = %g for logit %g ≤ 0; want exactly 0", tk, i, g, l)
				}
			} else {
				pos++
				if g != l {
					t.Errorf("gate[%d,%d] = %g, want the positive logit %g", tk, i, g, l)
				}
			}
		}
	}
	if pos == 0 || neg == 0 {
		t.Fatalf("degenerate seed: want both positive (%d) and non-positive (%d) logits", pos, neg)
	}
}

// remoeToyTask builds a deterministic toy regression x[T,d] → target[T,d].
func remoeToyTask(tks, d int, seed uint64) (x, target *tensor.Tensor) {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	x = randMat(rng, tks, d)
	target = tensor.New(tensor.F64, tensor.Shape{tks, d})
	for i := range tks {
		for k := range d {
			target.SetF64(math.Sin(x.AtF64(i, k))+0.5*x.AtF64(i, (k+1)%d), i, k)
		}
	}
	return x, target
}

// remoeTrain runs `steps` Adam steps of MSE+penalty on the toy task, calling
// UpdateLambda once per step (with δ=0 that only resets the accumulator); it
// returns the final MSE and the average active-expert count measured on a
// fresh forward pass.
func remoeTrain(t *testing.T, m *nn.ReMoE, x, target *tensor.Tensor, steps int, lr float64) (finalLoss, avgActive float64) {
	t.Helper()
	opt := nn.NewAdam(m.Params(), lr)
	for range steps {
		tape := autograd.NewTape()
		ctx := tape.Context()
		y, _, pen, err := m.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		mse, err := nn.MSE(ctx, y, target)
		if err != nil {
			t.Fatal(err)
		}
		total, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{mse, pen}, nil)
		if err != nil {
			t.Fatal(err)
		}
		finalLoss = mse.AtF64()
		if err := tape.Backward(total[0]); err != nil {
			t.Fatal(err)
		}
		m.UpdateLambda()
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	_, gates, _, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	tks, e := gates.Shape()[0], gates.Shape()[1]
	active := 0
	for i := range tks {
		for k := range e {
			if gates.AtF64(i, k) > 0 {
				active++
			}
		}
	}
	return finalLoss, float64(active) / float64(tks)
}

// remoeAvgActive measures the per-token average number of strictly-positive
// gates for the current router state.
func remoeAvgActive(t *testing.T, m *nn.ReMoE, x *tensor.Tensor) float64 {
	t.Helper()
	_, gates, _, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	tks, e := gates.Shape()[0], gates.Shape()[1]
	active := 0
	for i := range tks {
		for k := range e {
			if gates.AtF64(i, k) > 0 {
				active++
			}
		}
	}
	return float64(active) / float64(tks)
}

// Sparsity control, part 1: with the MSE objective removed and ONLY the L1 gate
// penalty driving training (the controller frozen, δ=0), raising the fixed λ
// monotonically reduces the average number of active experts — the isolated
// proof that λ is the sparsity knob the paper claims it is. Starting from a
// dense router (all gates open), each SGD step lowers every active logit by
// exactly λ/T·lr, so a larger λ crosses more logits below zero in the same
// number of steps → strictly fewer active experts. (Training the full MSE+L1
// objective couples sparsity to the fit and blurs the clean monotone signal;
// the e2e test covers the joint dynamics.)
func TestReMoELambdaMonotoneSparsity(t *testing.T) {
	const tks, d, e, hidden, steps = 16, 4, 4, 8, 60
	rng := rand.New(rand.NewPCG(3, 4))
	x := randMat(rng, tks, d)
	lambdas := []float64{0, 0.05, 0.2, 1.0}
	actives := make([]float64, len(lambdas))
	for i, l := range lambdas {
		m := nn.NewReMoE(tensor.F64, d, hidden, e, 2, 17,
			nn.WithReMoELambda(l), nn.WithReMoEAdaptRate(0))
		// dense warm-start: bias every gate open so all runs begin identical and
		// the only difference across runs is λ.
		for j := range e {
			m.Router.B.SetF64(1.0, j)
		}
		opt := nn.NewSGD(m.Router.Params(), 0.05, 0) // train ONLY the router, L1-only loss
		for range steps {
			tape := autograd.NewTape()
			_, _, pen, err := m.Forward(tape.Context(), x)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(pen); err != nil {
				t.Fatal(err)
			}
			m.UpdateLambda() // δ=0 ⇒ only resets the accumulator; λ stays fixed
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
		actives[i] = remoeAvgActive(t, m, x)
		if got := m.Lambda(); got != l {
			t.Errorf("frozen λ drifted: %g → %g", l, got)
		}
	}
	t.Logf("λ %v → avg active experts %v (of %d)", lambdas, actives, e)
	for i := 1; i < len(actives); i++ {
		if actives[i] > actives[i-1] {
			t.Errorf("active experts not monotone in λ: λ=%g→%.3f but λ=%g→%.3f",
				lambdas[i-1], actives[i-1], lambdas[i], actives[i])
		}
	}
	if actives[len(actives)-1] >= actives[0] {
		t.Errorf("largest λ (%g) must be strictly sparser than λ=0: %.3f vs %.3f",
			lambdas[len(lambdas)-1], actives[len(actives)-1], actives[0])
	}
}

// Sparsity control, part 2: the adaptive controller drives the average active-
// expert count toward TargetK over training, raising λ from a tiny dense-warm-up
// value on its own (the paper's λ thermostat).
func TestReMoEAdaptiveControllerConvergence(t *testing.T) {
	const tks, d, e, hidden, steps, targetK = 16, 4, 4, 8, 400, 1.5
	x, target := remoeToyTask(tks, d, 3)
	m := nn.NewReMoE(tensor.F64, d, hidden, e, targetK, 17,
		nn.WithReMoELambda(1e-6), nn.WithReMoEAdaptRate(0.1)) // dense warm-up start
	lambda0 := m.Lambda()

	// starting point: near-dense routing (Xavier router ⇒ ~half the gates open;
	// far above targetK is not guaranteed, but λ must move if avg ≠ target).
	_, avg := remoeTrain(t, m, x, target, steps, 1e-2)
	t.Logf("controller: target %.2f, achieved avg active %.3f, λ %g → %g",
		targetK, avg, lambda0, m.Lambda())
	if math.Abs(avg-targetK) > 0.75 {
		t.Errorf("adaptive λ failed to steer sparsity: avg active %.3f, target %.2f (±0.75)", avg, targetK)
	}
	if m.Lambda() <= lambda0 {
		t.Errorf("starting dense, the controller must RAISE λ: %g → %g", lambda0, m.Lambda())
	}
}

// Collapse sanity: with a single expert and a router forced to a constant
// positive logit c, ReMoE reduces exactly to c·FFN(x); with all logits negative
// the output is exactly zero (the documented degenerate) and the penalty is 0.
func TestReMoECollapseAndAllNegative(t *testing.T) {
	const tks, d, hidden = 4, 3, 6
	rng := rand.New(rand.NewPCG(8, 3))
	x := randMat(rng, tks, d)

	// forced-positive constant gate: W=0, B=c ⇒ gate ≡ c ⇒ y = c·Expert(x)
	const c = 1.7
	m := nn.NewReMoE(tensor.F64, d, hidden, 1, 1, 9)
	nn.Zeros(m.Router.W)
	m.Router.B.SetF64(c, 0)
	y, _, _, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Experts[0].Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range tks {
		for k := range d {
			if got, w := y.AtF64(i, k), c*want.AtF64(i, k); math.Abs(got-w) > 1e-12*math.Max(1, math.Abs(w)) {
				t.Errorf("collapse: y[%d,%d] = %.15g, want c·FFN = %.15g", i, k, got, w)
			}
		}
	}

	// all-negative logits: W=0, B=-1 ⇒ every gate 0 ⇒ y ≡ 0, penalty = 0
	m.Router.B.SetF64(-1, 0)
	y0, g0, p0, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range tks {
		if g0.AtF64(i, 0) != 0 {
			t.Errorf("gate[%d] = %g, want exactly 0", i, g0.AtF64(i, 0))
		}
		for k := range d {
			if y0.AtF64(i, k) != 0 {
				t.Errorf("all-negative router must yield y[%d,%d] = 0, got %g", i, k, y0.AtF64(i, k))
			}
		}
	}
	if p0.AtF64() != 0 {
		t.Errorf("all-negative router must yield penalty 0, got %g", p0.AtF64())
	}
}

// Shapes and determinism: identical construction + input ⇒ identical outputs;
// output/gate/penalty shapes are as documented.
func TestReMoEShapesDeterminism(t *testing.T) {
	const tks, d, e, hidden = 6, 5, 3, 7
	rng := rand.New(rand.NewPCG(1, 9))
	x := randMat(rng, tks, d)
	a := nn.NewReMoE(tensor.F64, d, hidden, e, 2, 33)
	b := nn.NewReMoE(tensor.F64, d, hidden, e, 2, 33)
	ya, ga, pa, err := a.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	yb, _, pb, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !ya.Shape().Equal(tensor.Shape{tks, d}) || !ga.Shape().Equal(tensor.Shape{tks, e}) {
		t.Fatalf("shapes y=%v gates=%v, want [%d,%d] and [%d,%d]", ya.Shape(), ga.Shape(), tks, d, tks, e)
	}
	if pa.Numel() != 1 {
		t.Fatalf("penalty must be scalar, got shape %v", pa.Shape())
	}
	if pa.AtF64() != pb.AtF64() {
		t.Errorf("penalty not deterministic: %g vs %g", pa.AtF64(), pb.AtF64())
	}
	for i := range tks {
		for k := range d {
			if ya.AtF64(i, k) != yb.AtF64(i, k) {
				t.Errorf("y[%d,%d] not deterministic: %g vs %g", i, k, ya.AtF64(i, k), yb.AtF64(i, k))
			}
		}
	}
	// rank/width validation
	if _, _, _, err := a.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, d + 1})); err == nil {
		t.Error("wrong input width must error")
	}
}

// §V16 e2e: a tiny ReMoE block trains — MSE decreases well below its starting
// value while the adaptive controller holds the routing sparse.
func TestReMoETrainsE2E(t *testing.T) {
	const tks, d, e, hidden, steps, targetK = 16, 4, 4, 8, 300, 2
	x, target := remoeToyTask(tks, d, 12)
	m := nn.NewReMoE(tensor.F64, d, hidden, e, targetK, 77)

	y0, _, _, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss0, err := nn.MSE(backend.NewContext(), y0, target)
	if err != nil {
		t.Fatal(err)
	}
	final, avg := remoeTrain(t, m, x, target, steps, 1e-2)
	t.Logf("e2e: MSE %.4f → %.4f in %d steps; avg active experts %.3f (target %d of %d), λ=%g",
		loss0.AtF64(), final, steps, avg, targetK, e, m.Lambda())
	if final >= loss0.AtF64()*0.5 {
		t.Errorf("training failed: MSE %.4f → %.4f (want < half)", loss0.AtF64(), final)
	}
}

func ExampleReMoE() {
	// 4 experts, target ~2 active per token; the ReLU router is trained by
	// ordinary backprop and the L1 penalty (already scaled by λ) is simply
	// added to the loss. UpdateLambda is the per-batch sparsity thermostat.
	m := nn.NewReMoE(tensor.F64, 3, 6, 4, 2, 42)
	x := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{1, 0, -1, 0.5, 2, 0.3, -1, 1, 0.2, 0.7, -0.5, 1.5})
	y, gates, penalty, _ := m.Forward(backend.NewContext(), x)
	avgActive, lambda := m.UpdateLambda()
	fmt.Println("y:", y.Shape(), "gates:", gates.Shape())
	fmt.Println("penalty is scalar:", penalty.Numel() == 1, "λ:", lambda == m.Lambda())
	fmt.Println("avg active within [0,4]:", avgActive >= 0 && avgActive <= 4)
	fmt.Println("params:", len(m.Params()))
	// Output:
	// y: (4, 3) gates: (4, 4)
	// penalty is scalar: true λ: true
	// avg active within [0,4]: true
	// params: 14
}
