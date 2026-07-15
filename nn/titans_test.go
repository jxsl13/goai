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
)

// titansZeroCol returns a zero [seq,1] gate column.
func titansZeroCol(seq int) *tensor.Tensor {
	return tensor.New(tensor.F64, tensor.Shape{seq, 1})
}

// COLLAPSE TEST (pins the hand-derived linear inner gradient): with a LINEAR
// memory (M0 = 0), zero momentum (η = 0) and zero forget (α = 0), the Titans
// update M_t = M_{t-1} − θ_t·2(M_{t-1}k−v)kᵀ IS the delta rule with writing
// strength β = 2θ. The full output trajectory must match nn.DeltaNet on the
// same q/k/v to 1e-12 (task bar 1e-8) — if the hand-derived ∇_M ℓ were wrong
// in sign, scale, or orientation, every token after the first would diverge.
func TestTitansLinearCollapsesToDeltaRule(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 21))
	const seq, dim = 6, 5
	q, k, v := randMat(rng, seq, dim), randMat(rng, seq, dim), randMat(rng, seq, dim)
	bvals, beta := randBeta(rng, seq) // β ∈ (0,1)
	theta := tensor.New(tensor.F64, tensor.Shape{seq, 1})
	for i, b := range bvals {
		theta.SetF64(b/2, i, 0) // θ = β/2 (exact in floating point)
	}

	mem, err := nn.NewNeuralMemory(tensor.F64, dim, 0, 3, nn.WithTitansLinearMemory())
	if err != nil {
		t.Fatal(err)
	}
	got, err := mem.Scan(backend.NewContext(), q, k, v, titansZeroCol(seq), theta, titansZeroCol(seq))
	if err != nil {
		t.Fatal(err)
	}
	want, err := nn.DeltaNet(backend.NewContext(), q, k, v, beta)
	if err != nil {
		t.Fatal(err)
	}
	var maxDiff float64
	for i := range seq {
		for j := range dim {
			if d := math.Abs(got.AtF64(i, j) - want.AtF64(i, j)); d > maxDiff {
				maxDiff = d
			}
		}
	}
	t.Logf("Titans(linear, η=0, α=0, θ=β/2) vs DeltaNet: max |diff| = %.3g", maxDiff)
	if maxDiff > 1e-12 {
		t.Fatalf("linear Titans does not collapse to the delta rule: max diff %.3g > 1e-12", maxDiff)
	}
}

// titansGradcheck runs central-difference gradcheck for every named parameter
// of a forward closure, asserting a non-nil, not-all-zero analytic gradient
// and numeric agreement within 1e-4·max(1,|num|).
func titansGradcheck(t *testing.T, params map[string]*tensor.Tensor,
	forward func(ctx *backend.Context) (*tensor.Tensor, error)) {
	t.Helper()
	tape := autograd.NewTape()
	out, err := forward(tape.Context())
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
	sumLoss := func() float64 {
		o, err := forward(backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	const h = 1e-6
	for name, p := range params {
		g := tape.Grad(p)
		if g == nil {
			t.Errorf("%s: nil grad — no gradient reached this parameter", name)
			continue
		}
		var nz bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", name)
		}
	}
}

// GRADCHECK: the outer tape differentiates THROUGH the hand-derived inner
// gradient (ADR-0022 option 1 — no second-order autograd). Every learnable
// parameter of the full MAC block (deep memory + persistent tokens) and of a
// standalone linear-memory module gets a numeric check: k/v/q projections, the
// η/θ/α gate weights, the initial memory (W1/W2 deep, M0 linear), the
// attention in/out projections and the persistent tokens.
func TestTitansGradcheck(t *testing.T) {
	t.Run("MACBlockDeep", func(t *testing.T) {
		const dim, heads, hidden, seq = 4, 2, 3, 3
		b, err := nn.NewTitans(tensor.F64, dim, heads, hidden, 5, nn.WithTitansPersistentTokens(2))
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewPCG(11, 3))
		x := randMat(rng, seq, dim)
		titansGradcheck(t, map[string]*tensor.Tensor{
			"Memory.WK":     b.Memory.WK,
			"Memory.WV":     b.Memory.WV,
			"Memory.WQ":     b.Memory.WQ,
			"Memory.WEta":   b.Memory.WEta,
			"Memory.WTheta": b.Memory.WTheta,
			"Memory.WAlpha": b.Memory.WAlpha,
			"Memory.W1":     b.Memory.W1,
			"Memory.W2":     b.Memory.W2,
			"NormMem.Gamma": b.NormMem.Gamma,
			"InProj.W":      b.InProj.W,
			"InProj.B":      b.InProj.B,
			"OutProj.W":     b.OutProj.W,
			"OutProj.B":     b.OutProj.B,
			"Persistent":    b.Persistent,
		}, func(ctx *backend.Context) (*tensor.Tensor, error) { return b.Forward(ctx, x) })
	})
	t.Run("NeuralMemoryLinear", func(t *testing.T) {
		const dim, seq = 3, 3
		m, err := nn.NewNeuralMemory(tensor.F64, dim, 0, 9, nn.WithTitansLinearMemory())
		if err != nil {
			t.Fatal(err)
		}
		// non-zero M0 so its gradient path is exercised away from the zero init.
		rng := rand.New(rand.NewPCG(13, 4))
		for i := range dim {
			for j := range dim {
				m.M0.SetF64(0.3*rng.NormFloat64(), i, j)
			}
		}
		x := randMat(rng, seq, dim)
		titansGradcheck(t, map[string]*tensor.Tensor{
			"WK":     m.WK,
			"WV":     m.WV,
			"WQ":     m.WQ,
			"WEta":   m.WEta,
			"WTheta": m.WTheta,
			"WAlpha": m.WAlpha,
			"M0":     m.M0,
		}, func(ctx *backend.Context) (*tensor.Tensor, error) { return m.Forward(ctx, x) })
	})
}

// F32 PARITY: with weights copied from the F64 block, the F32 forward matches
// the F64 forward to ~1e-4 relative.
func TestTitansF32Parity(t *testing.T) {
	const dim, heads, hidden, seq = 4, 2, 3, 4
	b64, err := nn.NewTitans(tensor.F64, dim, heads, hidden, 17, nn.WithTitansPersistentTokens(2))
	if err != nil {
		t.Fatal(err)
	}
	b32, err := nn.NewTitans(tensor.F32, dim, heads, hidden, 17, nn.WithTitansPersistentTokens(2))
	if err != nil {
		t.Fatal(err)
	}
	p64, p32 := b64.Params(), b32.Params()
	if len(p64) != len(p32) {
		t.Fatalf("param count mismatch %d vs %d", len(p64), len(p32))
	}
	for i := range p64 {
		for idx := range p64[i].Numel() {
			c := tensor.Unravel(idx, p64[i].Shape())
			p32[i].SetF64(p64[i].AtF64(c...), c...)
		}
	}
	rng := rand.New(rand.NewPCG(19, 6))
	x64 := randMat(rng, seq, dim)
	x32 := tensor.New(tensor.F32, tensor.Shape{seq, dim})
	for i := range seq {
		for j := range dim {
			x32.SetF64(x64.AtF64(i, j), i, j)
		}
	}
	o64, err := b64.Forward(backend.NewContext(), x64)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := b32.Forward(backend.NewContext(), x32)
	if err != nil {
		t.Fatal(err)
	}
	for i := range seq {
		for j := range dim {
			a, b := o64.AtF64(i, j), o32.AtF64(i, j)
			if math.Abs(a-b) > 1e-4*math.Max(1, math.Abs(a)) {
				t.Errorf("out[%d,%d]: f64 %.8g vs f32 %.8g", i, j, a, b)
			}
		}
	}
}

// MEMORIZATION (the point of a long-term memory): write an association
// (k₀ → v₀) early, push several distractor associations through the memory,
// then query k₀ much later WITHOUT writing (θ = 0 on the probe token). The
// retrieval must recover v₀ — exactly for the linear memory over orthonormal
// keys, and by clear cosine preference over every distractor value for the
// deep MLP memory.
func TestTitansMemorizes(t *testing.T) {
	const dim, seq = 8, 10
	rng := rand.New(rand.NewPCG(23, 9))
	basis := func(i int) []float64 {
		r := make([]float64, dim)
		r[i] = 1
		return r
	}
	randVal := func() []float64 {
		r := make([]float64, dim)
		for i := range r {
			r[i] = rng.NormFloat64()
		}
		return r
	}
	va := randVal() // the early association's value
	var kRows, vRows, qRows [][]float64
	distractors := make([][]float64, 0, 6)
	thetaVals := make([]float64, seq)
	for tk := 0; tk < 3; tk++ { // t = 0..2: write (e0 → va) repeatedly
		kRows, vRows, qRows = append(kRows, basis(0)), append(vRows, va), append(qRows, basis(0))
		thetaVals[tk] = 0.5
	}
	for tk := 3; tk < 9; tk++ { // t = 3..8: six distractor associations on other keys
		dv := randVal()
		distractors = append(distractors, dv)
		kRows, vRows, qRows = append(kRows, basis(tk-2)), append(vRows, dv), append(qRows, basis(tk-2))
		thetaVals[tk] = 0.5
	}
	// t = 9: probe — fresh key, θ = 0 (no write), query the EARLY key e0.
	kRows, vRows, qRows = append(kRows, basis(7)), append(vRows, make([]float64, dim)), append(qRows, basis(0))
	thetaVals[9] = 0

	q, k, v := toT(qRows), toT(kRows), toT(vRows)
	theta := betaCol(thetaVals)
	cos := func(a []float64, b *tensor.Tensor, row int) float64 {
		var dot, na, nb float64
		for j := range a {
			bj := b.AtF64(row, j)
			dot += a[j] * bj
			na += a[j] * a[j]
			nb += bj * bj
		}
		return dot / math.Sqrt(na*nb+1e-30)
	}

	t.Run("Linear", func(t *testing.T) {
		m, err := nn.NewNeuralMemory(tensor.F64, dim, 0, 31, nn.WithTitansLinearMemory())
		if err != nil {
			t.Fatal(err)
		}
		out, err := m.Scan(backend.NewContext(), q, k, v, titansZeroCol(seq), theta, titansZeroCol(seq))
		if err != nil {
			t.Fatal(err)
		}
		// orthonormal keys + β = 2θ = 1 writes ⇒ exact recall of va at the probe.
		for j := range dim {
			if d := math.Abs(out.AtF64(9, j) - va[j]); d > 1e-9 {
				t.Errorf("probe[%d] = %.8g, want va = %.8g (diff %.3g)", j, out.AtF64(9, j), va[j], d)
			}
		}
	})
	t.Run("Deep", func(t *testing.T) {
		// Three associations written round-robin (the early key keeps
		// repeating while OTHER writes interleave), then a probe queries the
		// first key without writing. The MLP memory is given a
		// well-separated key code (strong ±2 entries in W1 — Xavier-scale z
		// keeps every σ near 0.5, making all keys look alike) and a
		// GD-stable surprise lr θ = 0.1 (θ ≥ 1/‖h‖² oscillates).
		const hidden, epochs = 12, 4
		m, err := nn.NewNeuralMemory(tensor.F64, dim, hidden, 31)
		if err != nil {
			t.Fatal(err)
		}
		wrng := rand.New(rand.NewPCG(41, 5))
		for i := range hidden {
			for j := range dim {
				sign := 1.0
				if wrng.IntN(2) == 0 {
					sign = -1
				}
				m.W1.SetF64(2*sign, i, j)
			}
		}
		vals := [][]float64{va, randVal(), randVal()}
		var dk, dv, dq [][]float64
		n := 3 * epochs
		th := make([]float64, n+1)
		for tk := range n {
			a := tk % 3
			dk, dv, dq = append(dk, basis(a)), append(dv, vals[a]), append(dq, basis(a))
			th[tk] = 0.1
		}
		// probe: fresh key, θ = 0 (no write), query the FIRST key.
		dk, dv, dq = append(dk, basis(7)), append(dv, make([]float64, dim)), append(dq, basis(0))
		out, err := m.Scan(backend.NewContext(), toT(dq), toT(dk), toT(dv),
			titansZeroCol(n+1), betaCol(th), titansZeroCol(n+1))
		if err != nil {
			t.Fatal(err)
		}
		cva := cos(va, out, n)
		t.Logf("deep probe: cos(out, va) = %.4f, cos(out, other) = %.4f / %.4f",
			cva, cos(vals[1], out, n), cos(vals[2], out, n))
		if cva <= 0.3 {
			t.Errorf("probe not correlated with the memorized value: cos = %.4f", cva)
		}
		for i, ov := range vals[1:] {
			if cd := cos(ov, out, n); cd >= cva {
				t.Errorf("other association %d preferred: cos %.4f >= cos(va) %.4f", i+1, cd, cva)
			}
		}
	})
}

// SHAPES + ERRORS: constructor validation and Forward/Scan input checking.
func TestTitansShapesErrors(t *testing.T) {
	if _, err := nn.NewTitans(tensor.F64, 0, 1, 2, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewTitans(tensor.F64, 6, 4, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewTitans(tensor.F64, 4, 2, 0, 1); err == nil {
		t.Error("deep mode memHidden=0: want error")
	}
	if _, err := nn.NewTitans(tensor.F64, 4, 2, 0, 1, nn.WithTitansLinearMemory()); err != nil {
		t.Errorf("linear mode ignores memHidden, got error: %v", err)
	}
	if _, err := nn.NewTitans(tensor.F64, 4, 2, 3, 1, nn.WithTitansPersistentTokens(-1)); err == nil {
		t.Error("negative persistent tokens: want error")
	}
	if _, err := nn.NewNeuralMemory(tensor.F64, 0, 3, 1); err == nil {
		t.Error("NeuralMemory dim=0: want error")
	}
	if _, err := nn.NewNeuralMemory(tensor.F64, 4, 0, 1); err == nil {
		t.Error("NeuralMemory deep hidden=0: want error")
	}

	b, err := nn.NewTitans(tensor.F64, 4, 2, 3, 1, nn.WithTitansPersistentTokens(2))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("wrong input width: want error")
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, 5, 4)
	out, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{5, 4}) {
		t.Fatalf("output shape %v, want [5 4]", out.Shape())
	}
	// params: 6 memory proj/gates + W1 + W2 + NormMem γ + InProj(W,B) + OutProj(W,B) + persistent = 14
	if got, want := len(b.Params()), 14; got != want {
		t.Errorf("Params count %d, want %d", got, want)
	}

	m, err := nn.NewNeuralMemory(tensor.F64, 3, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := m.Scan(backend.NewContext(), r(2, 3, 1), r(2, 3), r(2, 3), r(2, 1), r(2, 1), r(2, 1)); err == nil {
		t.Error("rank-3 q: want error")
	}
	if _, err := m.Scan(backend.NewContext(), r(2, 3), r(3, 3), r(2, 3), r(2, 1), r(2, 1), r(2, 1)); err == nil {
		t.Error("seq mismatch: want error")
	}
	if _, err := m.Scan(backend.NewContext(), r(2, 4), r(2, 4), r(2, 4), r(2, 1), r(2, 1), r(2, 1)); err == nil {
		t.Error("q/k/v width != Dim: want error")
	}
	if _, err := m.Scan(backend.NewContext(), r(2, 3), r(2, 3), r(2, 3), r(2, 2), r(2, 1), r(2, 1)); err == nil {
		t.Error("eta not [T,1]: want error")
	}
	if _, err := m.Forward(backend.NewContext(), r(2, 4)); err == nil {
		t.Error("Forward wrong width: want error")
	}
}

// E2E char-LM: a 2-layer pre-norm residual LM built from Titans MAC blocks
// (deep memory + persistent tokens) must (1) be strictly causal — perturbing a
// later token cannot change earlier logits, the property the interleaved MAC
// layout was designed to keep — and (2) at least HALVE its cross-entropy on
// the deterministic grammar corpus (the incumbent Mamba/Hymba/FoX bar), then
// (3) generate valid tokens greedily. Self-contained copy of the incumbent
// harness (mamba_lm_e2e_test.go).
func TestTitansCharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a Titans LM; skipped in -short")
	}
	rng := rand.New(rand.NewPCG(2, 0x3a3))
	subjects := []string{"the cat", "a dog"}
	verbs := []string{" sees ", " fears "}
	objects := []string{"a star", "the sun"}
	var text []byte
	for range 400 {
		text = append(text, subjects[rng.IntN(2)]...)
		text = append(text, verbs[rng.IntN(2)]...)
		text = append(text, objects[rng.IntN(2)]...)
		text = append(text, '.', ' ')
	}
	seen := map[byte]int{}
	for _, b := range text {
		if _, ok := seen[b]; !ok {
			seen[b] = len(seen)
		}
	}
	vocab := len(seen)
	toks := make([]int, len(text))
	for i, b := range text {
		toks[i] = seen[b]
	}

	const (
		dim    = 48
		layers = 2
		window = 64
		steps  = 250
	)
	emb := tensor.New(tensor.F64, tensor.Shape{vocab, dim})
	nn.XavierUniform(emb, vocab, dim, 1)
	head := nn.NewLinear(tensor.F64, dim, vocab, 2)
	var blocks []*nn.Titans
	var norms []*nn.RMSNorm
	params := []*tensor.Tensor{emb}
	for l := range layers {
		b, err := nn.NewTitans(tensor.F64, dim, 4, 24, uint64(10+l), nn.WithTitansPersistentTokens(2))
		if err != nil {
			t.Fatal(err)
		}
		blocks = append(blocks, b)
		nrm := nn.NewRMSNorm(tensor.F64, dim)
		norms = append(norms, nrm)
		params = append(params, b.Params()...)
		params = append(params, nrm.Params()...)
	}
	finalNorm := nn.NewRMSNorm(tensor.F64, dim)
	params = append(params, finalNorm.Params()...)
	params = append(params, head.Params()...)

	onehot := func(seq []int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{len(seq), vocab})
		for i, tk := range seq {
			x.SetF64(1, i, tk)
		}
		return x
	}
	forward := func(ctx *backend.Context, seq []int) (*tensor.Tensor, error) {
		x, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{onehot(seq), emb}, nil)
		if err != nil {
			return nil, err
		}
		h := x[0]
		for l := range layers {
			n, err := norms[l].Forward(ctx, h)
			if err != nil {
				return nil, err
			}
			tb, err := blocks[l].Forward(ctx, n)
			if err != nil {
				return nil, err
			}
			sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{h, tb}, nil)
			if err != nil {
				return nil, err
			}
			h = sum[0]
		}
		n, err := finalNorm.Forward(ctx, h)
		if err != nil {
			return nil, err
		}
		return head.Forward(ctx, n)
	}

	// (1) causality: mutate the LAST input token; earlier logits must not move.
	probe := toks[:16]
	l1, err := forward(backend.NewContext(), probe)
	if err != nil {
		t.Fatal(err)
	}
	mut := append([]int(nil), probe...)
	mut[15] = (mut[15] + 1) % vocab
	l2, err := forward(backend.NewContext(), mut)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 15 {
		for v := range vocab {
			if l1.AtF64(i, v) != l2.AtF64(i, v) {
				t.Fatalf("causality violated: row %d changed after mutating token 15", i)
			}
		}
	}

	// (2) training: CE must at least halve from the first step.
	opt := nn.NewAdamW(params, 3e-3, 0.01)
	targets := tensor.New(tensor.F64, tensor.Shape{window})
	var first, last float64
	for step := range steps {
		start := (step * 89) % (len(toks) - window - 1)
		seq := toks[start : start+window]
		for i := range window {
			targets.SetF64(float64(toks[start+1+i]), i)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := forward(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("Titans char-LM: CE %.3f → %.3f (uniform = %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("Titans LM did not train (%.3f → %.3f)", first, last)
	}

	// (3) greedy generation: deterministic, valid ids.
	gen := append([]int(nil), toks[:8]...)
	for range 24 {
		logits, err := forward(backend.NewContext(), gen)
		if err != nil {
			t.Fatal(err)
		}
		lastRow := logits.Shape()[0] - 1
		best := 0
		for v := 1; v < vocab; v++ {
			if logits.AtF64(lastRow, v) > logits.AtF64(lastRow, best) {
				best = v
			}
		}
		if best < 0 || best >= vocab {
			t.Fatalf("invalid token %d", best)
		}
		gen = append(gen, best)
	}
	if len(gen) != 32 {
		t.Fatalf("generation length %d", len(gen))
	}
}

func ExampleTitans() {
	// A tiny Titans MAC block: a deep test-time memory whose retrievals are
	// interleaved as context tokens before causal attention.
	b, _ := nn.NewTitans(tensor.F64, 4, 2, 3, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := b.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(b.Params()))
	// Output: (3, 4) params=13
}

func ExampleNeuralMemory() {
	// Write the association key→[10,20] into a LINEAR memory with a single
	// test-time step (θ = 0.5 ⇒ delta-rule β = 1: an exact write), then read
	// the same key back via the query.
	m, _ := nn.NewNeuralMemory(tensor.F64, 2, 0, 1, nn.WithTitansLinearMemory())
	key := toT([][]float64{{1, 0}})
	val := toT([][]float64{{10, 20}})
	zero := toT([][]float64{{0}})
	half := toT([][]float64{{0.5}})
	out, _ := m.Scan(backend.NewContext(), key, key, val, zero, half, zero)
	fmt.Printf("%.2f %.2f params=%d\n", out.AtF64(0, 0), out.AtF64(0, 1), len(m.Params()))
	// Output: 10.00 20.00 params=7
}
