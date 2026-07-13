package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T498: Mixture-of-Experts in a REAL language model (Jamba-style: Mamba mixing + sparse MoE
// FFN) — routing and load balancing were only unit-tested. Asserted: (1) causality survives the
// per-token MoE (mutating a later token leaves earlier logits bit-identical); (2) the MoE LM
// trains (CE at least halves); (3) NO EXPERT COLLAPSE — with the auxiliary balance loss every
// expert serves a meaningful share of tokens (the failure mode the loss exists to prevent).
func TestMoECharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains an MoE LM; skipped in -short")
	}
	rng := rand.New(rand.NewPCG(2, 0x30e))
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
		dim     = 48
		experts = 4
		window  = 64
		steps   = 250
	)
	emb := tensor.New(tensor.F64, tensor.Shape{vocab, dim})
	nn.XavierUniform(emb, vocab, dim, 1)
	mamba, err := nn.NewMambaBlock(tensor.F64, dim, 8, 4, 2, 10)
	if err != nil {
		t.Fatal(err)
	}
	moe := nn.NewSparseMoE(tensor.F64, dim, 96, experts, 1, 11)
	n1, n2, nf := nn.NewRMSNorm(tensor.F64, dim), nn.NewRMSNorm(tensor.F64, dim), nn.NewRMSNorm(tensor.F64, dim)
	head := nn.NewLinear(tensor.F64, dim, vocab, 12)
	params := []*tensor.Tensor{emb}
	params = append(params, mamba.Params()...)
	params = append(params, moe.Params()...)
	params = append(params, n1.Params()...)
	params = append(params, n2.Params()...)
	params = append(params, nf.Params()...)
	params = append(params, head.Params()...)

	onehot := func(seq []int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{len(seq), vocab})
		for i, tk := range seq {
			x.SetF64(1, i, tk)
		}
		return x
	}
	forward := func(ctx *backend.Context, seq []int) (logits, gateLogits *tensor.Tensor, err error) {
		x, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{onehot(seq), emb}, nil)
		if err != nil {
			return nil, nil, err
		}
		h := x[0]
		nrm, err := n1.Forward(ctx, h)
		if err != nil {
			return nil, nil, err
		}
		mm, err := mamba.Forward(ctx, nrm)
		if err != nil {
			return nil, nil, err
		}
		s1, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{h, mm}, nil)
		if err != nil {
			return nil, nil, err
		}
		h = s1[0]
		if nrm, err = n2.Forward(ctx, h); err != nil {
			return nil, nil, err
		}
		my, gl, err := moe.Forward(ctx, nrm)
		if err != nil {
			return nil, nil, err
		}
		s2, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{h, my}, nil)
		if err != nil {
			return nil, nil, err
		}
		h = s2[0]
		if nrm, err = nf.Forward(ctx, h); err != nil {
			return nil, nil, err
		}
		lg, err := head.Forward(ctx, nrm)
		return lg, gl, err
	}
	assign := func(gl *tensor.Tensor) *tensor.Tensor {
		tk := gl.Shape()[0]
		a := tensor.New(tensor.F64, tensor.Shape{tk})
		for i := range tk {
			best := 0
			for e := 1; e < experts; e++ {
				if gl.AtF64(i, e) > gl.AtF64(i, best) {
					best = e
				}
			}
			a.SetF64(float64(best), i)
		}
		return a
	}

	// (1) causality.
	probe := toks[:16]
	l1, _, err := forward(backend.NewContext(), probe)
	if err != nil {
		t.Fatal(err)
	}
	mut := append([]int(nil), probe...)
	mut[15] = (mut[15] + 1) % vocab
	l2, _, err := forward(backend.NewContext(), mut)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 15 {
		for v := range vocab {
			if l1.AtF64(i, v) != l2.AtF64(i, v) {
				t.Fatalf("causality violated at row %d", i)
			}
		}
	}

	// (2) training with the balance auxiliary.
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
		logits, gl, err := forward(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		ce, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		bal, err := nn.MoEBalanceLoss(ctx, gl, assign(gl), 0.01)
		if err != nil {
			t.Fatal(err)
		}
		total, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{ce, bal}, nil)
		if err != nil {
			t.Fatal(err)
		}
		lv := ce.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(total[0]); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("MoE char-LM: CE %.3f → %.3f (uniform %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("MoE LM did not train (%.3f → %.3f)", first, last)
	}

	// (3) expert utilization on a fresh window: no collapse.
	_, gl, err := forward(backend.NewContext(), toks[500:500+window])
	if err != nil {
		t.Fatal(err)
	}
	counts := make([]int, experts)
	a := assign(gl)
	for i := range window {
		counts[int(a.AtF64(i))]++
	}
	t.Logf("expert utilization over %d tokens: %v", window, counts)
	for e, c := range counts {
		if c < window/10 {
			t.Fatalf("expert %d underutilized (%d/%d tokens) — routing collapse", e, c, window)
		}
	}
}
