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

// §T499: an attention-free CHARACTER LANGUAGE MODEL built from MLA attention layers — the post-transformer
// blocks' first real-LM e2e. Architecture: one-hot embedding matmul → 2 × (RMSNorm → MambaBlock →
// residual) → RMSNorm → head. Asserted: (1) CAUSALITY — perturbing a later input token must not
// change any earlier position's logits (causal latent attention's defining structural property); (2) the LM
// trains on the char grammar (CE at least halves from the uniform start); (3) greedy generation
// emits valid tokens deterministically.
func TestMLACharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a MLA LM; skipped in -short")
	}
	rng := rand.New(rand.NewPCG(2, 0x3a3))
	// tiny grammar corpus (self-contained; the llamagpu helpers live in another package).
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
	var blocks []*nn.MLA
	var norms []*nn.RMSNorm
	params := []*tensor.Tensor{emb}
	for l := range layers {
		b, err := nn.NewMLA(tensor.F64, dim, 4, 12, 8, 16, 16, true, uint64(10+l))
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
			mb, err := blocks[l].Forward(ctx, n)
			if err != nil {
				return nil, err
			}
			sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{h, mb}, nil)
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

	// (1) causality: change the LAST input token; logits at earlier rows must be identical.
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

	// (2) training.
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
	t.Logf("MLA char-LM: CE %.3f → %.3f (uniform = %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("MLA LM did not train (%.3f → %.3f)", first, last)
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
