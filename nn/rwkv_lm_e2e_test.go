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

// §T517: an attention-free CHARACTER LANGUAGE MODEL built from full RWKV-4 blocks — the
// last architecture family of the e2e series (Mamba §T493, RetNet §T497, MLA, MoE),
// unblocked by the trainable OpWKV (§T516). Architecture: one-hot embedding matmul →
// 2 × RWKVBlock (pre-LN time-mixing + channel-mixing with internal residuals) →
// LayerNorm → head. Asserted: (1) CAUSALITY — perturbing the last input token must not
// change any earlier position's logits (WKV's recurrence and the token shift both only
// look backward); (2) the LM trains through the OpWKV VJP (CE at least halves);
// (3) greedy generation emits valid tokens.
func TestRWKVCharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains an RWKV LM; skipped in -short")
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
	var blocks []*nn.RWKVBlock
	params := []*tensor.Tensor{emb}
	for l := range layers {
		b, err := nn.NewRWKVBlock(tensor.F64, dim, 4*dim, uint64(10+8*l))
		if err != nil {
			t.Fatal(err)
		}
		blocks = append(blocks, b)
		params = append(params, b.Params()...)
	}
	finalNorm := nn.NewLayerNorm(tensor.F64, dim)
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
			if h, err = blocks[l].Forward(ctx, h); err != nil {
				return nil, err
			}
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

	// (2) training through the OpWKV VJP.
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
	t.Logf("RWKV char-LM: CE %.3f → %.3f (uniform = %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("RWKV LM did not train (%.3f → %.3f)", first, last)
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

// A single RWKV block maps [seq, dim] → [seq, dim] with both sublayers and their
// residuals inside.
func ExampleRWKVBlock() {
	b, _ := nn.NewRWKVBlock(tensor.F64, 8, 32, 1)
	x := tensor.New(tensor.F64, tensor.Shape{5, 8})
	for i := range x.Numel() {
		x.SetF64(math.Sin(float64(i)), tensor.Unravel(i, x.Shape())...)
	}
	y, _ := b.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (5, 8)
}

// §T518: RNN-mode/parallel-mode equivalence — the paper's defining inference property.
// Feeding a sequence row by row through Step (O(1) state, no cache) must reproduce the
// parallel Forward's rows: same recurrence in the same order, so agreement to 1e−12.
func TestRWKVStepMatchesForward(t *testing.T) {
	const seq, dim = 12, 16
	b, err := nn.NewRWKVBlock(tensor.F64, dim, 4*dim, 9)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	for i := range x.Numel() {
		x.SetF64(math.Sin(float64(i)*0.7)*0.9, tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	par, err := b.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	st := b.NewState()
	for tt := range seq {
		row := tensor.New(tensor.F64, tensor.Shape{1, dim})
		for c := range dim {
			row.SetF64(x.AtF64(tt, c), 0, c)
		}
		y, err := b.Step(ctx, st, row)
		if err != nil {
			t.Fatal(err)
		}
		for c := range dim {
			if d := math.Abs(y.AtF64(0, c) - par.AtF64(tt, c)); d > 1e-12 {
				t.Fatalf("step %d ch %d: recurrent %.17g vs parallel %.17g (Δ %.3g)",
					tt, c, y.AtF64(0, c), par.AtF64(tt, c), d)
			}
		}
	}
}

// Recurrent inference: NewState + Step consume one token at a time with O(1)
// memory — no KV cache (§T518).
func ExampleRWKVState() {
	b, _ := nn.NewRWKVBlock(tensor.F64, 8, 32, 1)
	st := b.NewState()
	x := tensor.New(tensor.F64, tensor.Shape{1, 8})
	x.SetF64(0.5, 0, 3)
	y, _ := b.Step(backend.NewContext(), st, x)
	fmt.Println(y.Shape())
	// Output: (1, 8)
}
