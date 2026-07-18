package nn_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// gptOracleEmbed reimplements nlp/gpt.go's hand-rolled token lookup verbatim:
// build a float index tensor of the table's dtype, then one OpEmbed through the
// dispatch. nn.Embedding must be bit-identical to this, proving it is a drop-in
// replacement for the six hand-rolled sites in the tree.
func gptOracleEmbed(ctx *backend.Context, table *tensor.Tensor, tokens []int) (*tensor.Tensor, error) {
	tokIdx := tensor.New(table.Dtype(), tensor.Shape{len(tokens)})
	for i, t := range tokens {
		tokIdx.SetF64(float64(t), i)
	}
	out, err := backend.Execute(ctx, backend.OpEmbed, []*tensor.Tensor{table, tokIdx}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Bit-identity against the nlp/gpt.go oracle, for both dtypes and both call forms.
func TestEmbeddingMatchesGPTOracle(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		emb := nn.NewEmbedding(dt, 16, 5, 42)
		tokens := []int{0, 3, 15, 3, 7, 0}

		want, err := gptOracleEmbed(backend.NewContext(), emb.W, tokens)
		if err != nil {
			t.Fatal(err)
		}
		got, err := emb.Lookup(backend.NewContext(), tokens)
		if err != nil {
			t.Fatal(err)
		}
		if !equalTensors(got, want) {
			t.Errorf("%v: Embedding.Lookup differs from the gpt.go oracle", dt)
		}

		idx := tensor.New(dt, tensor.Shape{len(tokens)})
		for i, tok := range tokens {
			idx.SetF64(float64(tok), i)
		}
		got2, err := emb.Forward(backend.NewContext(), idx)
		if err != nil {
			t.Fatal(err)
		}
		if !equalTensors(got2, want) {
			t.Errorf("%v: Embedding.Forward differs from the gpt.go oracle", dt)
		}
		if !got.Shape().Equal(tensor.Shape{len(tokens), 5}) {
			t.Errorf("shape = %v, want [%d,5]", got.Shape(), len(tokens))
		}
	}
}

// Params exposes exactly the table, and the gathered rows really are the table's.
func TestEmbeddingParamsAndValues(t *testing.T) {
	emb := nn.NewEmbedding(tensor.F64, 4, 3, 7)
	ps := emb.Params()
	if len(ps) != 1 || ps[0] != emb.W {
		t.Fatalf("Params() must be exactly the table, got %d tensors", len(ps))
	}
	out, err := emb.Lookup(backend.NewContext(), []int{2, 0})
	if err != nil {
		t.Fatal(err)
	}
	for j := range 3 {
		if out.AtF64(0, j) != emb.W.AtF64(2, j) || out.AtF64(1, j) != emb.W.AtF64(0, j) {
			t.Fatal("gathered rows must equal the table rows")
		}
	}
}

// Shape/dtype/range validation is reported as errors, not panics.
func TestEmbeddingValidation(t *testing.T) {
	emb := nn.NewEmbedding(tensor.F64, 8, 2, 1)
	cases := []struct {
		name string
		ids  *tensor.Tensor
		want string
	}{
		{"rank-2 ids", tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0, 1, 2, 3}), "rank-1"},
		{"out of range", tensor.FromFloat64(tensor.Shape{2}, []float64{0, 8}), "outside vocab"},
		{"negative", tensor.FromFloat64(tensor.Shape{1}, []float64{-1}), "outside vocab"},
		{"non-integer", tensor.FromFloat64(tensor.Shape{1}, []float64{1.5}), "not an integer"},
	}
	for _, c := range cases {
		_, err := emb.Forward(backend.NewContext(), c.ids)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want one containing %q", c.name, err, c.want)
		}
	}
	if _, err := emb.Forward(backend.NewContext(), nil); err == nil {
		t.Error("nil ids must error")
	}
	// dtype mismatch
	f32ids := tensor.New(tensor.F32, tensor.Shape{1})
	if _, err := emb.Forward(backend.NewContext(), f32ids); err == nil ||
		!strings.Contains(err.Error(), "dtype") {
		t.Errorf("dtype mismatch must be reported, got %v", err)
	}
	if _, err := emb.Lookup(backend.NewContext(), nil); err == nil {
		t.Error("empty id slice must error")
	}
}

func TestEmbeddingConstructorPanics(t *testing.T) {
	for _, c := range []struct {
		name              string
		vocab, dim, padIx int
	}{
		{"zero vocab", 0, 4, -1},
		{"zero dim", 4, 0, -1},
		{"pad out of range", 4, 2, 4},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected a panic", c.name)
				}
			}()
			nn.NewEmbeddingPadded(tensor.F64, c.vocab, c.dim, c.padIx, 1)
		}()
	}
}

// The padding row is zero-initialised, and the grad hook zeroes exactly that row
// of the gradient while leaving every other row untouched.
func TestEmbeddingPaddingIdxGradient(t *testing.T) {
	const vocab, dim, pad = 6, 3, 2
	emb := nn.NewEmbeddingPadded(tensor.F64, vocab, dim, pad, 5)
	for j := range dim {
		if emb.W.AtF64(pad, j) != 0 {
			t.Fatal("padding row must be zero-initialised")
		}
	}

	// Reference gradient without the hook: the pad row DOES accumulate.
	tokens := []int{0, pad, 4, pad}
	bare := func() *tensor.Tensor {
		tape := autograd.NewTape()
		out, err := emb.Lookup(tape.Context(), tokens)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatal(err)
		}
		return tape.Grad(emb.W)
	}()
	padSum := 0.0
	for j := range dim {
		padSum += bare.AtF64(pad, j)
	}
	if padSum == 0 {
		t.Fatal("precondition: without the hook the pad row must accumulate gradient " +
			"(that is exactly why the hook exists)")
	}

	// With the hook installed the pad row is EXACTLY zero and the others are
	// bit-identical to the unhooked gradient.
	tape := autograd.NewTape()
	tape.RegisterGradHook(emb.W, emb.PadGradHook())
	out, err := emb.Lookup(tape.Context(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(emb.W)
	if g == nil {
		t.Fatal("no gradient for the table")
	}
	for i := range vocab {
		for j := range dim {
			got := g.AtF64(i, j)
			if i == pad {
				if got != 0 {
					t.Errorf("pad row grad[%d,%d] = %g, want exactly 0", i, j, got)
				}
				continue
			}
			if got != bare.AtF64(i, j) {
				t.Errorf("grad[%d,%d] = %g, want %g (unchanged by the hook)", i, j, got, bare.AtF64(i, j))
			}
		}
	}

	// No padding row → the hook is inert (returns nil = keep the gradient).
	if nn.NewEmbedding(tensor.F64, 4, 2, 1).PadGradHook()(bare) != nil {
		t.Error("a padding-free embedding's hook must observe only")
	}
}

// ZeroPadRow is the manual alternative and clears exactly the pad row.
func TestEmbeddingZeroPadRow(t *testing.T) {
	emb := nn.NewEmbeddingPadded(tensor.F64, 4, 3, 1, 9)
	for j := range 3 {
		emb.W.SetF64(1.5, 1, j) // simulate an optimizer step having moved it
	}
	emb.ZeroPadRow()
	for j := range 3 {
		if emb.W.AtF64(1, j) != 0 {
			t.Error("ZeroPadRow must clear the padding row")
		}
		if emb.W.AtF64(0, j) == 0 {
			t.Error("ZeroPadRow must not touch other rows")
		}
	}
	nn.NewEmbedding(tensor.F64, 4, 3, 9).ZeroPadRow() // no padding row: no-op
}

// The embedding gradient is verified against central finite differences: the ids
// are captured in the closure (they are indices, not differentiable inputs), so
// only the table is perturbed.
func TestEmbeddingGradCheck(t *testing.T) {
	emb := nn.NewEmbedding(tensor.F64, 5, 3, 13)
	ids := tensor.FromFloat64(tensor.Shape{4}, []float64{0, 4, 1, 4})
	f := func(ctx *backend.Context, in []*tensor.Tensor) (*tensor.Tensor, error) {
		return (&nn.Embedding{W: in[0], PadIdx: -1}).Forward(ctx, ids)
	}
	if err := autograd.GradCheck(f, []*tensor.Tensor{emb.W}); err != nil {
		t.Fatal(err)
	}
}

// The first layer of essentially every language model: integer token ids in,
// learned dense vectors out. Lookup takes plain []int and builds the index
// tensor for you; Forward is the form that takes the index tensor directly, so
// the layer drops into the hand-rolled lookups already in nlp/.
func ExampleEmbedding() {
	emb := nn.NewEmbedding(tensor.F64, 8, 4, 42) // vocab 8, dim 4, fixed seed

	out, err := emb.Lookup(backend.NewContext(), []int{3, 1, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println("shape:", out.Shape()) // one row per id

	// A repeated id yields the same row — it is a gather, not a computation.
	same := true
	for j := range 4 {
		if out.AtF64(0, j) != out.AtF64(2, j) {
			same = false
		}
	}
	fmt.Println("row for id 3 repeats:", same)
	fmt.Printf("first vector: %.4f\n", out.AtF64(0, 0))

	// Output:
	// shape: (3, 4)
	// row for id 3 repeats: true
	// first vector: 0.2852
}

// Freezing the padding row. PadIdx on its own does NOTHING but zero-initialise
// that row — the row still trains like any other. Zeroing its gradient is a HOOK
// you install on the tape, and this is what installing it looks like.
func ExampleEmbedding_padGradHook() {
	const vocab, dim, pad = 4, 3, 0

	// A batch where the padding id appears twice, so the padding row genuinely
	// accumulates gradient and the hook has something to suppress.
	batch := []int{pad, 2, pad, 3}

	padRowGrad := func(installHook bool) float64 {
		emb := nn.NewEmbeddingPadded(tensor.F64, vocab, dim, pad, 7)
		tape := autograd.NewTape()
		if installHook {
			tape.RegisterGradHook(emb.W, emb.PadGradHook()) // <- the one line
		}
		out, err := emb.Lookup(tape.Context(), batch)
		if err != nil {
			panic(err)
		}
		loss, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out}, nil)
		if err != nil {
			panic(err)
		}
		if err := tape.Backward(loss[0]); err != nil {
			panic(err)
		}
		return tape.Grad(emb.W).AtF64(pad, 0)
	}

	fmt.Printf("pad-row gradient without the hook: %.1f\n", padRowGrad(false))
	fmt.Printf("pad-row gradient with    the hook: %.1f\n", padRowGrad(true))

	// ZeroPadRow is the manual alternative if you cannot install a hook: call it
	// after each optimizer step. It repairs the weights but not the optimizer's
	// momentum for that row, so the hook stays the faithful option.
	emb := nn.NewEmbeddingPadded(tensor.F64, vocab, dim, pad, 7)
	emb.W.SetF64(1.5, pad, 0) // pretend an optimizer step moved the padding row
	emb.ZeroPadRow()
	fmt.Println("after ZeroPadRow:", emb.W.AtF64(pad, 0))

	// Output:
	// pad-row gradient without the hook: 2.0
	// pad-row gradient with    the hook: 0.0
	// after ZeroPadRow: 0
}
