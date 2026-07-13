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

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, 0xd1ce)) }

func randLogits(rng *rand.Rand, seq, vocab int) *tensor.Tensor {
	d := make([]float64, seq*vocab)
	for i := range d {
		d[i] = rng.NormFloat64()
	}
	return tensor.FromFloat64(tensor.Shape{seq, vocab}, d)
}

// manualRows / manualVec slice a tensor independently of the OpSlice path, for a
// cross-check reference.
func manualRows(t *tensor.Tensor, start, end int) *tensor.Tensor {
	v := t.Shape()[1]
	d := make([]float64, (end-start)*v)
	for r := start; r < end; r++ {
		for c := range v {
			d[(r-start)*v+c] = t.AtF64(r, c)
		}
	}
	return tensor.FromFloat64(tensor.Shape{end - start, v}, d)
}

func manualVec(t *tensor.Tensor, start, end int) *tensor.Tensor {
	d := make([]float64, end-start)
	for i := start; i < end; i++ {
		d[i-start] = t.AtF64(i)
	}
	return tensor.FromFloat64(tensor.Shape{end - start}, d)
}

// §V16 tier-1: n=1 multi-token loss is exactly the standard next-token cross-entropy
// (head 1 predicting x_{t+1}), i.e. CrossEntropy(logits[0:seq−1], targets[1:seq]) (§R134).
func TestMultiTokenLossN1EqualsCrossEntropy(t *testing.T) {
	const seq, vocab = 5, 4
	rng := newRNG(1)
	h := randLogits(rng, seq, vocab)
	tg := tensor.FromFloat64(tensor.Shape{seq}, []float64{2, 0, 3, 1, 2})
	ctx := backend.NewContext()
	got, err := nn.MultiTokenLoss(ctx, []*tensor.Tensor{h}, tg)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := nn.CrossEntropy(ctx, manualRows(h, 0, seq-1), manualVec(tg, 1, seq))
	if math.Abs(got.AtF64()-want.AtF64()) > 1e-12 {
		t.Errorf("n=1 loss %.12g != next-token CE %.12g", got.AtF64(), want.AtF64())
	}
}

// §V16 tier-1: the n-head loss is the plain unweighted sum of the n shifted
// cross-entropies, matching an independent manual recomputation.
func TestMultiTokenLossSumMatchesManual(t *testing.T) {
	const seq, vocab, n = 6, 5, 3
	rng := newRNG(7)
	heads := make([]*tensor.Tensor, n)
	for i := range heads {
		heads[i] = randLogits(rng, seq, vocab)
	}
	tg := tensor.FromFloat64(tensor.Shape{seq}, []float64{1, 4, 0, 2, 3, 1})
	ctx := backend.NewContext()
	got, err := nn.MultiTokenLoss(ctx, heads, tg)
	if err != nil {
		t.Fatal(err)
	}
	var want float64
	for i := range n {
		shift := i + 1
		ce, _ := nn.CrossEntropy(ctx, manualRows(heads[i], 0, seq-shift), manualVec(tg, shift, seq))
		want += ce.AtF64()
	}
	if math.Abs(got.AtF64()-want) > 1e-12 {
		t.Errorf("sum loss %.12g != manual %.12g", got.AtF64(), want)
	}
}

// §V2: the gradient flows into every head's logits, matching finite differences.
func TestMultiTokenLossGradcheck(t *testing.T) {
	const seq, vocab, n, h, rtol = 5, 3, 2, 1e-6, 2e-3
	rng := newRNG(3)
	heads := make([]*tensor.Tensor, n)
	for i := range heads {
		heads[i] = randLogits(rng, seq, vocab)
	}
	tg := tensor.FromFloat64(tensor.Shape{seq}, []float64{0, 2, 1, 2, 0})
	lossAt := func() float64 {
		l, err := nn.MultiTokenLoss(backend.NewContext(), heads, tg)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	tape := autograd.NewTape()
	l, err := nn.MultiTokenLoss(tape.Context(), heads, tg)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}
	for hi, head := range heads {
		g := tape.Grad(head)
		for i := range head.Numel() {
			idx := tensor.Unravel(i, head.Shape())
			o := head.AtF64(idx...)
			head.SetF64(o+h, idx...)
			lp := lossAt()
			head.SetF64(o-h, idx...)
			lm := lossAt()
			head.SetF64(o, idx...)
			if num, ana := (lp-lm)/(2*h), g.AtF64(idx...); math.Abs(num-ana) > rtol*math.Max(1, math.Abs(num)) {
				t.Errorf("head %d grad[%v]: numeric %.8g vs analytic %.8g", hi, idx, num, ana)
			}
		}
	}
}

func TestMultiTokenLossErrors(t *testing.T) {
	ctx := backend.NewContext()
	tg := tensor.FromFloat64(tensor.Shape{4}, []float64{0, 1, 2, 0})
	h := tensor.New(tensor.F64, tensor.Shape{4, 3})
	if _, err := nn.MultiTokenLoss(ctx, nil, tg); err == nil {
		t.Error("no heads must error")
	}
	if _, err := nn.MultiTokenLoss(ctx, []*tensor.Tensor{h, h, h, h}, tg); err == nil {
		t.Error("seq ≤ n must error")
	}
	if _, err := nn.MultiTokenLoss(ctx, []*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{5, 3})}, tg); err == nil {
		t.Error("head seq mismatch must error")
	}
}

// Multi-token prediction (Gloeckle et al. 2024) trains n heads to predict the next
// n tokens; the loss is the sum of the n shifted cross-entropies. With n=1 it is
// exactly next-token prediction.
func ExampleMultiTokenLoss() {
	// two heads over a length-4, vocab-3 sequence
	h1 := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0})
	h2 := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0})
	targets := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 0, 1})
	loss, _ := nn.MultiTokenLoss(backend.NewContext(), []*tensor.Tensor{h1, h2}, targets)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: 4.4791
}
