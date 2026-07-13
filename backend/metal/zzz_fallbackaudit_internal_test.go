//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// §T401 (§V22 audit): the metal backend registers OpEmbed-backward and OpCrossEntropy-backward but
// NOT their FORWARDS, so a GPT forward's token embedding and loss fall back to the reference (CPU
// f64) backend (§I4). Measure whether that silent fallback is a real fraction of the training step
// (~110ms, §T399) — like the §T352 GELU/AddBias fallbacks were — before writing metal kernels.
func TestFallbackAuditForward(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const seq, dim, vocab = 256, 512, 4096 // BenchmarkGPTForward shapes (§T355)
	mctx := backend.NewContext().WithBackend(Backend{})

	// OpEmbed forward: gather seq rows of the [vocab,dim] table.
	table := reshape2(t, randTensor(vocab*dim, 1), vocab, dim)
	idx := tensor.New(tensor.F32, tensor.Shape{seq})
	for i := 0; i < seq; i++ {
		idx.SetF64(float64(i%vocab), i)
	}
	timeOp := func(op backend.Op, in []*tensor.Tensor) float64 {
		if _, err := backend.Execute(mctx, op, in, nil); err != nil {
			t.Fatalf("%v: %v", op, err)
		}
		const N = 30
		s := time.Now()
		for range N {
			if _, err := backend.Execute(mctx, op, in, nil); err != nil {
				t.Fatal(err)
			}
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	embMs := timeOp(backend.OpEmbed, []*tensor.Tensor{table, idx})

	// OpCrossEntropy forward: softmax + NLL over [seq,vocab].
	logits := reshape2(t, randTensor(seq*vocab, 2), seq, vocab)
	tg := tensor.New(tensor.F32, tensor.Shape{seq})
	for i := 0; i < seq; i++ {
		tg.SetF64(float64(i%vocab), i)
	}
	ceMs := timeOp(backend.OpCrossEntropy, []*tensor.Tensor{logits, tg})

	t.Logf("silent CPU fallbacks in GPT forward (metal ctx → ref): OpEmbed %.2f ms | OpCrossEntropy %.2f ms | sum %.2f ms (vs ~32ms fwd / ~110ms training step §T399)",
		embMs, ceMs, embMs+ceMs)

	// backward-path reductions the VJPs use (also unregistered on metal → fall back).
	timeReduce := func(x *tensor.Tensor, axis int) float64 {
		attrs := backend.ReduceAttrs{Axes: []int{axis}}
		if _, err := backend.Execute(mctx, backend.OpSum, []*tensor.Tensor{x}, attrs); err != nil {
			t.Fatalf("OpSum: %v", err)
		}
		const N = 30
		s := time.Now()
		for range N {
			if _, err := backend.Execute(mctx, backend.OpSum, []*tensor.Tensor{x}, attrs); err != nil {
				t.Fatal(err)
			}
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	sumSmall := timeReduce(reshape2(t, randTensor(seq*dim, 5), seq, dim), 0)     // [256,512]→[512]
	sumVocab := timeReduce(reshape2(t, randTensor(seq*vocab, 6), seq, vocab), 0) // [256,4096]→[4096]
	t.Logf("backward-path OpSum fallbacks: [%d,%d]→ax0 %.2f ms | [%d,%d]→ax0 %.2f ms",
		seq, dim, sumSmall, seq, vocab, sumVocab)
}

// §T401: the new metal OpCrossEntropy forward == the reference @1e-4.
func TestCrossEntropyForwardCrossReference(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	for _, sh := range [][2]int{{4, 16}, {8, 40}, {17, 4096}} {
		b, c := sh[0], sh[1]
		logits := reshape2(t, randTensor(b*c, 3), b, c)
		tg := tensor.New(tensor.F32, tensor.Shape{b})
		for i := 0; i < b; i++ {
			tg.SetF64(float64((i*3)%c), i)
		}
		mctx := backend.NewContext().WithBackend(Backend{})
		gotOut, err := backend.Execute(mctx, backend.OpCrossEntropy, []*tensor.Tensor{logits, tg}, nil)
		if err != nil {
			t.Fatal(err)
		}
		wantOut, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpCrossEntropy, []*tensor.Tensor{logits, tg}, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, want := gotOut[0].AtF64(), wantOut[0].AtF64()
		if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Fatalf("[%dx%d]: metal %v vs ref %v", b, c, got, want)
		}
	}
}

// §T402: the new metal OpEmbed forward == the reference (exact — it's a gather).
func TestEmbedForwardCrossReference(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const n, d, m = 4096, 512, 256
	table := reshape2(t, randTensor(n*d, 8), n, d)
	idx := tensor.New(tensor.F32, tensor.Shape{m})
	for i := 0; i < m; i++ {
		idx.SetF64(float64((i*7)%n), i)
	}
	mctx := backend.NewContext().WithBackend(Backend{})
	got, err := backend.Execute(mctx, backend.OpEmbed, []*tensor.Tensor{table, idx}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpEmbed, []*tensor.Tensor{table, idx}, nil)
	if err != nil {
		t.Fatal(err)
	}
	gs, ws := got[0].Storage().F32(), want[0].Storage().F32()
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("[%d]: metal %v vs ref %v", i, gs[i], ws[i])
		}
	}
}

func randTensor(n, seed int) *tensor.Tensor {
	t := tensor.New(tensor.F32, tensor.Shape{n})
	s := t.Storage().F32()
	for i := range s {
		s[i] = float32((i*7+seed*13)%23-11) * 0.02
	}
	return t
}

func reshape2(t *testing.T, x *tensor.Tensor, r, c int) *tensor.Tensor {
	out, err := x.Reshape(tensor.Shape{r, c})
	if err != nil {
		t.Fatal(err)
	}
	return out
}
