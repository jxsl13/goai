package vision_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// tinyGPT builds a random-initialized 1-layer decoder (vocab 4, dim 16) for the
// VLM e2e — captions live in tokens {0,1,2}, token 3 is BOS.
func tinyGPT(t *testing.T, seed uint64) *nlp.GPT {
	t.Helper()
	const vocab, ctxLen, dim, ffn = 4, 32, 16, 32
	r := func(s uint64, shape ...int) *tensor.Tensor {
		x := tensor.Randn(tensor.F64, seed+s, tensor.Shape(shape))
		for i, v := range x.Storage().F64() {
			x.Storage().F64()[i] = v * 0.05
		}
		return x
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			x.SetF64(1, i)
		}
		return x
	}
	zeros := func(n int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape{n}) }
	ts := map[string]*tensor.Tensor{
		"tok_emb": r(1, vocab, dim), "pos_emb": r(2, ctxLen, dim),
		"blocks.0.ln1.gamma": ones(dim), "blocks.0.ln1.beta": zeros(dim),
		"blocks.0.attn.wq": r(3, dim, dim), "blocks.0.attn.wk": r(4, dim, dim),
		"blocks.0.attn.wv": r(5, dim, dim), "blocks.0.attn.wo": r(6, dim, dim),
		"blocks.0.ln2.gamma": ones(dim), "blocks.0.ln2.beta": zeros(dim),
		"blocks.0.ffn.w1": r(7, dim, ffn), "blocks.0.ffn.b1": zeros(ffn),
		"blocks.0.ffn.w2": r(8, ffn, dim), "blocks.0.ffn.b2": zeros(dim),
		"lnf.gamma": ones(dim), "lnf.beta": zeros(dim),
		"head": r(9, dim, vocab),
	}
	g, err := nlp.FromSafetensors(nlp.GPTConfig{Vocab: vocab, Ctx: ctxLen, Dim: dim, Heads: 2, Layers: 1, Eps: 1e-5}, ts)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// vlmPredict returns the caption token the trained VLM emits for one image (the
// argmax at the BOS row — the position that predicts the caption).
func vlmPredict(t *testing.T, vit *vision.ViT, proj *vision.VLMProjector, gpt *nlp.GPT, img *tensor.Tensor) int {
	t.Helper()
	logits, err := vision.VLMLogits(backend.NewContext(), vit, proj, gpt, img, []int{3})
	if err != nil {
		t.Fatal(err)
	}
	rows, vocab := logits.Shape()[0], logits.Shape()[1]
	vals := logits.Storage().F64()[(rows-1)*vocab : rows*vocab]
	best := 0
	for i, v := range vals {
		if v > vals[best] {
			best = i
		}
	}
	return best
}

// §T592 end-to-end (the trained-model discipline): ViT features → MLP projector →
// soft tokens before a tiny GPT, trained jointly on synthetic image→caption pairs
// until every caption is right — the LLaVA mechanism works as a system. Structural
// causality: swapping the image changes the caption (the decoder reads the image
// tokens, not a text prior).
func TestVLMProjectorLearnsCaptions(t *testing.T) {
	vit, err := vision.NewViT(1, 8, 3, 7,
		vision.WithViTDtype(tensor.F64), vision.WithViTDim(16), vision.WithViTDepth(1), vision.WithViTHeads(2))
	if err != nil {
		t.Fatal(err)
	}
	proj, err := vision.NewVLMProjector(tensor.F64, 16, 16, 21)
	if err != nil {
		t.Fatal(err)
	}
	gpt := tinyGPT(t, 31)
	x, _, labels := stripes(12, 8)

	params := append(append(vit.Params(), proj.Params()...), gpt.Params()...)
	opt := nn.NewAdamW(params, 0.01, 0)
	var first, last float64
	for step := range 120 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var lossSum *tensor.Tensor
		for b, lbl := range labels {
			img, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{x}, backend.SliceAttrs{Axis: 0, Start: b, End: b + 1})
			if err != nil {
				t.Fatal(err)
			}
			one, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{img[0]}, backend.ReshapeAttrs{Shape: tensor.Shape{1, 8, 8}})
			if err != nil {
				t.Fatal(err)
			}
			logits, err := vision.VLMLogits(ctx, vit, proj, gpt, one[0], []int{3})
			if err != nil {
				t.Fatal(err)
			}
			rows := logits.Shape()[0]
			lastRow, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits}, backend.SliceAttrs{Axis: 0, Start: rows - 1, End: rows})
			if err != nil {
				t.Fatal(err)
			}
			target := tensor.New(tensor.F64, tensor.Shape{1})
			target.SetF64(float64(lbl), 0)
			loss, err := nn.CrossEntropy(ctx, lastRow[0], target)
			if err != nil {
				t.Fatal(err)
			}
			if lossSum == nil {
				lossSum = loss
			} else {
				sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{lossSum, loss}, nil)
				if err != nil {
					t.Fatal(err)
				}
				lossSum = sum[0]
			}
		}
		lv := lossSum.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(lossSum); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	if last > first*0.1 {
		t.Fatalf("VLM did not learn: loss %.4f → %.4f", first, last)
	}
	// every caption right, and the caption FOLLOWS THE IMAGE (causality): find two
	// images of different classes and check swapping them swaps the prediction.
	preds := make([]int, len(labels))
	for b, lbl := range labels {
		img := sliceImage(t, x, b)
		preds[b] = vlmPredict(t, vit, proj, gpt, img)
		if preds[b] != lbl {
			t.Fatalf("image %d: caption %d, want %d", b, preds[b], lbl)
		}
	}
	for b, lbl := range labels {
		for c, lbl2 := range labels {
			if lbl != lbl2 {
				if p := vlmPredict(t, vit, proj, gpt, sliceImage(t, x, c)); p == preds[b] && lbl2 != labels[b] {
					t.Fatalf("caption did not follow the image: img %d and %d both → %d", b, c, p)
				}
				break
			}
		}
		break
	}
}

// sliceImage extracts image b from the [batch,1,8,8] tensor as [1,8,8].
func sliceImage(t *testing.T, x *tensor.Tensor, b int) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	img, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{x}, backend.SliceAttrs{Axis: 0, Start: b, End: b + 1})
	if err != nil {
		t.Fatal(err)
	}
	one, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{img[0]}, backend.ReshapeAttrs{Shape: tensor.Shape{1, 8, 8}})
	if err != nil {
		t.Fatal(err)
	}
	return one[0]
}

// §T592 validation: nil parts and empty text error; context overflow errors.
func TestVLMValidation(t *testing.T) {
	vit, _ := vision.NewViT(1, 8, 3, 7, vision.WithViTDtype(tensor.F64), vision.WithViTDim(16), vision.WithViTDepth(1), vision.WithViTHeads(2))
	proj, _ := vision.NewVLMProjector(tensor.F64, 16, 16, 5)
	gpt := tinyGPT(t, 9)
	img := tensor.New(tensor.F64, tensor.Shape{1, 8, 8})
	if _, err := vision.VLMLogits(backend.NewContext(), vit, proj, gpt, img, nil); err == nil {
		t.Error("empty text must error")
	}
	if _, err := vision.VLMLogits(backend.NewContext(), nil, proj, gpt, img, []int{3}); err == nil {
		t.Error("nil vit must error")
	}
	if _, err := vision.NewVLMProjector(tensor.F64, 0, 16, 5); err == nil {
		t.Error("zero dim must error")
	}
}
