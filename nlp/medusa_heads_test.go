package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T443: ForwardHidden is Forward without the LM head — projecting it through Head reproduces
// Forward's logits bit-identically (the surgery changed no numerics).
func TestForwardHiddenTimesHeadIsForward(t *testing.T) {
	model, prompt := loadGPTModel(t)
	ctx := backend.NewContext()
	want, err := model.Forward(ctx, prompt)
	if err != nil {
		t.Fatal(err)
	}
	hidden, err := model.ForwardHidden(ctx, prompt)
	if err != nil {
		t.Fatal(err)
	}
	got, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{hidden, model.Head}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tensorsEqual(t, got[0], want)
}

// §T443: Medusa heads TRAIN on a frozen base — the library's own loop (hidden computed outside
// the tape, tape over MedusaHeads.Logits) drives the multi-offset CE loss well below its start
// on a sequence whose future tokens are a deterministic function of the current one.
func TestMedusaHeadsTrainOnFrozenBase(t *testing.T) {
	model, _ := loadGPTModel(t)
	cfg := model.Config
	const K = 2
	heads, err := nlp.NewMedusaHeads(K, cfg.Dim, cfg.Vocab, 5, nlp.WithMedusaHeadsDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.NewMedusaHeads(0, 1, 1, 0); err == nil {
		t.Fatal("k=0 accepted")
	}

	// deterministic corpus: s_{i+1} = (s_i + 3) mod V ⇒ token at offset t+2+k is s_t + 3(2+k).
	seqs := make([][]int, 24)
	for n := range seqs {
		s := make([]int, cfg.Ctx)
		s[0] = (n * 5) % cfg.Vocab
		for i := 1; i < len(s); i++ {
			s[i] = (s[i-1] + 3) % cfg.Vocab
		}
		seqs[n] = s
	}
	ctx := backend.NewContext()
	hiddens := make([]*tensor.Tensor, len(seqs))
	for n, s := range seqs {
		h, err := model.ForwardHidden(ctx, s) // frozen base: outside any tape
		if err != nil {
			t.Fatal(err)
		}
		hiddens[n] = h
	}

	opt := nn.NewAdamW(heads.Params(), 0.05, 0.0)
	var first, last float64
	for step := range 600 {
		n := step % len(seqs)
		s := seqs[n]
		tape := autograd.NewTape()
		tctx := tape.Context()
		logits, err := heads.Logits(tctx, hiddens[n])
		if err != nil {
			t.Fatal(err)
		}
		var total *tensor.Tensor
		for k := range K {
			rows := len(s) - 2 - k // positions t with a target at t+2+k
			if rows <= 0 {
				continue
			}
			targets := tensor.New(tensor.F32, tensor.Shape{rows})
			for i := range rows {
				targets.SetF64(float64(s[i+2+k]), i)
			}
			sub, err := backend.Execute(tctx, backend.OpSlice, []*tensor.Tensor{logits[k]},
				backend.SliceAttrs{Axis: 0, Start: 0, End: rows})
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(tctx, sub[0], targets)
			if err != nil {
				t.Fatal(err)
			}
			if total == nil {
				total = loss
			} else {
				out, err := backend.Execute(tctx, backend.OpAdd, []*tensor.Tensor{total, loss}, nil)
				if err != nil {
					t.Fatal(err)
				}
				total = out[0]
			}
		}
		lv := total.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(total); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(heads.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	// bar: ≥30% below the uniform start (first ≈ K·ln V). LINEAR heads on hidden states that
	// mix token and position information don't reach memorization here — the assertion is that
	// gradients flow through the frozen-base setup and the heads genuinely learn, not head SOTA.
	if last >= first*0.7 {
		t.Fatalf("Medusa heads did not train: CE %.4f → %.4f", first, last)
	}
	t.Logf("Medusa heads trained on frozen base: CE %.4f → %.4f (600 steps, K=%d)", first, last, K)
}
