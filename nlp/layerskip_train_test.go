package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func layerSkipTargets(ids ...int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{len(ids)})
	for i, id := range ids {
		t.SetF64(float64(id), i)
	}
	return t
}

func layerSkipTiny(t *testing.T, layers int) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 6, Ctx: 8, Dim: 4, Heads: 1, KVHeads: 1,
		Layers: layers, Hidden: 6, Eps: 1e-5, RopeBase: 10000,
	}, 871)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// §R264 Eq.2-4: the schedule values are checked against a direct hand
// evaluation, including both endpoints and the from-scratch time curriculum.
func TestLayerSkipDropoutRatesPaperEquations(t *testing.T) {
	cfg := nlp.LayerSkipTrainConfig{
		MaxDropout: 0.2, FromScratch: true, Step: 50, TotalSteps: 100,
	}
	got, err := cfg.DropoutRates(5)
	if err != nil {
		t.Fatal(err)
	}
	timeScale := math.Sqrt(2) - 1
	for layer := range got {
		depthScale := math.Exp2(float64(layer)/4) - 1
		want := 0.2 * timeScale * depthScale
		if math.Abs(got[layer]-want) > 1e-15 {
			t.Fatalf("layer %d rate %.17g want %.17g", layer, got[layer], want)
		}
	}
	if got[0] != 0 {
		t.Fatalf("first-layer dropout = %g, want exact zero", got[0])
	}

	finetune, err := (nlp.LayerSkipTrainConfig{MaxDropout: 0.2}).DropoutRates(5)
	if err != nil {
		t.Fatal(err)
	}
	if finetune[4] != 0.2 {
		t.Fatalf("finetune final-layer dropout = %g, want pmax", finetune[4])
	}
	complete, err := (nlp.LayerSkipTrainConfig{
		MaxDropout: 0.2, FromScratch: true, Step: 200, TotalSteps: 100,
	}).DropoutRates(5)
	if err != nil {
		t.Fatal(err)
	}
	if complete[4] != 0.2 {
		t.Fatalf("completed curriculum rate = %g, want pmax", complete[4])
	}
}

// §R264 Eq.5-6: raw [0, .25, .75, 1.5, 5.5] weights normalize to the
// hand-computed vector below. Rotation then assigns every positive-weight
// layer to exactly one of R consecutive steps.
func TestLayerSkipExitWeightsPaperEquationsAndRotation(t *testing.T) {
	got, err := (nlp.LayerSkipTrainConfig{ExitScale: 0.25}).ExitWeights(5)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 0.03125, 0.09375, 0.1875, 0.6875}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-15 {
			t.Fatalf("weight[%d] %.17g want %.17g", i, got[i], want[i])
		}
	}

	seen := make([]int, 8)
	for step := range 3 {
		weights, err := (nlp.LayerSkipTrainConfig{
			ExitScale: 1, Rotation: 3, Step: step,
		}).ExitWeights(8)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for layer, w := range weights {
			sum += w
			if w != 0 {
				if layer%3 != step {
					t.Fatalf("step %d enabled layer %d outside its rotated phase", step, layer)
				}
				seen[layer]++
			}
		}
		if math.Abs(sum-1) > 1e-15 {
			t.Fatalf("step %d normalized sum %.17g", step, sum)
		}
	}
	if seen[0] != 0 { // Eq.5 deliberately gives block 0 zero direct exit mass.
		t.Fatalf("layer 0 received direct exit weight %d times", seen[0])
	}
	for layer := 1; layer < 8; layer++ {
		if seen[layer] != 1 {
			t.Fatalf("layer %d enabled %d times across one rotation, want 1", layer, seen[layer])
		}
	}
}

func layerSkipLossAndGrads(t *testing.T, m *nlp.Llama, tokens []int, targets *tensor.Tensor, recipe bool) (*tensor.Tensor, []*tensor.Tensor) {
	t.Helper()
	tape := autograd.NewTape()
	ctx := tape.Context().WithBackend(backend.Reference())
	var loss *tensor.Tensor
	var err error
	if recipe {
		loss, err = m.LayerSkipLoss(ctx, tokens, targets, nlp.LayerSkipTrainConfig{})
	} else {
		var logits *tensor.Tensor
		logits, err = m.Forward(ctx, tokens)
		if err == nil {
			loss, err = nn.CrossEntropy(ctx, logits, targets)
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	grads := make([]*tensor.Tensor, len(m.Params()))
	for i, p := range m.Params() {
		grads[i] = tape.Grad(p)
	}
	return loss, grads
}

// The zero-value recipe must be a true collapse, not merely close: the same
// Qwen2 biases, Qwen3 QK norms, Granite scalars and per-layer backend route run
// through the shared block path and yield bit-identical loss and every gradient.
func TestLayerSkipZeroConfigBitExactFinalCE(t *testing.T) {
	m := qwenGraniteLlama(t)
	m.LayerBackends = []backend.Name{backend.Ref, backend.Ref}
	tokens := []int{1, 3, 2, 5}
	targets := layerSkipTargets(3, 2, 5, 1)
	wantLoss, wantGrads := layerSkipLossAndGrads(t, m, tokens, targets, false)
	gotLoss, gotGrads := layerSkipLossAndGrads(t, m, tokens, targets, true)
	if math.Float64bits(gotLoss.AtF64()) != math.Float64bits(wantLoss.AtF64()) {
		t.Fatalf("zero-config loss %.17g want bit-exact %.17g", gotLoss.AtF64(), wantLoss.AtF64())
	}
	for p := range gotGrads {
		if gotGrads[p] == nil || wantGrads[p] == nil {
			t.Fatalf("param %d nil gradient: recipe=%v ordinary=%v", p, gotGrads[p] == nil, wantGrads[p] == nil)
		}
		for i := range gotGrads[p].Numel() {
			idx := tensor.Unravel(i, gotGrads[p].Shape())
			g, w := gotGrads[p].AtF64(idx...), wantGrads[p].AtF64(idx...)
			if math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("param %d grad[%d] %.17g want bit-exact %.17g", p, i, g, w)
			}
		}
	}
}

// With pmax=1 the final block's paper depth rate is exactly one while block 0
// is exactly zero. The dropped block becomes an identity, receives no gradient,
// and the result equals an explicit block-0 early exit bit-for-bit.
func TestLayerSkipWholeBlockDropoutAndGradientIsolation(t *testing.T) {
	m := layerSkipTiny(t, 2)
	tokens := []int{1, 4, 2}
	targets := layerSkipTargets(4, 2, 1)

	wantLogits, err := m.ForwardExit(backend.NewContext(), tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := nn.CrossEntropy(backend.NewContext(), wantLogits, targets)
	if err != nil {
		t.Fatal(err)
	}

	tape := autograd.NewTape()
	loss, err := m.LayerSkipLoss(tape.Context(), tokens, targets, nlp.LayerSkipTrainConfig{MaxDropout: 1, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if math.Float64bits(loss.AtF64()) != math.Float64bits(want.AtF64()) {
		t.Fatalf("dropped-final loss %.17g want explicit early exit %.17g", loss.AtF64(), want.AtF64())
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	for name, p := range map[string]*tensor.Tensor{
		"attn_norm": m.Blocks[1].AttnNorm.Gamma,
		"wq":        m.Blocks[1].Wq,
		"wv":        m.Blocks[1].Wv,
		"wo":        m.Blocks[1].Wo,
		"ffn_norm":  m.Blocks[1].FFNNorm.Gamma,
		"ffn_gate":  m.Blocks[1].FFN.Wgate,
		"ffn_down":  m.Blocks[1].FFN.Wdown,
	} {
		if g := tape.Grad(p); g != nil {
			t.Fatalf("dropped final block %s received gradient", name)
		}
	}
	if tape.Grad(m.Blocks[0].Wv) == nil {
		t.Fatal("kept block received no gradient")
	}
	if tape.Grad(m.Out) == nil || tape.Grad(m.Norm.Gamma) == nil {
		t.Fatal("shared final norm/head received no gradient")
	}
}

// §V2: central differences over EVERY parameter element validate the complete
// multi-exit graph (two shared-head losses, all three blocks, shared norm/head).
func TestLayerSkipFullParameterGradcheck(t *testing.T) {
	m := layerSkipTiny(t, 3)
	tokens := []int{1, 4}
	targets := layerSkipTargets(4, 2)
	cfg := nlp.LayerSkipTrainConfig{ExitScale: 0.5}

	tape := autograd.NewTape()
	loss, err := m.LayerSkipLoss(tape.Context(), tokens, targets, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for pi, p := range m.Params() {
		grad := tape.Grad(p)
		if grad == nil {
			t.Fatalf("param %d %v received no gradient", pi, p.Shape())
		}
		for flat := range p.Numel() {
			idx := tensor.Unravel(flat, p.Shape())
			old := p.AtF64(idx...)
			p.SetF64(old+h, idx...)
			plus, err := m.LayerSkipLoss(backend.NewContext(), tokens, targets, cfg)
			if err != nil {
				t.Fatal(err)
			}
			p.SetF64(old-h, idx...)
			minus, err := m.LayerSkipLoss(backend.NewContext(), tokens, targets, cfg)
			p.SetF64(old, idx...)
			if err != nil {
				t.Fatal(err)
			}
			numeric := (plus.AtF64() - minus.AtF64()) / (2 * h)
			analytic := grad.AtF64(idx...)
			if math.Abs(numeric-analytic) > 2e-4*math.Max(1, math.Abs(numeric)) {
				t.Fatalf("param %d elem %d: numeric %.8g analytic %.8g", pi, flat, numeric, analytic)
			}
		}
	}
}

// The recipe is useful end-to-end, not just algebraically connected: training
// a tiny three-block model lowers the directly supervised block-1 exit CE.
func TestLayerSkipTrainingImprovesEarlyExit(t *testing.T) {
	m := layerSkipTiny(t, 3)
	tokens := []int{1, 3, 2, 5}
	targets := layerSkipTargets(3, 2, 5, 1)
	earlyCE := func() float64 {
		logits, err := m.ForwardExit(backend.NewContext(), tokens, 1)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(backend.NewContext(), logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		return loss.AtF64()
	}
	first := earlyCE()
	opt := nn.NewAdam(m.Params(), 0.02)
	for step := range 60 {
		tape := autograd.NewTape()
		loss, err := m.LayerSkipLoss(tape.Context(), tokens, targets, nlp.LayerSkipTrainConfig{
			MaxDropout: 0.2, ExitScale: 1, Step: step, Seed: 871,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	last := earlyCE()
	if last >= first*0.5 {
		t.Fatalf("early-exit CE did not improve enough: %.4f -> %.4f", first, last)
	}
	t.Logf("LayerSkip early-exit CE %.4f -> %.4f", first, last)
}

func TestLayerSkipTrainingErrors(t *testing.T) {
	m := layerSkipTiny(t, 3)
	tokens := []int{1, 2}
	targets := layerSkipTargets(2, 1)
	ctx := backend.NewContext()
	var nilModel *nlp.Llama
	if _, err := nilModel.LayerSkipLoss(ctx, tokens, targets, nlp.LayerSkipTrainConfig{}); err == nil {
		t.Fatal("nil model accepted")
	}
	if _, err := m.LayerSkipLoss(ctx, tokens, nil, nlp.LayerSkipTrainConfig{}); err == nil {
		t.Fatal("nil targets accepted")
	}
	bad := []nlp.LayerSkipTrainConfig{
		{MaxDropout: -0.1}, {MaxDropout: 1.1},
		{ExitScale: -0.1}, {ExitScale: 1.1},
		{Step: -1}, {Rotation: 4},
		{FromScratch: true, TotalSteps: 0},
	}
	for i, cfg := range bad {
		if _, err := m.LayerSkipLoss(ctx, tokens, targets, cfg); err == nil {
			t.Fatalf("bad config %d accepted: %+v", i, cfg)
		}
	}
	if _, err := (nlp.LayerSkipTrainConfig{ExitScale: 1, Rotation: 2}).ExitWeights(2); err == nil {
		t.Fatal("zero-mass rotated exit set accepted")
	}
	if _, err := (nlp.LayerSkipTrainConfig{}).DropoutRates(0); err == nil {
		t.Fatal("zero layers accepted")
	}
	if _, err := m.LayerSkipLoss(ctx, []int{99}, layerSkipTargets(1), nlp.LayerSkipTrainConfig{}); err == nil {
		t.Fatal("out-of-vocab token accepted")
	}
	if _, err := m.LayerSkipLoss(ctx, tokens, layerSkipTargets(1), nlp.LayerSkipTrainConfig{}); err == nil {
		t.Fatal("target length mismatch accepted")
	}

	broken := layerSkipTiny(t, 3)
	broken.Config.Layers = 2
	if _, err := broken.LayerSkipLoss(ctx, tokens, targets, nlp.LayerSkipTrainConfig{}); err == nil {
		t.Fatal("config/block layer mismatch accepted")
	}
	m.LayerBackends = []backend.Name{backend.Ref}
	if _, err := m.LayerSkipLoss(ctx, tokens, targets, nlp.LayerSkipTrainConfig{}); err == nil {
		t.Fatal("malformed layer backend plan accepted")
	}
}
