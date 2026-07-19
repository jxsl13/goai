package nlp_test

import (
	"sync/atomic"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

const (
	offloadFastName backend.Name = "nlp-offload-fast-test"
	offloadHostName backend.Name = "nlp-offload-host-test"
)

// countingBackend delegates exact reference kernels while recording dispatch.
// The output is therefore parity-comparable while the counters prove that both
// sides of the simulated device→host layer split actually executed.
type countingBackend struct {
	name     backend.Name
	delegate backend.Backend
	calls    atomic.Int64
}

func (b *countingBackend) Name() backend.Name    { return b.name }
func (b *countingBackend) Device() tensor.Device { return b.delegate.Device() }
func (b *countingBackend) Synchronize() error    { return b.delegate.Synchronize() }
func (b *countingBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	k, ok := b.delegate.Kernel(op, dt)
	if !ok {
		return nil, false
	}
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		b.calls.Add(1)
		return k(ctx.WithBackend(b.delegate), in, attrs)
	}, true
}

func simulatedSplitPlan(t *testing.T, layers int) backend.OffloadPlan {
	t.Helper()
	sizes := make([]int64, layers)
	for i := range sizes {
		sizes[i] = 1 << 20
	}
	plan, err := backend.PlanOffload(backend.OffloadConfig{
		LayerBytes: sizes,
		Budgets:    map[backend.Name]int64{offloadFastName: 1 << 20},
		Prefer:     []backend.Name{offloadFastName},
		Fallback:   offloadHostName,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.OnFallback != layers-1 {
		t.Fatalf("simulated 1-layer device budget spilled %d/%d layers", plan.OnFallback, layers)
	}
	return plan
}

func requireTensorExact(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) || got.Dtype() != want.Dtype() {
		t.Fatalf("tensor metadata got %v/%v want %v/%v", got.Shape(), got.Dtype(), want.Shape(), want.Dtype())
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		if g, w := got.AtF64(idx...), want.AtF64(idx...); g != w {
			t.Fatalf("tensor[%d] got %.17g want %.17g", i, g, w)
		}
	}
}

// T631 C24 E2E: the shared PlanOffload policy drives real GPT and Llama block
// loops. A two-layer model exceeds the simulated one-layer device budget, both
// selected backends run, and the split is bit-exact with all-reference output.
func TestOffloadPlanRunsGPTAndLlamaLayers(t *testing.T) {
	ref := backend.Reference()
	fast := &countingBackend{name: offloadFastName, delegate: ref}
	host := &countingBackend{name: offloadHostName, delegate: ref}
	backend.Register(fast)
	backend.Register(host)
	ctx := &backend.Context{Backend: ref}

	gpt, golden := loadGPT(t)
	wantGPT, err := gpt.Forward(ctx, golden.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	gpt.LayerBackends = simulatedSplitPlan(t, len(gpt.Blocks)).Layers
	gotGPT, err := gpt.Forward(ctx, golden.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	requireTensorExact(t, gotGPT, wantGPT)
	if fast.calls.Load() == 0 || host.calls.Load() == 0 {
		t.Fatalf("GPT split did not execute both sides: fast=%d host=%d", fast.calls.Load(), host.calls.Load())
	}

	fast.calls.Store(0)
	host.calls.Store(0)
	llama, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 11, Ctx: 8, Dim: 8, Heads: 2, KVHeads: 1, Layers: 2, Hidden: 16, Eps: 1e-5,
	}, 31)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{1, 4, 7}
	wantLlama, err := llama.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	llama.LayerBackends = simulatedSplitPlan(t, len(llama.Blocks)).Layers
	gotLlama, err := llama.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	requireTensorExact(t, gotLlama, wantLlama)
	if fast.calls.Load() == 0 || host.calls.Load() == 0 {
		t.Fatalf("Llama split did not execute both sides: fast=%d host=%d", fast.calls.Load(), host.calls.Load())
	}

	llama.LayerBackends = llama.LayerBackends[:1]
	if _, err := llama.Forward(ctx, tokens); err == nil {
		t.Fatal("malformed layer plan must fail before partial execution")
	}
}
