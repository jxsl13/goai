package backend

import (
	"sync/atomic"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

type cacheProbeBackend struct {
	lookups atomic.Int32
}

type cacheProbeWrapper struct {
	Backend
	lookups atomic.Int32
}

func (w *cacheProbeWrapper) Kernel(op Op, dtype tensor.Dtype) (Kernel, bool) {
	w.lookups.Add(1)
	return w.Backend.Kernel(op, dtype)
}

var registeredCacheProbe = &cacheProbeBackend{}

func init() { Register(registeredCacheProbe) }

func (*cacheProbeBackend) Name() Name            { return "cache-probe" }
func (*cacheProbeBackend) Device() tensor.Device { return tensor.CPU() }
func (*cacheProbeBackend) Synchronize() error    { return nil }
func (b *cacheProbeBackend) Kernel(Op, tensor.Dtype) (Kernel, bool) {
	b.lookups.Add(1)
	return func(_ *Context, inputs []*tensor.Tensor, _ Attrs) ([]*tensor.Tensor, error) {
		return inputs, nil
	}, true
}

func TestExecuteResolutionCacheAndGenerationInvalidation(t *testing.T) {
	probe := registeredCacheProbe
	probe.lookups.Store(0)
	ctx := &Context{Backend: probe}
	input := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	for range 2 {
		if _, err := Execute(ctx, OpNeg, []*tensor.Tensor{input}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if got := probe.lookups.Load(); got != 1 {
		t.Fatalf("warmed resolver performed %d Kernel lookups, want 1", got)
	}

	preference := Preference()
	defer SetPreference(preference...)
	SetPreference(preference...)
	if _, err := Execute(ctx, OpNeg, []*tensor.Tensor{input}, nil); err != nil {
		t.Fatal(err)
	}
	if got := probe.lookups.Load(); got != 2 {
		t.Fatalf("registry generation change left a stale entry: Kernel lookups=%d, want 2", got)
	}
}

func TestRegisteredDispatchTableRequiresExactBackendIdentity(t *testing.T) {
	if table := registeredDispatchTable(activeMock); table == nil {
		t.Fatal("registered backend did not receive its shared dispatch table")
	}
	wrapped := &cacheProbeWrapper{Backend: activeMock}
	if table := registeredDispatchTable(wrapped); table != nil {
		t.Fatal("same-name backend wrapper incorrectly received the registered backend's dispatch table")
	}
	ctx := &Context{Backend: wrapped}
	input := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	for range 2 {
		if _, err := Execute(ctx, OpAdd, []*tensor.Tensor{input}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if ctx.dispatch.Load() != uncachedDispatchTable {
		t.Fatal("same-name backend wrapper did not remain on the live-dispatch bypass")
	}
	if got := wrapped.lookups.Load(); got != 2 {
		t.Fatalf("same-name backend wrapper performed %d live Kernel lookups, want 2", got)
	}
}
