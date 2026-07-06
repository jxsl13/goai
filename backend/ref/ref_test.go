package ref

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func idKernel(_ *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	return in, nil
}

// The reference backend self-registers as the truth/fallback backend (§V9, §I4).
func TestSelfRegistered(t *testing.T) {
	if backend.Reference() == nil || backend.Reference().Name() != "ref" {
		t.Fatalf("ref must register as the reference backend, got %v", backend.Reference())
	}
	if std.Device().Kind() != tensor.KindCPU {
		t.Error("reference device must be CPU")
	}
	if err := std.Synchronize(); err != nil {
		t.Errorf("reference Synchronize must be a no-op: %v", err)
	}
}

// add + Kernel lookup on a throwaway instance (avoids mutating std / colliding
// with real kernels registered by §T6+).
func TestAddAndLookup(t *testing.T) {
	b := &Backend{table: make(map[kernelKey]backend.Kernel)}
	b.add(backend.OpNeg, tensor.F64, idKernel)
	if _, ok := b.Kernel(backend.OpNeg, tensor.F64); !ok {
		t.Error("registered kernel must be found")
	}
	if _, ok := b.Kernel(backend.OpAdd, tensor.F64); ok {
		t.Error("unregistered kernel must not be found")
	}
}

func TestAddDuplicatePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("duplicate kernel registration must panic")
		}
	}()
	b := &Backend{table: make(map[kernelKey]backend.Kernel)}
	b.add(backend.OpNeg, tensor.F64, idKernel)
	b.add(backend.OpNeg, tensor.F64, idKernel)
}
