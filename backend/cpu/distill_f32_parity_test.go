package cpu

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestDistillF32CPUByteIdenticalToRef locks the fresh F32 CPU distill fast path
// (distillRowsF32) to backend/ref's F32 scan. The CPU kernel mirrors ref exactly — F32
// reads widened to F64, per-row stable softmax via z/temp division, KL = Σ p·(log p −
// log q) over p>0 — and merely fans the independent rows across goroutines, summing the
// per-row contributions back in index order. Since each row is an independent KL
// contribution and the final reduction is ordered, the scalar loss must be BYTE-IDENTICAL
// to the serial ref. Sizes fire the parallel path (b*c >= 1<<13, per-worker chunk < b),
// including a non-power-of-two b.
func TestDistillF32CPUByteIdenticalToRef(t *testing.T) {
	cpuB, _ := backend.Get(backend.CPU)
	refB, _ := backend.Get(backend.Ref)
	attrs := backend.DistillAttrs{Temperature: 2.0}
	for _, sz := range [][2]int{{16, 512}, {64, 4096}, {33, 1291}, {256, 8000}} {
		b, c := sz[0], sz[1]
		in := []*tensor.Tensor{
			bench.RandF32(tensor.Shape{b, c}, 1),
			bench.RandF32(tensor.Shape{b, c}, 2),
		}
		gc, err := backend.Execute(backend.NewContext().WithBackend(cpuB), backend.OpDistill, in, attrs)
		if err != nil {
			t.Fatalf("cpu distill [%d,%d]: %v", b, c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(refB), backend.OpDistill, in, attrs)
		if err != nil {
			t.Fatalf("ref distill [%d,%d]: %v", b, c, err)
		}
		cv, rv := gc[0].Storage().F32()[0], gr[0].Storage().F32()[0]
		if cv != rv {
			t.Fatalf("[%d,%d] byte mismatch: cpu=%v ref=%v", b, c, cv, rv)
		}
	}
}
