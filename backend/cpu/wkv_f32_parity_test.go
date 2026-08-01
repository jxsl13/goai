package cpu

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestWKVF32CPUByteIdenticalToRef locks the fresh F32 CPU WKV fast path
// (wkvParallelScanF32) to backend/ref's F32 typed scan. The CPU kernel mirrors the
// ref recurrence exactly — F32 reads widened to F64, running state kept in F64, only
// the store rounded to F32 — and merely fans the independent channels across
// goroutines. Since each channel is an independent recurrence writing a disjoint
// output column, the parallel scan must be BYTE-IDENTICAL to the serial ref, not
// merely within f32 tolerance. Sizes are chosen so the parallel path fires
// (seq*d >= 1<<14 and per-worker chunk < d).
func TestWKVF32CPUByteIdenticalToRef(t *testing.T) {
	cpuB, _ := backend.Get(backend.CPU)
	refB, _ := backend.Get(backend.Ref)
	for _, sz := range [][2]int{{64, 256}, {128, 512}, {512, 1024}, {13, 1291}} {
		seq, d := sz[0], sz[1]
		in := []*tensor.Tensor{
			bench.RandF32(tensor.Shape{seq, d}, 1),
			bench.RandF32(tensor.Shape{seq, d}, 2),
			bench.RandF32(tensor.Shape{d}, 3),
			bench.RandF32(tensor.Shape{d}, 4),
		}
		gc, err := backend.Execute(backend.NewContext().WithBackend(cpuB), backend.OpWKV, in, nil)
		if err != nil {
			t.Fatalf("cpu wkv [%d,%d]: %v", seq, d, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(refB), backend.OpWKV, in, nil)
		if err != nil {
			t.Fatalf("ref wkv [%d,%d]: %v", seq, d, err)
		}
		cs, rs := gc[0].Storage().F32(), gr[0].Storage().F32()
		if len(cs) != len(rs) {
			t.Fatalf("[%d,%d] len mismatch %d vs %d", seq, d, len(cs), len(rs))
		}
		for i := range cs {
			if cs[i] != rs[i] {
				t.Fatalf("[%d,%d] byte mismatch at %d: cpu=%v ref=%v", seq, d, i, cs[i], rs[i])
			}
		}
	}
}
