//go:build darwin && cgo

package metal

import (
	"sort"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Interleaved same-process A/B: standalone matmul with zero-copy operand wrapping
// (newBufferWithBytesNoCopy over the page-truncated enclosing range + MPSMatrix
// byte offsets) vs the pooled-copy path (3 host memcpys ≈330µs at 1024³). Rounds
// alternate tightly so thermal/warmup drift cancels; medians reported.
func TestMatMulNoCopyAB(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	if testing.Short() {
		t.Skip("perf A/B — skipped in -short")
	}
	be, _ := backend.Get(backend.Metal)
	ctx := backend.NewContext().WithBackend(be)

	for _, sz := range []int{512, 1024} {
		x := bench.RandF32(tensor.Shape{sz, sz}, 1)
		y := bench.RandF32(tensor.Shape{sz, sz}, 2)
		ins := []*tensor.Tensor{x, y}
		run := func(iters int) float64 { // µs/op
			s := time.Now()
			for range iters {
				if _, err := backend.Execute(ctx, backend.OpMatMul, ins, nil); err != nil {
					t.Fatal(err)
				}
			}
			return float64(time.Since(s).Microseconds()) / float64(iters)
		}
		setMatMulNoCopy(true)
		run(30)
		setMatMulNoCopy(false)
		run(30)

		const rounds, iters = 6, 60
		var noCopy, pooled []float64
		for r := 0; r < rounds; r++ {
			setMatMulNoCopy(true)
			noCopy = append(noCopy, run(iters))
			setMatMulNoCopy(false)
			pooled = append(pooled, run(iters))
		}
		setMatMulNoCopy(true)
		med := func(v []float64) float64 {
			sort.Float64s(v)
			return v[len(v)/2]
		}
		gf := 2 * float64(sz) * float64(sz) * float64(sz) / 1e3
		t.Logf("%d: nocopy median %.1f us/op (%.0f GFLOP/s) | pooled median %.1f us/op (%.0f GFLOP/s) | speedup %.2fx  rounds nocopy=%v pooled=%v",
			sz, med(noCopy), gf/med(noCopy), med(pooled), gf/med(pooled), med(pooled)/med(noCopy), noCopy, pooled)
	}
}
