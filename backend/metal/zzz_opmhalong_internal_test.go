//go:build darwin && cgo

package metal

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func TestOpMHADecodeLongContext(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const sk, heads, dk = 1920, 8, 64
	const dm = heads * dk
	q := bench.RandF32(tensor.Shape{1, dm}, 1)
	k := bench.RandF32(tensor.Shape{sk, dm}, 2)
	v := bench.RandF32(tensor.Shape{sk, dm}, 3)
	ctx := backend.NewContext().WithBackend(Backend{})
	attrs := backend.AttnAttrs{Heads: heads, Causal: true}
	ins := []*tensor.Tensor{q, k, v}
	if _, err := backend.Execute(ctx, backend.OpMHA, ins, attrs); err != nil {
		t.Fatal(err)
	}
	const it = 30
	s := time.Now()
	for range it {
		if _, err := backend.Execute(ctx, backend.OpMHA, ins, attrs); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("per-op OpMHA sq=1 sk=%d: %.2f ms/op (was ~40ms via the serial two-pass kernel, §T428 profile)", sk, float64(time.Since(s).Microseconds())/it/1000)
}
