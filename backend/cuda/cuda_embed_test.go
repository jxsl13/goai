//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// The on-device embedding gather (table[ids[i]] → out[i]) must exactly match the
// reference row gather (it is a copy — bit-exact).
func TestCUDAEmbedMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	const vocab, d = 50, 32
	table := bench.RandF32(tensor.Shape{vocab, d}, 7)
	ids := []int32{0, 3, 49, 12, 3, 0, 7}

	rt, err := cuda.NewResidentB(table)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Free()
	dout, err := rt.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := dout.ToHost()
	dout.Free()

	idx := tensor.New(tensor.F32, tensor.Shape{len(ids)})
	for i, id := range ids {
		idx.SetF64(float64(id), i)
	}
	gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpEmbed, []*tensor.Tensor{table, idx}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Numel() {
		gi := tensor.Unravel(i, got.Shape())
		if g, w := got.AtF64(gi...), gr[0].AtF64(gi...); g != w {
			t.Fatalf("[%d]: embed %v != ref %v", i, g, w)
		}
	}
}
