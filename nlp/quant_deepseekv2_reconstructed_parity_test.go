package nlp_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// The fused per-head gathers in attnReconstructed are claimed bit-identical to the six slice
// dispatches they replace. The arms are selected by ctx.Recorder, so one raw-bit comparison
// covers the layout algebra and the tape guard together.
//
// Reuse of the value buffer across heads is safe because rowBuf.Append COPIES its row argument
// into its own backing (copyRows) and returns a view of that backing — checked in the source
// rather than assumed, since retaining it would alias every head's value block in the cache.
func TestQuantDeepSeekV2ReconstructedFusedMatchesDispatch(t *testing.T) {
	raw := legacyQuantDeepSeekV2GGUFBytes(t, newQuantTestDeepSeekV2())
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if q.Blocks[0].WkvB == nil {
		t.Fatal("fixture took the absorbed path; this test would not exercise attnReconstructed")
	}
	toks := []int{1, 3, 2, 5, 4}
	run := func(ctx *backend.Context) []float32 {
		cache := q.NewCache()
		var last []float32
		for pos, tok := range toks {
			out, err := q.DecodeStep(ctx, cache, tok, pos)
			if err != nil {
				t.Fatal(err)
			}
			last = append([]float32(nil), out.Contiguous().Storage().F32()...)
		}
		return last
	}
	fused := run(backend.NewContext())
	dispatch := run(autograd.NewTape().Context())
	if len(fused) != len(dispatch) {
		t.Fatalf("length %d vs %d", len(fused), len(dispatch))
	}
	for i := range fused {
		if math.Float32bits(fused[i]) != math.Float32bits(dispatch[i]) {
			t.Fatalf("logit %d differs: fused %08x, dispatch %08x", i,
				math.Float32bits(fused[i]), math.Float32bits(dispatch[i]))
		}
	}
}
