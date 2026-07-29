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

// The fused per-head query gather in attnAbsorbed is claimed bit-identical to the three slice
// dispatches it replaces, so the arms must agree on RAW BITS. They are selected by
// ctx.Recorder — a plain context takes the fused gather, a taped one keeps the dispatch path —
// so comparing them exercises the arithmetic and the tape guard together.
func TestQuantDeepSeekV2FusedDecodeMatchesDispatch(t *testing.T) {
	m := newQuantTestDeepSeekV2()
	raw := quantDeepSeekV2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
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
		t.Fatalf("length mismatch: fused %d, dispatch %d", len(fused), len(dispatch))
	}
	for i := range fused {
		if math.Float32bits(fused[i]) != math.Float32bits(dispatch[i]) {
			t.Fatalf("logit %d differs: fused %08x, dispatch %08x — the gather moves the same floats "+
				"into the same kernels and must not change a value",
				i, math.Float32bits(fused[i]), math.Float32bits(dispatch[i]))
		}
	}
}
