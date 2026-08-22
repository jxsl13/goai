package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

type quantLlamaQKVRecorder struct{}

func (quantLlamaQKVRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

func TestQuantLlamaEagerCPUGroupedQKVMatchesRecordedFallback(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 16, Ctx: 8, Dim: 256, Heads: 8, KVHeads: 4, Layers: 1, Hidden: 256, Eps: 1e-5, RopeBase: 10000,
	}, 23)
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantizeLlama(m, gguf.Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	cpu, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend is not registered")
	}
	ctx := backend.NewContext().WithBackend(cpu)
	compare := func(t *testing.T, eager, recorded *tensor.Tensor) {
		t.Helper()
		if !eager.Shape().Equal(recorded.Shape()) {
			t.Fatalf("eager shape %v != recorded shape %v", eager.Shape(), recorded.Shape())
		}
		for i, value := range eager.Storage().F32() {
			if math.Float32bits(value) != math.Float32bits(recorded.Storage().F32()[i]) {
				t.Fatalf("logit %d differs: eager=%08x recorded=%08x", i, math.Float32bits(value), math.Float32bits(recorded.Storage().F32()[i]))
			}
		}
	}
	t.Run("DecodeStep", func(t *testing.T) {
		eager, err := q.DecodeStep(ctx, q.NewCache(), 3, 0)
		if err != nil {
			t.Fatal(err)
		}
		recorded, err := q.DecodeStep(ctx.WithRecorder(quantLlamaQKVRecorder{}), q.NewCache(), 3, 0)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, eager, recorded)
	})
	t.Run("ForwardM1", func(t *testing.T) {
		eager, err := q.Forward(ctx, []int{3})
		if err != nil {
			t.Fatal(err)
		}
		recorded, err := q.Forward(ctx.WithRecorder(quantLlamaQKVRecorder{}), []int{3})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, eager, recorded)
	})
}
