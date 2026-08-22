package nn_test

import (
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type quantSwiGLURecorder struct {
	ops []backend.Op
}

func (r *quantSwiGLURecorder) Record(op backend.Op, _, _ []*tensor.Tensor, _ backend.Attrs) {
	r.ops = append(r.ops, op)
}

func TestQuantSwiGLUInPlaceFusionMatchesRecordedFallback(t *testing.T) {
	const dim = 32
	mkLinear := func(seed float64) *nn.QuantLinear {
		data := make([]float32, dim*dim)
		for i := range data {
			data[i] = float32(math.Sin(float64(i)*0.13+seed) * 0.2)
		}
		weight := tensor.FromFloat32(tensor.Shape{dim, dim}, data)
		raw, err := gguf.Quantize(weight, gguf.Q8_0)
		if err != nil {
			t.Fatal(err)
		}
		return &nn.QuantLinear{Weight: raw, QT: gguf.Q8_0, In: dim, Out: dim}
	}
	block := &nn.QuantSwiGLU{
		Gate: mkLinear(0.1),
		Up:   mkLinear(0.2),
		Down: mkLinear(0.3),
	}
	xData := make([]float32, dim)
	for i := range xData {
		xData[i] = float32(math.Cos(float64(i)*0.19) * 0.5)
	}
	x := tensor.FromFloat32(tensor.Shape{1, dim}, xData)
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend is not registered")
	}
	ctx := backend.NewContext().WithBackend(be)
	eager, err := block.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	recorder := &quantSwiGLURecorder{}
	recorded, err := block.Forward(ctx.WithRecorder(recorder), x)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(recorder.ops, backend.OpSiLU) || !slices.Contains(recorder.ops, backend.OpMul) {
		t.Fatalf("recorded fallback ops = %v, want SiLU and Mul", recorder.ops)
	}
	for i, value := range eager.Storage().F32() {
		if math.Float32bits(value) != math.Float32bits(recorded.Storage().F32()[i]) {
			t.Fatalf("output %d = %08x, recorded fallback %08x", i, math.Float32bits(value), math.Float32bits(recorded.Storage().F32()[i]))
		}
		if math.Float32bits(x.Storage().F32()[i]) != math.Float32bits(xData[i]) {
			t.Fatalf("public input element %d was mutated", i)
		}
	}
}
