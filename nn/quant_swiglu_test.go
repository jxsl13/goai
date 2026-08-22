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
	for _, tc := range []struct {
		name string
		qt   gguf.QuantType
		dim  int
	}{
		{name: "Q8_0 separate projections", qt: gguf.Q8_0, dim: 32},
		{name: "Q4_K paired projections", qt: gguf.Q4_K, dim: 256},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mkLinear := func(seed float64) *nn.QuantLinear {
				data := make([]float32, tc.dim*tc.dim)
				for i := range data {
					data[i] = float32(math.Sin(float64(i)*0.13+seed) * 0.2)
				}
				weight := tensor.FromFloat32(tensor.Shape{tc.dim, tc.dim}, data)
				raw, err := gguf.Quantize(weight, tc.qt)
				if err != nil {
					t.Fatal(err)
				}
				return &nn.QuantLinear{Weight: raw, QT: tc.qt, In: tc.dim, Out: tc.dim}
			}
			block := &nn.QuantSwiGLU{
				Gate: mkLinear(0.1),
				Up:   mkLinear(0.2),
				Down: mkLinear(0.3),
			}
			xData := make([]float32, tc.dim)
			for i := range xData {
				xData[i] = float32(math.Cos(float64(i)*0.19) * 0.5)
			}
			x := tensor.FromFloat32(tensor.Shape{1, tc.dim}, xData)
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
		})
	}
}
