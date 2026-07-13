package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func qlRand(n, seed int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(float64(i*13+seed)*0.21+0.3) * 2
	}
	return x
}

// §V3/§V11 (§T142): QuantLinear.Forward with a GPU-accelerated quant type (Q4_K) matches the
// CPU reference gguf.QMatMul within crossTol (the active backend runs the in-kernel matmul when
// it is a QuantMatMuler; on this host Default() is Metal), and approximates the full-precision
// matmul within the Q4_K quant error — proving the accelerated quantized linear layer is wired
// end to end.
func TestQuantLinearForwardAccelerated(t *testing.T) {
	const m, in, out = 4, 256, 8
	xf := tensor.New(tensor.F32, tensor.Shape{m, in})
	xr := qlRand(m*in, 1)
	for i := range m * in {
		xf.Storage().F32()[i] = float32(xr[i])
	}
	wf := qlRand(out*in, 2)
	w := tensor.FromFloat64(tensor.Shape{out, in}, wf)
	wq, err := gguf.Quantize(w, gguf.Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	ql := &nn.QuantLinear{Weight: wq, QT: gguf.Q4_K, In: in, Out: out}
	got, err := ql.Forward(backend.NewContext(), xf)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Shape().Equal(tensor.Shape{m, out}) {
		t.Fatalf("shape %v, want [%d,%d]", got.Shape(), m, out)
	}
	ref, _ := gguf.QMatMul(xf, wq, gguf.Q4_K, out, in)
	rtol, atol := 1e-6*math.Sqrt(float64(in)), 1e-4
	for mi := range m {
		for oi := range out {
			g, r := got.AtF64(mi, oi), ref.AtF64(mi, oi)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
				t.Errorf("[%d,%d] QuantLinear %g vs gguf.QMatMul %g (Δ %g)", mi, oi, g, r, math.Abs(g-r))
			}
			// vs full precision, within the Q4_K quant error
			var want float64
			for ki := range in {
				want += xf.AtF64(mi, ki) * wf[oi*in+ki]
			}
			if e := math.Abs(g - want); e > 0.06*math.Max(1, math.Abs(want)) {
				t.Errorf("[%d,%d] QuantLinear %g vs full-precision %g (err %g)", mi, oi, g, want, e)
			}
		}
	}
}

// §T142: for a quant type the active backend does NOT accelerate (Q4_0 — a legacy format with
// no GPU kernel), QuantLinear falls back to the CPU reference — the result is exactly
// gguf.QMatMul (identical code path), proving the ErrQuantUnsupported fallback selects CPU
// rather than erroring.
func TestQuantLinearForwardCPUFallback(t *testing.T) {
	const m, in, out = 2, 256, 5
	xf := tensor.New(tensor.F32, tensor.Shape{m, in})
	for i := range m * in {
		xf.Storage().F32()[i] = float32(qlRand(m*in, 3)[i])
	}
	w := tensor.FromFloat64(tensor.Shape{out, in}, qlRand(out*in, 4))
	wq, _ := gguf.Quantize(w, gguf.Q4_0) // Q4_0 has no GPU kernel → CPU fallback
	ql := &nn.QuantLinear{Weight: wq, QT: gguf.Q4_0, In: in, Out: out}
	got, err := ql.Forward(backend.NewContext(), xf)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := gguf.QMatMul(xf, wq, gguf.Q4_0, out, in)
	for mi := range m {
		for oi := range out {
			if got.AtF64(mi, oi) != ref.AtF64(mi, oi) {
				t.Errorf("[%d,%d] fallback %g != gguf.QMatMul %g", mi, oi, got.AtF64(mi, oi), ref.AtF64(mi, oi))
			}
		}
	}
}

// §V3/§V11 (§T154): a Q8_0 QuantLinear uploads its weight to a device-resident buffer on the first
// Forward (when the active backend supports it — Metal on this host) and reuses it across calls,
// with every call matching the CPU reference; after Close it correctly falls back to the per-call
// path. This is the decode-loop perf lever wired into the model.
func TestQuantLinearResidentReuse(t *testing.T) {
	const m, in, out = 3, 256, 8
	xf := tensor.New(tensor.F32, tensor.Shape{m, in})
	for i := range m * in {
		xf.Storage().F32()[i] = float32(qlRand(m*in, 6)[i])
	}
	w := tensor.FromFloat64(tensor.Shape{out, in}, qlRand(out*in, 7))
	wq, _ := gguf.Quantize(w, gguf.Q8_0)
	ref, _ := gguf.QMatMul(xf, wq, gguf.Q8_0, out, in)
	ql := &nn.QuantLinear{Weight: wq, QT: gguf.Q8_0, In: in, Out: out}

	check := func(tag string) {
		got, err := ql.Forward(backend.NewContext(), xf)
		if err != nil {
			t.Fatalf("%s: %v", tag, err)
		}
		for mi := range m {
			for oi := range out {
				if g, r := got.AtF64(mi, oi), ref.AtF64(mi, oi); math.Abs(g-r) > 1e-4*math.Max(1, math.Abs(r)) {
					t.Errorf("%s [%d,%d]: %g vs ref %g", tag, mi, oi, g, r)
				}
			}
		}
	}
	check("first (upload)")
	check("second (reuse resident)")
	if err := ql.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	check("after Close (per-call)")
}

// §V15-adjacent: a malformed activation shape is rejected before the matmul.
func TestQuantLinearValidation(t *testing.T) {
	ql := &nn.QuantLinear{Weight: make([]byte, 144), QT: gguf.Q4_K, In: 256, Out: 1}
	if _, err := ql.Forward(backend.NewContext(), tensor.New(tensor.F32, tensor.Shape{2, 64})); err == nil {
		t.Error("x with In=64 (want 256) must be rejected")
	}
}

// A QuantLinear keeps its weight quantized and runs the linear layer without materializing an
// f32 weight matrix — on the GPU when the active backend accelerates the quant type, else on CPU.
func ExampleQuantLinear() {
	w := tensor.FromFloat64(tensor.Shape{4, 256}, qlRand(4*256, 0))
	wq, _ := gguf.Quantize(w, gguf.Q4_K)
	ql := &nn.QuantLinear{Weight: wq, QT: gguf.Q4_K, In: 256, Out: 4}
	x := tensor.New(tensor.F32, tensor.Shape{2, 256})
	y, _ := ql.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape(), len(wq), "weight bytes (vs", 4*256*4, "f32)")
	// Output: (2, 4) 576 weight bytes (vs 4096 f32)
}
