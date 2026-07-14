//go:build darwin && cgo

package metal_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

func qmRand(n, seed int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(float64(i*13+seed)*0.21+0.3) * 2
	}
	return x
}

// §V3/§V11 (§T136): the Metal Q8_0 quantized matmul equals the reference gguf.QMatMul (the
// f64 CPU truth over the identical dequantized weights) within crossTol — the GPU kernel
// dequantizes the same f16-scale·int8 blocks in-kernel; only the accumulation precision (f32
// vs f64) differs. Covers decode (M=1) and prefill (M>1) over several K.
func TestMetalQMatMulQ8CrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 32, 8}, {1, 256, 16}, {4, 64, 5}, {8, 128, 12}, {2, 512, 7}, {16, 256, 3},
		// Model-scale K (§T624): the quantization error CANCELS here (ref
		// dequantizes the SAME bytes), so the comparison is purely f32-vs-f64
		// accumulation over K. Unlike the dense matmul (§B45), the Q8_0 kernel
		// accumulates block-wise (32-element blocks, scale applied per block), which
		// keeps its f32 error FAR below the raw-GEMM √K growth — measured maxRel
		// ≤4.4e-7 @K4096, ~145× under the (unchanged) tight 1e-6·√k tol. So this tol
		// needs NO recalibration at model scale; the large-K cases simply confirm it.
		{1, 1024, 8}, {4, 2048, 6}, {1, 4096, 4},
	}
	for _, c := range cases {
		x := tensor.FromFloat64(tensor.Shape{c.m, c.k}, qmRand(c.m*c.k, 1))
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k}) // GPU path is f32
		for i := range c.m * c.k {
			idx := tensor.Unravel(i, x.Shape())
			xf.SetF64(x.AtF64(idx...), idx...)
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q8_0)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q8_0, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ8_0(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := 1e-6*math.Sqrt(float64(c.k)), 1e-4
		var maxRel float64
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if rel := math.Abs(g-r) / math.Max(1, math.Abs(r)); rel > maxRel {
					maxRel = rel
				}
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: metal %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
		t.Logf("Q8_0 K=%d: maxRel=%.2e (tol %.2e)", c.k, maxRel, rtol)
		// Non-vacuousness (§T624): a real quantized-matmul bug yields ≥1e-2 rel
		// deviation; confirm crossTol at this K still catches it.
		r := ref.AtF64(0, 0)
		if bug := 1e-2 * math.Max(1, math.Abs(r)); bug <= rtol*math.Max(1, math.Abs(r))+atol {
			t.Fatalf("K=%d: crossTol %g too loose — a 1e-2 rel bug (Δ=%g) would slip through", c.k, rtol, bug)
		}
	}
}

// §V3/§V11 (§T138): the Metal Q4_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 144-byte super-blocks
// in-kernel (asymmetric affine, 6-bit scale/min via get_scale_min_k4); only the accumulation
// precision (f32 vs f64) differs. K is a multiple of 256 (the Q4_K super-block).
func TestMetalQMatMulQ4KCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 256, 8}, {4, 256, 5}, {8, 512, 12}, {2, 512, 7}, {16, 256, 3},
		// Model-scale K (§T624): multiples of the 256-element Q4_K super-block.
		// As in Q8_0, quantization cancels (ref dequantizes the SAME bytes) → pure
		// f32-vs-f64 accumulation, and the super-block-wise kernel accumulation keeps
		// the error far below raw-GEMM √K growth: measured maxRel ≤2.2e-6 @K4096,
		// ~30× under the (unchanged) tight 1e-6·√k tol. No recalibration needed.
		{1, 1024, 8}, {4, 2048, 6}, {1, 4096, 4},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(xr[i])
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q4_K)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q4_K, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ4_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := 1e-6*math.Sqrt(float64(c.k)), 1e-4
		var maxRel float64
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if rel := math.Abs(g-r) / math.Max(1, math.Abs(r)); rel > maxRel {
					maxRel = rel
				}
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: metal %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
		t.Logf("Q4_K K=%d: maxRel=%.2e (tol %.2e)", c.k, maxRel, rtol)
		// Non-vacuousness (§T624): a real bug yields ≥1e-2 rel deviation.
		r := ref.AtF64(0, 0)
		if bug := 1e-2 * math.Max(1, math.Abs(r)); bug <= rtol*math.Max(1, math.Abs(r))+atol {
			t.Fatalf("K=%d: crossTol %g too loose — a 1e-2 rel bug (Δ=%g) would slip through", c.k, rtol, bug)
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestMetalQMatMulQ4KValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := metal.QMatMulQ4_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 256})
	if _, err := metal.QMatMulQ4_K(x2, make([]byte, 10), 2, 256); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}

// §V3/§V11 (§T140): the Metal Q6_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 210-byte super-blocks
// in-kernel (symmetric, signed int8 sub-scale, 6-bit quant from ql+qh); only the accumulation
// precision differs. K is a multiple of 256.
func TestMetalQMatMulQ6KCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 256, 8}, {4, 256, 5}, {8, 512, 12}, {2, 512, 7}, {16, 256, 3},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(xr[i])
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q6_K)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q6_K, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ6_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := 1e-6*math.Sqrt(float64(c.k)), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: metal %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestMetalQMatMulQ6KValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := metal.QMatMulQ6_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 256})
	if _, err := metal.QMatMulQ6_K(x2, make([]byte, 10), 2, 256); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}

// §V3/§V11 (§T143): the Metal Q5_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 176-byte super-blocks
// in-kernel (Q4_K affine + the qh 5th-bit plane); only the accumulation precision differs.
func TestMetalQMatMulQ5KCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 256, 8}, {4, 256, 5}, {8, 512, 12}, {2, 512, 7}, {16, 256, 3},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(xr[i])
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q5_K)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q5_K, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ5_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := 1e-6*math.Sqrt(float64(c.k)), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: metal %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestMetalQMatMulQ5KValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := metal.QMatMulQ5_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 256})
	if _, err := metal.QMatMulQ5_K(x2, make([]byte, 10), 2, 256); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}

// §V3/§V11 (§T145): the Metal Q2_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 84-byte super-blocks
// in-kernel (asymmetric affine, plain 4-bit nibble scale/min, 2-bit quants); only the
// accumulation precision differs.
func TestMetalQMatMulQ2KCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 256, 8}, {4, 256, 5}, {8, 512, 12}, {2, 512, 7}, {16, 256, 3},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(xr[i])
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q2_K)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q2_K, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ2_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := 1e-6*math.Sqrt(float64(c.k)), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: metal %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestMetalQMatMulQ2KValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := metal.QMatMulQ2_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
}

// §V3/§V11 (§T147): the Metal Q3_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 110-byte super-blocks
// in-kernel, including the aux/kmask 6-bit-scale splice, the signed scale and the inverted
// hmask; only the accumulation precision differs. This anchors the most intricate quant kernel.
func TestMetalQMatMulQ3KCrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 256, 8}, {4, 256, 5}, {8, 512, 12}, {2, 512, 7}, {16, 256, 3},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(xr[i])
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q3_K)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q3_K, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ3_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := 1e-6*math.Sqrt(float64(c.k)), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: metal %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestMetalQMatMulQ3KValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := metal.QMatMulQ3_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestMetalQMatMulQ8Validation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 40})
	if _, err := metal.QMatMulQ8_0(x, make([]byte, 100), 2, 40); err == nil {
		t.Error("K=40 (not a multiple of 32) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 32})
	if _, err := metal.QMatMulQ8_0(x2, make([]byte, 10), 2, 32); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}

// TestMetalQMatMulQ4_0CrossReference V-CROSSes the GPU Q4_0 matmul against the f64 CPU reference
// gguf.QMatMul over decode (M=1) and prefill (M=4..16) shapes: both dequantize the SAME Q4_0 bytes,
// so they must agree within an f32-vs-f64 accumulation crossTol (§V3/§V11). Q4_0 is the classic,
// very common 4-bit GGUF format (§T166).
func TestMetalQMatMulQ4_0CrossReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 32, 8}, {1, 256, 16}, {4, 64, 5}, {8, 128, 12}, {2, 512, 7}, {16, 256, 3},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k}) // GPU path is f32
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.SetF64(xr[i], tensor.Unravel(i, xf.Shape())...)
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q4_0)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q4_0, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metal.QMatMulQ4_0(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := 1e-6*math.Sqrt(float64(c.k)), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: metal %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// TestMetalQMatMulQ4_0Dispatch checks the backend.QuantMatMuler dispatch routes ggml type 2 (Q4_0)
// to the GPU kernel and matches the reference.
func TestMetalQMatMulQ4_0Dispatch(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const m, k, n = 3, 128, 6
	xf := tensor.New(tensor.F32, tensor.Shape{m, k})
	xr := qmRand(m*k, 5)
	for i := range m * k {
		xf.SetF64(xr[i], tensor.Unravel(i, xf.Shape())...)
	}
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 6))
	wq, _ := gguf.Quantize(w, gguf.Q4_0)
	qm, ok := any(metal.Backend{}).(interface {
		QMatMul(*tensor.Tensor, []byte, uint32, int, int) (*tensor.Tensor, error)
	})
	if !ok {
		t.Fatal("metal.Backend must implement QuantMatMuler")
	}
	got, err := qm.QMatMul(xf, wq, 2 /* ggml Q4_0 */, n, k)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	ref, _ := gguf.QMatMul(xf, wq, gguf.Q4_0, n, k)
	rtol, atol := 1e-6*math.Sqrt(float64(k)), 1e-4
	for mi := range m {
		for ni := range n {
			if math.Abs(got.AtF64(mi, ni)-ref.AtF64(mi, ni)) > rtol*math.Max(1, math.Abs(ref.AtF64(mi, ni)))+atol {
				t.Errorf("[%d,%d]: %g vs %g", mi, ni, got.AtF64(mi, ni), ref.AtF64(mi, ni))
			}
		}
	}
}

func TestMetalQMatMulQ4_0Validation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 40})
	if _, err := metal.QMatMulQ4_0(x, make([]byte, 100), 2, 40); err == nil {
		t.Error("K=40 (not a multiple of 32) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 32})
	if _, err := metal.QMatMulQ4_0(x2, make([]byte, 10), 2, 32); err == nil {
		t.Error("wrong weight length must be rejected") // Q4_0 needs 2·(32/32)·18=36 bytes
	}
}

// BenchmarkMetalQMatMulQ8 vs the CPU reference, informing the device-residency discussion
// (§B): the per-call weight upload makes the GPU path memory-bound for small M.
func BenchmarkMetalQMatMulQ8(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const m, k, n = 8, 4096, 4096
	x := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i := range m * k {
		x.Storage().F32()[i] = float32(qmRand(1, i)[0])
	}
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 3))
	wq, _ := gguf.Quantize(w, gguf.Q8_0)
	b.Run("gpu", func(b *testing.B) {
		for range b.N {
			if _, err := metal.QMatMulQ8_0(x, wq, n, k); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("host", func(b *testing.B) {
		for range b.N {
			if _, err := gguf.QMatMul(x, wq, gguf.Q8_0, n, k); err != nil {
				b.Fatal(err)
			}
		}
	})
}
