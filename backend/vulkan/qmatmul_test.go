//go:build vulkan && cgo

package vulkan_test

import (
	"errors"
	"math"
	"runtime"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// §T157: a Vulkan resident weight dropped without Close is freed by a runtime cleanup (the cgo
// vk_qweight_free runs from the cleanup goroutine) without crashing; a fresh resident matmul still
// works afterward. Best-effort: forces GC.
func TestVulkanResidentAutoFree(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	var rb backend.ResidentQuantMatMuler = vulkan.Backend{}
	const k, n = 256, 4
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 9))
	wq, _ := gguf.Quantize(w, gguf.Q4_K)
	for range 8 {
		rw, err := rb.UploadQuant(wq, 12, n, k)
		if err != nil {
			t.Fatal(err)
		}
		_ = rw // dropped without Close
	}
	for range 3 {
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, k})
	for i := range k {
		x.Storage().F32()[i] = float32(qmRand(k, 1)[i])
	}
	rw, err := rb.UploadQuant(wq, 12, n, k)
	if err != nil {
		t.Fatal(err)
	}
	defer rw.Close()
	got, err := rw.QMatMul(x)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := gguf.QMatMul(x, wq, gguf.Q4_K, n, k)
	for ni := range n {
		if g, r := got.AtF64(0, ni), ref.AtF64(0, ni); math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
			t.Errorf("[0,%d] after auto-free: %g vs %g", ni, g, r)
		}
	}
}

// §T142: vulkan.Backend implements backend.QuantMatMuler — the {8,12,14} dispatcher routes to
// the in-kernel matmul (== gguf.QMatMul within crossTol) and returns ErrQuantUnsupported for a
// quant type without a GPU kernel (so nn.QuantLinear falls back to CPU).
func TestVulkanBackendQuantMatMuler(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	var qm backend.QuantMatMuler = vulkan.Backend{}
	const m, k, n = 2, 256, 5
	xf := tensor.New(tensor.F32, tensor.Shape{m, k})
	for i := range m * k {
		xf.Storage().F32()[i] = float32(qmRand(m*k, 1)[i])
	}
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 2))
	wq, _ := gguf.Quantize(w, gguf.Q4_K)
	got, err := qm.QMatMul(xf, wq, 12, n, k) // 12 = ggml Q4_K
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := gguf.QMatMul(xf, wq, gguf.Q4_K, n, k)
	rtol := crossTol(k)
	for mi := range m {
		for ni := range n {
			if g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni); math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+1e-4 {
				t.Errorf("[%d,%d] %g vs %g", mi, ni, g, r)
			}
		}
	}
	if _, err := qm.QMatMul(xf, wq, 3, n, k); !errors.Is(err, backend.ErrQuantUnsupported) { // 3 = Q4_1, no kernel
		t.Errorf("unsupported type: got %v, want ErrQuantUnsupported", err)
	}
}

// §V3/§V11 (§T146): the Vulkan Q2_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 84-byte super-blocks
// in-kernel (asymmetric affine, plain 4-bit nibble scale/min, 2-bit quants, bytes from a uint
// word array); only the accumulation precision differs. K is a multiple of 256, on MoltenVK.
func TestVulkanQMatMulQ2KCrossReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
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
		got, err := vulkan.QMatMulQ2_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := crossTol(c.k), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: vulkan %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V3/§V11 (§T144): the Vulkan Q5_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 176-byte super-blocks
// in-kernel (Q4_K affine + the qh 5th-bit plane, bytes from a uint word array); only the
// accumulation precision differs. K is a multiple of 256, on MoltenVK.
func TestVulkanQMatMulQ5KCrossReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
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
		got, err := vulkan.QMatMulQ5_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := crossTol(c.k), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: vulkan %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestVulkanQMatMulQ5KValidation(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := vulkan.QMatMulQ5_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
}

func qmRand(n, seed int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(float64(i*13+seed)*0.21+0.3) * 2
	}
	return x
}

// §V3/§V11 (§T137): the Vulkan Q8_0 quantized matmul equals the reference gguf.QMatMul (the
// f64 CPU truth over the identical dequantized weights) within crossTol — the GPU kernel
// dequantizes the same f16-scale·int8 blocks in-kernel (bytes read from a uint word array,
// f16 via unpackHalf2x16); only the accumulation precision (f32 vs f64) differs. Covers
// decode (M=1) and prefill (M>1) over several K, on MoltenVK.
func TestVulkanQMatMulQ8CrossReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 32, 8}, {1, 256, 16}, {4, 64, 5}, {8, 128, 12}, {2, 512, 7}, {16, 256, 3},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(xr[i])
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
		got, err := vulkan.QMatMulQ8_0(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := crossTol(c.k), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: vulkan %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V3/§V11 (§T139): the Vulkan Q4_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 144-byte super-blocks
// in-kernel (asymmetric affine, 6-bit scale/min via get_scale_min_k4, bytes from a uint word
// array); only the accumulation precision differs. K is a multiple of 256, on MoltenVK.
func TestVulkanQMatMulQ4KCrossReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
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
		wq, err := gguf.Quantize(w, gguf.Q4_K)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := gguf.QMatMul(xf, wq, gguf.Q4_K, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		got, err := vulkan.QMatMulQ4_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := crossTol(c.k), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: vulkan %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestVulkanQMatMulQ4KValidation(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := vulkan.QMatMulQ4_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 256})
	if _, err := vulkan.QMatMulQ4_K(x2, make([]byte, 10), 2, 256); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}

// §V3/§V11 (§T167, the Vulkan twin of Metal §T166): the Vulkan Q4_0 quantized matmul equals the
// reference gguf.QMatMul (the f64 CPU truth) within crossTol — the GPU kernel dequantizes the same
// 18-byte Q4_0 blocks in-kernel (f16 scale + 32 nibbles offset −8, low nibble → element i, high →
// i+16, bytes from a uint word array); only the accumulation precision differs. K is a multiple of
// 32, on MoltenVK. Q4_0 is the classic, very common 4-bit GGUF format — closing the never-metal-only
// gap for it.
func TestVulkanQMatMulQ4_0CrossReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 32, 8}, {1, 256, 16}, {4, 64, 5}, {8, 128, 12}, {2, 512, 7}, {16, 256, 3},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		xr := qmRand(c.m*c.k, 1)
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(xr[i])
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
		got, err := vulkan.QMatMulQ4_0(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := crossTol(c.k), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: vulkan %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V3/§V11: the backend.QuantMatMuler dispatch routes ggml type 2 (Q4_0) to the Vulkan kernel and
// matches the reference.
func TestVulkanQMatMulQ4_0Dispatch(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	const m, k, n = 3, 128, 6
	xf := tensor.New(tensor.F32, tensor.Shape{m, k})
	xr := qmRand(m*k, 5)
	for i := range m * k {
		xf.Storage().F32()[i] = float32(xr[i])
	}
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 6))
	wq, _ := gguf.Quantize(w, gguf.Q4_0)
	got, err := vulkan.Backend{}.QMatMul(xf, wq, 2 /* ggml Q4_0 */, n, k)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	ref, _ := gguf.QMatMul(xf, wq, gguf.Q4_0, n, k)
	rtol, atol := crossTol(k), 1e-4
	for mi := range m {
		for ni := range n {
			if math.Abs(got.AtF64(mi, ni)-ref.AtF64(mi, ni)) > rtol*math.Max(1, math.Abs(ref.AtF64(mi, ni)))+atol {
				t.Errorf("[%d,%d]: %g vs %g", mi, ni, got.AtF64(mi, ni), ref.AtF64(mi, ni))
			}
		}
	}
}

func TestVulkanQMatMulQ4_0Validation(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 40})
	if _, err := vulkan.QMatMulQ4_0(x, make([]byte, 100), 2, 40); err == nil {
		t.Error("K=40 (not a multiple of 32) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 32})
	if _, err := vulkan.QMatMulQ4_0(x2, make([]byte, 10), 2, 32); err == nil {
		t.Error("wrong weight length must be rejected") // Q4_0 needs 2·(32/32)·18=36 bytes
	}
}

// §V3/§V11 (§T141): the Vulkan Q6_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 210-byte super-blocks
// in-kernel (symmetric, signed int8 sub-scale, 6-bit quant from ql+qh, bytes from a uint word
// array); only the accumulation precision differs. K is a multiple of 256, on MoltenVK.
func TestVulkanQMatMulQ6KCrossReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
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
		got, err := vulkan.QMatMulQ6_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := crossTol(c.k), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: vulkan %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestVulkanQMatMulQ6KValidation(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 128})
	if _, err := vulkan.QMatMulQ6_K(x, make([]byte, 100), 2, 128); err == nil {
		t.Error("K=128 (not a multiple of 256) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 256})
	if _, err := vulkan.QMatMulQ6_K(x2, make([]byte, 10), 2, 256); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}

// §V3/§V11 (§T148): the Vulkan Q3_K quantized matmul equals the reference gguf.QMatMul (the f64
// CPU truth) within crossTol — the GPU kernel dequantizes the same 110-byte super-blocks
// in-kernel (aux/kmask 6-bit-scale splice, signed scale, inverted hmask, bytes from a uint word
// array); only the accumulation precision differs. K is a multiple of 256, on MoltenVK. This
// completes the GPU quant family on both backends.
func TestVulkanQMatMulQ3KCrossReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
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
		got, err := vulkan.QMatMulQ3_K(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		if !got.Shape().Equal(tensor.Shape{c.m, c.n}) {
			t.Fatalf("case %+v: shape %v", c, got.Shape())
		}
		rtol, atol := crossTol(c.k), 1e-4
		for mi := range c.m {
			for ni := range c.n {
				g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("case %+v [%d,%d]: vulkan %g vs ref %g (Δ %g)", c, mi, ni, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V3/§V11 (§T156): vulkan.Backend implements backend.ResidentQuantMatMuler — a device-resident
// weight (uploaded once) gives the SAME result as the per-call QMatMul for every supported k-quant/
// Q8_0 type, and an unsupported type (Q4_0) returns ErrQuantUnsupported. On MoltenVK.
func TestVulkanResidentQuantMatMuler(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	var rb backend.ResidentQuantMatMuler = vulkan.Backend{}
	const k, n = 256, 5
	x := tensor.New(tensor.F32, tensor.Shape{2, k})
	for i := range 2 * k {
		x.Storage().F32()[i] = float32(qmRand(2*k, 1)[i])
	}
	types := []struct {
		qt   gguf.QuantType
		code uint32
	}{{gguf.Q8_0, 8}, {gguf.Q4_0, 2}, {gguf.Q4_K, 12}, {gguf.Q5_K, 13}, {gguf.Q6_K, 14}, {gguf.Q2_K, 10}, {gguf.Q3_K, 11}}
	for _, tc := range types {
		w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 2))
		wq, _ := gguf.Quantize(w, tc.qt)
		rw, err := rb.UploadQuant(wq, tc.code, n, k)
		if err != nil {
			t.Fatalf("%v upload: %v", tc.qt, err)
		}
		got, err := rw.QMatMul(x)
		if err != nil {
			t.Fatalf("%v: %v", tc.qt, err)
		}
		ref, _ := gguf.QMatMul(x, wq, tc.qt, n, k)
		for mi := range 2 {
			for ni := range n {
				if g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni); math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
					t.Errorf("%v [%d,%d] %g vs %g", tc.qt, mi, ni, g, r)
				}
			}
		}
		_ = rw.Close()
	}
	if _, err := rb.UploadQuant(make([]byte, 100), 3, n, k); !errors.Is(err, backend.ErrQuantUnsupported) { // 3 = Q4_1, no resident kernel
		t.Errorf("Q4_1 resident: got %v, want ErrQuantUnsupported", err)
	}
}

// §V7-adjacent: input validation rejects a bad K / weight length rather than reading OOB.
func TestVulkanQMatMulQ8Validation(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, 40})
	if _, err := vulkan.QMatMulQ8_0(x, make([]byte, 100), 2, 40); err == nil {
		t.Error("K=40 (not a multiple of 32) must be rejected")
	}
	x2 := tensor.New(tensor.F32, tensor.Shape{1, 32})
	if _, err := vulkan.QMatMulQ8_0(x2, make([]byte, 10), 2, 32); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}
