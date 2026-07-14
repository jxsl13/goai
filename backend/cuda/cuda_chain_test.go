//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// An on-device matmul chain x·W1·W2 (intermediate never leaves the GPU) must
// equal the per-call cuda chain EXACTLY (same cuBLAS Sgemm sequence) and the
// f64 reference within tolerance. Guards the §V14 Phase-2 device path.
func TestCUDADeviceChainMatchesRefAndCuda(t *testing.T) {
	skipNoGPU(t)
	cb, _ := backend.Get(backend.CUDA)
	ref, _ := backend.Get(backend.Ref)

	shapes := []struct{ m, k0, n1, n2 int }{
		{2, 3, 4, 5}, {8, 64, 32, 16}, {33, 65, 17, 49}, {64, 128, 256, 128},
	}
	for _, s := range shapes {
		x := bench.RandF32(tensor.Shape{s.m, s.k0}, 1)
		w1 := bench.RandF32(tensor.Shape{s.k0, s.n1}, 2)
		w2 := bench.RandF32(tensor.Shape{s.n1, s.n2}, 3)

		// on-device chain: 1 upload + 2 device matmuls + 1 download
		rb1, err := cuda.NewResidentB(w1)
		if err != nil {
			t.Fatalf("%v: %v", s, err)
		}
		rb2, _ := cuda.NewResidentB(w2)
		dx, err := cuda.UploadF32(x)
		if err != nil {
			t.Fatal(err)
		}
		dh, err := rb1.MatMulDevice(dx)
		if err != nil {
			t.Fatal(err)
		}
		dy, err := rb2.MatMulDevice(dh)
		if err != nil {
			t.Fatal(err)
		}
		got, err := dy.ToHost()
		if err != nil {
			t.Fatal(err)
		}

		// per-call cuda chain (same Sgemm ops) → must be bit-identical
		ch, _ := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpMatMul, []*tensor.Tensor{x, w1}, nil)
		cy, _ := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpMatMul, []*tensor.Tensor{ch[0], w2}, nil)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			if g, c := got.AtF64(idx...), cy[0].AtF64(idx...); g != c {
				t.Fatalf("%v [%d]: device-chain %v != per-call cuda %v", s, i, g, c)
			}
		}
		// ref chain within tolerance (two f32 matmuls vs f64 truth)
		rh, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{x, w1}, nil)
		ry, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{rh[0], w2}, nil)
		rtol := crossTol(s.k0) + crossTol(s.n1)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), ry[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
				t.Fatalf("%v [%d]: device-chain %v vs ref %v (rtol %g)", s, i, g, r, rtol)
			}
		}
		dh.Free()
		dy.Free()
		dx.Free()
		rb1.Free()
		rb2.Free()
	}
}

func TestCUDADeviceUseAfterFree(t *testing.T) {
	skipNoGPU(t)
	dx, err := cuda.UploadF32(bench.RandF32(tensor.Shape{4, 4}, 1))
	if err != nil {
		t.Fatal(err)
	}
	dx.Free()
	if _, err := dx.ToHost(); err == nil {
		t.Fatal("expected error on ToHost after Free")
	}
}

// The chain win: an MLP-shaped x·W1·W2 with small activation, large weights.
// Device chain = 1 H2D (x) + 1 D2H (y); the intermediate h stays on-GPU and the
// weights are resident. Per-call = both operands uploaded + result downloaded
// per matmul (weights re-uploaded, h bounced host↔device).
func benchChain(b *testing.B, device bool, m, d int) {
	be, ok := backend.Get(backend.CUDA)
	if !ok {
		b.Skip("cuda not available")
	}
	x := bench.RandF32(tensor.Shape{m, d}, 1)
	w1 := bench.RandF32(tensor.Shape{d, d}, 2)
	w2 := bench.RandF32(tensor.Shape{d, d}, 3)
	if device {
		rb1, _ := cuda.NewResidentB(w1)
		rb2, _ := cuda.NewResidentB(w2)
		defer rb1.Free()
		defer rb2.Free()
		b.ResetTimer()
		for range b.N {
			dx, _ := cuda.UploadF32(x)
			dh, err := rb1.MatMulDevice(dx)
			if err != nil {
				b.Fatal(err)
			}
			dy, _ := rb2.MatMulDevice(dh)
			if _, err := dy.ToHost(); err != nil {
				b.Fatal(err)
			}
			dx.Free()
			dh.Free()
			dy.Free()
		}
		return
	}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		h, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, w1}, nil)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{h[0], w2}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDeviceChain_M8_D4096(b *testing.B)  { benchChain(b, true, 8, 4096) }
func BenchmarkPerCallChain_M8_D4096(b *testing.B) { benchChain(b, false, 8, 4096) }
