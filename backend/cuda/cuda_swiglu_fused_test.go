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

// Fused SwiGLU (SiLU(gate)⊙up in one pass) must match SiLU-then-Mul / the
// reference within the f32 tolerance.
func TestCUDADeviceSwiGLUMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ r, c int }{{1, 1}, {4, 8}, {33, 17}, {8, 5504}} {
		gate := bench.RandF32(tensor.Shape{s.r, s.c}, 1)
		up := bench.RandF32(tensor.Shape{s.r, s.c}, 2)
		dg, _ := cuda.UploadF32(gate)
		du, _ := cuda.UploadF32(up)
		if err := dg.SwiGLU(du); err != nil {
			t.Fatal(err)
		}
		got, _ := dg.ToHost()
		sm := func() *tensor.Tensor {
			o, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSiLU, []*tensor.Tensor{gate}, nil)
			return o[0]
		}()
		wantT, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMul, []*tensor.Tensor{sm, up}, nil)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, w := got.AtF64(idx...), wantT[0].AtF64(idx...)
			if math.Abs(g-w) > 1e-5*math.Max(1, math.Abs(w)) {
				t.Fatalf("%v [%d]: swiglu %v vs ref %v", s, i, g, w)
			}
		}
		dg.Free()
		du.Free()
	}
}

// Fused vs unfused: one kernel/one memory pass vs SiLU then Mul.
func benchGatedElem(b *testing.B, fused bool, r, c int) {
	if _, ok := backend.Get(backend.CUDA); !ok {
		b.Skip("cuda not available")
	}
	gate := bench.RandF32(tensor.Shape{r, c}, 1)
	up := bench.RandF32(tensor.Shape{r, c}, 2)
	du, _ := cuda.UploadF32(up)
	defer du.Free()
	b.ResetTimer()
	for range b.N {
		dg, _ := cuda.UploadF32(gate)
		if fused {
			if err := dg.SwiGLU(du); err != nil {
				b.Fatal(err)
			}
		} else {
			dg.SiLU()
			dg.Mul(du)
		}
		if _, err := dg.ToHost(); err != nil {
			b.Fatal(err)
		}
		dg.Free()
	}
}

func BenchmarkSwiGLUFused_8x5504(b *testing.B)   { benchGatedElem(b, true, 8, 5504) }
func BenchmarkSwiGLUUnfused_8x5504(b *testing.B) { benchGatedElem(b, false, 8, 5504) }
