package cpu_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// TestConvChunkedScratchExact covers the conv shapes whose im2col scratch is walked in
// SEVERAL chunks per worker band, which is the case the whole-tensor buffer never produced
// and where reusing one window across chunks can silently carry state forward.
//
// It found a real defect. gemmF64Band ACCUMULATES into its destination — the whole-tensor
// product buffer was pooled zeroed and each row's slot written once, so nothing in the old
// code depended on noticing that. A reused product window does: without clearing it between
// chunks every chunk after the first added its result on top of the previous one. The
// padding taps have the same shape of hazard in the column window, since im2col skips them
// and relies on the buffer already being zero.
//
// The shapes are chosen so the chunking is FORCED, not incidental. A wide kernel makes the
// per-row column count large, which drives the rows-per-chunk down, while the output stays
// large enough that a band spans several of them. Both a padded and an unpadded case are
// present: only the padded one can expose stale column taps, and only a case with more than
// one output channel gives the product window room to accumulate visibly.
//
// The oracle is the reference backend, an independent implementation. Comparing the CPU
// backend against itself under a different GOMAXPROCS would not do: both arms chunk.
func TestConvChunkedScratchExact(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("chunking per band needs more than one worker to be interesting")
	}

	cases := []struct {
		name                        string
		n, c, h, w, f, kh, kw, s, p int
		bias                        bool
	}{
		// k = c*kh*kw = 1024 columns per row, so a chunk holds only a few dozen rows while
		// the output has 1225 of them: several chunks per band on any core count.
		{"wide-kernel-padded", 1, 16, 40, 40, 4, 8, 8, 1, 1, true},
		{"wide-kernel-unpadded", 1, 16, 40, 40, 4, 8, 8, 1, 0, false},
		// Strided and padded, still wide enough to chunk.
		{"wide-kernel-strided", 1, 8, 48, 48, 3, 8, 8, 2, 3, true},
	}
	for _, tc := range cases {
		for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
			var x, w, b *tensor.Tensor
			if dtype == tensor.F64 {
				x = bench.RandF64(tensor.Shape{tc.n, tc.c, tc.h, tc.w}, 11)
				w = bench.RandF64(tensor.Shape{tc.f, tc.c, tc.kh, tc.kw}, 22)
				b = bench.RandF64(tensor.Shape{tc.f}, 33)
			} else {
				x = bench.RandF32(tensor.Shape{tc.n, tc.c, tc.h, tc.w}, 11)
				w = bench.RandF32(tensor.Shape{tc.f, tc.c, tc.kh, tc.kw}, 22)
				b = bench.RandF32(tensor.Shape{tc.f}, 33)
			}
			ins := []*tensor.Tensor{x, w}
			if tc.bias {
				ins = append(ins, b)
			}
			attrs := backend.ConvAttrs{Stride: tc.s, Pad: tc.p}
			gc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatalf("%s/%v cpu: %v", tc.name, dtype, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatalf("%s/%v ref: %v", tc.name, dtype, err)
			}
			assertMatMul(t, gc[0], gr[0], tc.name+"/"+dtype.String())
		}
	}
}
