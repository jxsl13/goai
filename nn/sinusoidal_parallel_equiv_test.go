package nn_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The sinusoidal PE builders fan their per-position row fill over GOMAXPROCS. Each row is a
// deterministic Sincos of pos writing disjoint columns, so the table must be BYTE-FOR-BYTE
// identical to the single-worker build. Locked by building at GOMAXPROCS=1 and N for both the
// interleaved and concat layouts across dtypes.
func TestSinusoidalPEParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	const seqLen, dModel = 300, 128
	build := func(concat bool, dt tensor.Dtype) *tensor.Tensor {
		var pe *tensor.Tensor
		var err error
		if concat {
			pe, err = nn.SinusoidalPositionalEncodingConcat(seqLen, dModel, 10000, dt)
		} else {
			pe, err = nn.SinusoidalPositionalEncoding(seqLen, dModel, 10000, dt)
		}
		if err != nil {
			t.Fatal(err)
		}
		return pe
	}

	for _, concat := range []bool{false, true} {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			runtime.GOMAXPROCS(1)
			serial := build(concat, dt)
			runtime.GOMAXPROCS(prev)
			par := build(concat, dt)
			for r := 0; r < seqLen; r++ {
				for c := 0; c < dModel; c++ {
					if s, p := serial.AtF64(r, c), par.AtF64(r, c); s != p {
						t.Fatalf("concat=%v dt=%v [%d,%d]: serial %v != parallel %v", concat, dt, r, c, s, p)
					}
				}
			}
		}
	}
}
