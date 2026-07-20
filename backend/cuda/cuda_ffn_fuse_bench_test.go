//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// Gate+Up GEMM fusion: the FFN runs gate and up as TWO separate f16 GEMMs [dim->hidden]. Fusing them
// into ONE [dim->2*hidden] GEMM saves a launch and may tile better. Context-INDEPENDENT (helps every
// decode step incl. the short-context regime we already win). This measures the GEMM part only (the
// SwiGLU is unchanged/cheap); a win here justifies adding a strided SwiGLU to consume the fused layout.
func benchFFNGateUp(b *testing.B, batch, dim, hidden int, fused bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	a := cuda.AllocU16(batch * dim)
	defer cuda.FreeDev(a)
	if fused {
		w := cuda.AllocU16(dim * 2 * hidden)
		c := cuda.AllocU16(batch * 2 * hidden)
		defer cuda.FreeDev(w)
		defer cuda.FreeDev(c)
		cuda.GemmF16Pure(a, w, c, batch, dim, 2*hidden) // warm
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			cuda.GemmF16Pure(a, w, c, batch, dim, 2*hidden)
		}
		cuda.GraphSync()
	} else {
		wg := cuda.AllocU16(dim * hidden)
		wu := cuda.AllocU16(dim * hidden)
		cg := cuda.AllocU16(batch * hidden)
		cu := cuda.AllocU16(batch * hidden)
		defer cuda.FreeDev(wg)
		defer cuda.FreeDev(wu)
		defer cuda.FreeDev(cg)
		defer cuda.FreeDev(cu)
		cuda.GemmF16Pure(a, wg, cg, batch, dim, hidden) // warm
		cuda.GemmF16Pure(a, wu, cu, batch, dim, hidden)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			cuda.GemmF16Pure(a, wg, cg, batch, dim, hidden)
			cuda.GemmF16Pure(a, wu, cu, batch, dim, hidden)
		}
		cuda.GraphSync()
	}
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "ffn/s")
}

func BenchmarkFFNGateUpSeparate_b512(b *testing.B) { benchFFNGateUp(b, 512, 2048, 5632, false) }
func BenchmarkFFNGateUpFused_b512(b *testing.B)    { benchFFNGateUp(b, 512, 2048, 5632, true) }
