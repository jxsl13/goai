package nn

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// gather fans its per-token neighbour search over GOMAXPROCS. Each token writes only its own
// output block and its retrieveHead is deterministic (strict total order), so the gathered
// tensors must be BYTE-FOR-BYTE identical to the single-worker result. Locked at GOMAXPROCS=1
// vs N for F64 and F32 outputs.
func TestMemGatherParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	const n, dim, headDim, tSeg, topM = 512, 128, 64, 96, 16
	m := &MemMemory{dim: dim, cap: n}
	m.keys = make([][]float64, n)
	m.vals = make([][]float64, n)
	for i := range m.keys {
		kr := make([]float64, dim)
		vr := make([]float64, dim)
		for d := range kr {
			kr[d] = math.Sin(float64(i*dim+d) * 0.0011)
			vr[d] = math.Cos(float64(i*dim+d) * 0.0017)
		}
		m.keys[i], m.vals[i] = kr, vr
	}
	qh := tensor.New(tensor.F64, tensor.Shape{tSeg, headDim})
	qs := qh.Storage().F64()
	for i := range qs {
		qs[i] = math.Sin(float64(i) * 0.006)
	}

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		runtime.GOMAXPROCS(1)
		kg1, vg1 := m.gather(dt, qh, headDim, headDim, topM) // headOff=headDim exercises a nonzero offset
		runtime.GOMAXPROCS(prev)
		kg2, vg2 := m.gather(dt, qh, headDim, headDim, topM)
		for name, pr := range map[string][2]*tensor.Tensor{"kg": {kg1, kg2}, "vg": {vg1, vg2}} {
			var a, b []float64
			if dt == tensor.F64 {
				a, b = pr[0].Storage().F64(), pr[1].Storage().F64()
			} else {
				af, bf := pr[0].Storage().F32(), pr[1].Storage().F32()
				a = make([]float64, len(af))
				b = make([]float64, len(bf))
				for i := range af {
					a[i], b[i] = float64(af[i]), float64(bf[i])
				}
			}
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("dt=%v %s[%d]: serial %v != parallel %v", dt, name, i, a[i], b[i])
				}
			}
		}
	}
}
