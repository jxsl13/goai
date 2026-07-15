//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// ropeBandRef rotates, in place on a host copy, the `heads`×hd band starting at
// column `off` inside rows of `stride` floats — the exact float64 arithmetic the
// rope_band kernel does (HF rotate_half), so the GPU result must match to f32.
func ropeBandRef(x []float32, seq, stride, off, heads, hd, posOffset int, posDiv, base float64) {
	half := hd / 2
	inv := make([]float64, half)
	for i := range inv {
		inv[i] = 1.0 / math.Pow(base, float64(2*i)/float64(hd))
	}
	for p := 0; p < seq; p++ {
		pos := float64(posOffset+p) / posDiv
		for h := 0; h < heads; h++ {
			b := p*stride + off + h*hd
			for i := 0; i < half; i++ {
				ang := pos * inv[i]
				c, s := math.Cos(ang), math.Sin(ang)
				qi, qih := float64(x[b+i]), float64(x[b+i+half])
				x[b+i] = float32(qi*c - qih*s)
				x[b+i+half] = float32(qih*c + qi*s)
			}
		}
	}
}

// RoPEPairBand (fused-QKV strided RoPE — the last llamagpu CUDA-adapter recorder
// gap) must rotate the q and k bands of one [seq,stride] buffer to match the host
// reference, AND leave the v band (past the k band) untouched.
func TestCUDARoPEBand(t *testing.T) {
	skipNoGPU(t)

	const (
		seq       = 3
		hd        = 4
		headsQ    = 2
		headsK    = 1
		offQ      = 0
		offK      = headsQ * hd      // 8
		vOff      = offK + headsK*hd // 12
		stride    = 20               // q(8) + k(4) + v(8)
		posOffset = 5
		base      = 10000.0
		posDiv    = 1.0
	)

	host := bench.RandF32(tensor.Shape{seq, stride}, 7)
	hf := host.Contiguous().Storage().F32()
	want := append([]float32{}, hf...)
	// Reference: rotate q band then k band; v band left alone.
	ropeBandRef(want, seq, stride, offQ, headsQ, hd, posOffset, posDiv, base)
	ropeBandRef(want, seq, stride, offK, headsK, hd, posOffset, posDiv, base)

	inv, err := cuda.BuildRoPEInv(hd, base)
	must(t, err)
	defer inv.Free()
	d, err := cuda.UploadF32(host)
	must(t, err)
	defer d.Free()

	must(t, d.RoPEPairBand(inv, stride, headsQ, offQ, headsK, offK, hd, posOffset, posDiv))
	got, err := d.ToHost()
	must(t, err)

	const tol = 1e-4
	for p := 0; p < seq; p++ {
		for c := 0; c < stride; c++ {
			g := got.AtF64(p, c)
			w := float64(want[p*stride+c])
			if math.Abs(g-w) > tol+tol*math.Abs(w) {
				t.Fatalf("RoPEPairBand[%d,%d]=%v want %v (Δ%g)", p, c, g, w, math.Abs(g-w))
			}
		}
	}
	// v band must be byte-identical to the input (RoPE never touched it).
	for p := 0; p < seq; p++ {
		for c := vOff; c < stride; c++ {
			if got.AtF64(p, c) != float64(hf[p*stride+c]) {
				t.Fatalf("RoPEPairBand corrupted v band at [%d,%d]: %v != %v", p, c, got.AtF64(p, c), hf[p*stride+c])
			}
		}
	}
	t.Log("RoPEPairBand == host reference; v band untouched")
}
