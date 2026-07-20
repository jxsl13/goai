//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

func h2f32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(0)
		m := mant
		for m&0x400 == 0 {
			m <<= 1
			e++
		}
		m &= 0x3ff
		return math.Float32frombits(sign | (127-15-e+1)<<23 | m<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (uint32(exp)-15+127)<<23 | mant<<13)
	}
}

func relRMS(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num / den)
}

// uploadF16 rounds f32 host data through f16 on device; returns the device u16 ptr and the host
// f32 of the rounded values (the reference input, so parity isolates the kernel's math, not rounding).
func uploadF16(t *testing.T, f []float32) (unsafe.Pointer, []float32) {
	t.Helper()
	d, _ := cuda.NewDeviceF32(1, len(f))
	d.UploadF32(f)
	u16 := cuda.AllocU16(len(f))
	if rc := cuda.CvtF32ToF16(u16, d.DevPtr(), len(f)); rc != 0 {
		t.Fatalf("cvt f32->f16 rc=%d", rc)
	}
	d.Free()
	raw := make([]uint16, len(f))
	cuda.DownloadU16(u16, raw)
	rounded := make([]float32, len(f))
	for i := range raw {
		rounded[i] = h2f32(raw[i])
	}
	return u16, rounded
}

func downloadF16(src unsafe.Pointer, n int) []float32 {
	raw := make([]uint16, n)
	cuda.DownloadU16(src, raw)
	out := make([]float32, n)
	for i := range raw {
		out[i] = h2f32(raw[i])
	}
	return out
}

func TestA1RMSNormF16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const rows, cols = 8, 2048
	rng := rand.New(rand.NewSource(1))
	xf := make([]float32, rows*cols)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64())
	}
	gt := tensor.New(tensor.F32, tensor.Shape{cols})
	gf := gt.Storage().F32()
	for i := range gf {
		gf[i] = 1.0 + float32(rng.NormFloat64())*0.1
	}
	inPtr, xr := uploadF16(t, xf)
	defer cuda.FreeDev(inPtr)
	gv, _ := cuda.NewResidentVec(gt)
	defer gv.Free()
	outU := cuda.AllocU16(rows * cols)
	defer cuda.FreeDev(outU)
	if rc := cuda.RMSNormF16(inPtr, outU, gv.VecPtr(), rows, cols, 1e-5); rc != 0 {
		t.Fatalf("rmsnorm_f16 rc=%d", rc)
	}
	got := downloadF16(outU, rows*cols)
	ref := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		var ss float64
		for c := 0; c < cols; c++ {
			v := float64(xr[r*cols+c])
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(cols)+1e-5))
		for c := 0; c < cols; c++ {
			ref[r*cols+c] = xr[r*cols+c] * inv * gf[c]
		}
	}
	rms := relRMS(ref, got)
	t.Logf("rmsnorm f16 rel-RMS %.3e", rms)
	if rms > 3e-3 {
		t.Fatalf("rmsnorm f16 rel-RMS %.3e too high", rms)
	}
}

func TestA1SwiGLUF16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const n = 4096
	rng := rand.New(rand.NewSource(2))
	gf := make([]float32, n)
	uf := make([]float32, n)
	for i := range gf {
		gf[i] = float32(rng.NormFloat64())
		uf[i] = float32(rng.NormFloat64())
	}
	gPtr, gr := uploadF16(t, gf)
	defer cuda.FreeDev(gPtr)
	uPtr, ur := uploadF16(t, uf)
	defer cuda.FreeDev(uPtr)
	if rc := cuda.SwiGLUF16(gPtr, uPtr, n); rc != 0 {
		t.Fatalf("swiglu_f16 rc=%d", rc)
	}
	got := downloadF16(gPtr, n)
	ref := make([]float32, n)
	for i := range ref {
		v := gr[i]
		s := float32(1.0 / (1.0 + math.Exp(-float64(v))))
		ref[i] = v * s * ur[i]
	}
	rms := relRMS(ref, got)
	t.Logf("swiglu f16 rel-RMS %.3e", rms)
	if rms > 3e-3 {
		t.Fatalf("swiglu f16 rel-RMS %.3e too high", rms)
	}
}

func TestA1RoPEF16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const seq, heads, hd = 4, 2, 64
	half := hd / 2
	rng := rand.New(rand.NewSource(3))
	xf := make([]float32, seq*heads*hd)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64())
	}
	attrs := backend.RoPEAttrs{Base: 10000, Heads: heads, PosOffset: 5}
	invD, posDiv := backend.RoPEFreqs(hd, attrs)
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, half)
	invDev.UploadF32(inv)
	defer invDev.Free()
	xPtr, xr := uploadF16(t, xf)
	defer cuda.FreeDev(xPtr)
	if rc := cuda.RoPEF16(xPtr, invDev.DevPtr(), seq, heads, hd, attrs.PosOffset, posDiv); rc != 0 {
		t.Fatalf("rope_f16 rc=%d", rc)
	}
	got := downloadF16(xPtr, seq*heads*hd)
	ref := make([]float32, seq*heads*hd)
	copy(ref, xr)
	for p := 0; p < seq; p++ {
		pos := float64(attrs.PosOffset+p) / posDiv
		for h := 0; h < heads; h++ {
			base := p*heads*hd + h*hd
			for i := 0; i < half; i++ {
				ang := pos * float64(inv[i])
				c := float32(math.Cos(ang))
				s := float32(math.Sin(ang))
				qi, qih := xr[base+i], xr[base+i+half]
				ref[base+i] = qi*c - qih*s
				ref[base+i+half] = qih*c + qi*s
			}
		}
	}
	rms := relRMS(ref, got)
	t.Logf("rope f16 rel-RMS %.3e", rms)
	if rms > 3e-3 {
		t.Fatalf("rope f16 rel-RMS %.3e too high", rms)
	}
}
