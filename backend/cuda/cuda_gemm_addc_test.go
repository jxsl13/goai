//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestGemmF16PureAddCParity: the fused residual GEMM (C += A·W, beta=1) must match the unfused
// path (tmp = A·W; C += tmp) within f16 rounding — both use f16-accumulate, so they should be
// numerically equivalent, validating the +1-2% residual-fusion win is not a correctness trade.
func TestGemmF16PureAddCParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 8, 128, 96
	rng := rand.New(rand.NewSource(5))
	af := make([]float32, M*K)
	wf := make([]float32, K*N)
	cf := make([]float32, M*N)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.2
	}
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.05
	}
	for i := range cf {
		cf[i] = float32(rng.NormFloat64()) * 0.3
	}
	// upload f32 -> device f32 -> f16 buffers
	toF16 := func(v []float32) unsafe.Pointer {
		d, _ := cuda.NewDeviceF32(1, len(v))
		d.UploadF32(v)
		h := cuda.AllocU16(len(v))
		cuda.CvtF32ToF16(h, d.DevPtr(), len(v))
		d.Free()
		return unsafe.Pointer(h)
	}
	a16 := toF16(af)
	w16 := toF16(wf)
	cFused := toF16(cf)
	cUnfused := toF16(cf)
	tmp := cuda.AllocU16(M * N)
	defer func() {
		cuda.FreeDev(a16)
		cuda.FreeDev(w16)
		cuda.FreeDev(cFused)
		cuda.FreeDev(cUnfused)
		cuda.FreeDev(tmp)
	}()

	// fused: cFused += a·w
	if rc := cuda.GemmF16PureAddC(a16, w16, cFused, M, K, N); rc != 0 {
		t.Fatalf("addc rc %d", rc)
	}
	// unfused: tmp = a·w ; cUnfused += tmp
	if rc := cuda.GemmF16Pure(a16, w16, tmp, M, K, N); rc != 0 {
		t.Fatalf("pure rc %d", rc)
	}
	cuda.AddF16(cUnfused, tmp, M*N)

	dl := func(p unsafe.Pointer) []float32 {
		d, _ := cuda.NewDeviceF32(1, M*N)
		cuda.CvtF16ToF32(d.DevPtr(), p, M*N)
		out := make([]float32, M*N)
		d.DownloadF32(out)
		d.Free()
		return out
	}
	fu := dl(cFused)
	un := dl(cUnfused)
	var num, den float64
	for i := range fu {
		d := float64(fu[i] - un[i])
		num += d * d
		den += float64(un[i]) * float64(un[i])
	}
	rel := math.Sqrt(num / den)
	if rel > 1e-3 {
		t.Fatalf("fused vs unfused residual GEMM rel-RMS %.3e too high", rel)
	}
	t.Logf("fused (beta=1) vs unfused (GEMM+AddF16) residual: rel-RMS %.3e — equivalent", rel)
}
