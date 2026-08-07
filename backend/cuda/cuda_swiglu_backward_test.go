//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestSwiGLUBackward validates the GPU SwiGLU VJP two ways: the kernel output matches the host analytic
// gradient, AND the analytic gradient matches a finite-difference of the forward o=SiLU(g)*u (so the
// formula itself is confirmed, not just that the kernel reproduces it).
func TestSwiGLUBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const n = 4096
	rng := rand.New(rand.NewSource(9))
	g := make([]float32, n)
	u := make([]float32, n)
	dO := make([]float32, n)
	for i := 0; i < n; i++ {
		g[i] = float32(rng.NormFloat64() * 2)
		u[i] = float32(rng.NormFloat64())
		dO[i] = float32(rng.NormFloat64())
	}
	silu := func(x float64) float64 { return x / (1 + math.Exp(-x)) }

	up := func(h []float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(1, n)
		if err != nil {
			t.Fatal(err)
		}
		if h != nil {
			if err := d.UploadF32(h); err != nil {
				t.Fatal(err)
			}
		}
		return d
	}
	gd, ud, dod := up(g), up(u), up(dO)
	dg, du := up(nil), up(nil)
	defer gd.Free()
	defer ud.Free()
	defer dod.Free()
	defer dg.Free()
	defer du.Free()
	if err := cuda.SwiGLUBackward(dg, du, gd, ud, dod); err != nil {
		t.Fatal(err)
	}
	gotDg := make([]float32, n)
	gotDu := make([]float32, n)
	dg.DownloadF32(gotDg)
	du.DownloadF32(gotDu)

	// (1) GPU vs host analytic.
	var maxDg, maxDu float64
	for i := 0; i < n; i++ {
		x := float64(g[i])
		s := 1 / (1 + math.Exp(-x))
		refDu := float64(dO[i]) * silu(x)
		refDg := float64(dO[i]) * float64(u[i]) * (s * (1 + x*(1-s)))
		if d := math.Abs(refDu - float64(gotDu[i])); d > maxDu {
			maxDu = d
		}
		if d := math.Abs(refDg - float64(gotDg[i])); d > maxDg {
			maxDg = d
		}
	}

	// (2) Analytic ∂o/∂g and ∂o/∂u vs central finite-difference of the forward, on a sample.
	const h = 1e-3
	var maxFD float64
	for _, i := range []int{0, 1, 100, 1000, 4095} {
		x := float64(g[i])
		s := 1 / (1 + math.Exp(-x))
		dodg := float64(u[i]) * (s * (1 + x*(1-s))) // ∂o/∂g
		fdG := (silu(x+h)*float64(u[i]) - silu(x-h)*float64(u[i])) / (2 * h)
		if d := math.Abs(dodg - fdG); d > maxFD {
			maxFD = d
		}
		dodu := silu(x) // ∂o/∂u
		fdU := (silu(x)*(float64(u[i])+h) - silu(x)*(float64(u[i])-h)) / (2 * h)
		if d := math.Abs(dodu - fdU); d > maxFD {
			maxFD = d
		}
	}

	t.Logf("SwiGLU backward: GPU vs analytic dg %.3e du %.3e; analytic vs finite-diff %.3e", maxDg, maxDu, maxFD)
	if maxDg > 1e-3 || maxDu > 1e-3 {
		t.Fatalf("GPU SwiGLU backward diverges from analytic: dg %.3e du %.3e", maxDg, maxDu)
	}
	if maxFD > 1e-3 {
		t.Fatalf("analytic SwiGLU gradient disagrees with finite-difference: %.3e", maxFD)
	}
}
