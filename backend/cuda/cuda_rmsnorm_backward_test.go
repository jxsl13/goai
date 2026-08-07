//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestRMSNormBackward validates the GPU RMSNorm VJP against a host-analytic gradient AND against a
// central finite-difference of the forward (which confirms the formula, not just the kernel).
func TestRMSNormBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, D = 24, 128
	const eps = 1e-5
	rng := rand.New(rand.NewSource(11))
	x := make([]float32, M*D)
	g := make([]float32, D)
	dy := make([]float32, M*D)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range g {
		g[i] = float32(rng.NormFloat64()*0.5 + 1)
	}
	for i := range dy {
		dy[i] = float32(rng.NormFloat64())
	}

	up := func(rows, cols int, h []float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(rows, cols)
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
	xd, gd, dyd := up(M, D, x), up(1, D, g), up(M, D, dy)
	dxd, dgd := up(M, D, nil), up(1, D, nil)
	defer xd.Free()
	defer gd.Free()
	defer dyd.Free()
	defer dxd.Free()
	defer dgd.Free()
	if err := cuda.RMSNormBackward(dxd, dgd, xd, dyd, gd, eps); err != nil {
		t.Fatal(err)
	}
	gotDx := make([]float32, M*D)
	gotDg := make([]float32, D)
	dxd.DownloadF32(gotDx)
	dgd.DownloadF32(gotDg)

	// Host analytic.
	refDx := make([]float64, M*D)
	refDg := make([]float64, D)
	for m := 0; m < M; m++ {
		var ss float64
		for j := 0; j < D; j++ {
			v := float64(x[m*D+j])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(D)+eps)
		var a float64
		for j := 0; j < D; j++ {
			a += float64(dy[m*D+j]) * float64(g[j]) * float64(x[m*D+j])
		}
		coef := a * inv * inv * inv / float64(D)
		for j := 0; j < D; j++ {
			refDx[m*D+j] = float64(g[j])*float64(dy[m*D+j])*inv - float64(x[m*D+j])*coef
			refDg[j] += float64(dy[m*D+j]) * float64(x[m*D+j]) * inv
		}
	}
	var maxDx, maxDg float64
	for i := range refDx {
		if d := math.Abs(refDx[i] - float64(gotDx[i])); d > maxDx {
			maxDx = d
		}
	}
	for j := range refDg {
		if d := math.Abs(refDg[j] - float64(gotDg[j])); d > maxDg {
			maxDg = d
		}
	}

	// Finite-difference of L = Σ dy·y (so dL/dx = the VJP dx, dL/dgamma = dgamma), a few coords.
	fwdRow := func(xrow []float64, gv []float64) []float64 {
		var ss float64
		for j := 0; j < D; j++ {
			ss += xrow[j] * xrow[j]
		}
		inv := 1.0 / math.Sqrt(ss/float64(D)+eps)
		y := make([]float64, D)
		for j := 0; j < D; j++ {
			y[j] = xrow[j] * inv * gv[j]
		}
		return y
	}
	gv := make([]float64, D)
	for j := range gv {
		gv[j] = float64(g[j])
	}
	rowL := func(xrow, gvv []float64, m int) float64 {
		y := fwdRow(xrow, gvv)
		var l float64
		for j := 0; j < D; j++ {
			l += float64(dy[m*D+j]) * y[j]
		}
		return l
	}
	const h = 1e-3
	var maxFD float64
	for _, mk := range [][2]int{{0, 0}, {3, 17}, {10, 64}, {23, 127}} {
		m, k := mk[0], mk[1]
		xrow := make([]float64, D)
		for j := 0; j < D; j++ {
			xrow[j] = float64(x[m*D+j])
		}
		xrow[k] += h
		lp := rowL(xrow, gv, m)
		xrow[k] -= 2 * h
		lm := rowL(xrow, gv, m)
		xrow[k] += h
		fdDx := (lp - lm) / (2 * h)
		if d := math.Abs(fdDx - refDx[m*D+k]); d > maxFD {
			maxFD = d
		}
	}

	t.Logf("RMSNorm backward: GPU vs analytic dx %.3e dgamma %.3e; analytic vs finite-diff %.3e", maxDx, maxDg, maxFD)
	if maxDx > 1e-3 || maxDg > 1e-3 {
		t.Fatalf("GPU RMSNorm backward diverges from analytic: dx %.3e dgamma %.3e", maxDx, maxDg)
	}
	if maxFD > 1e-2 {
		t.Fatalf("analytic RMSNorm dx disagrees with finite-difference: %.3e", maxFD)
	}
}
