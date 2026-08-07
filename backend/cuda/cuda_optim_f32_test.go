//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestDeviceAdamF32MatchesHostF32 validates the f32-moment optimizer against a host f32 AdamW reference
// (the incumbent precision it targets — NOT the f64 nn.Adam). Over several steps the device params must
// match the host f32 computation to f32 tolerance (device sqrtf vs host float32(sqrt) differ by ~1 ulp).
func TestDeviceAdamF32MatchesHostF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	sizes := []int{2048, 257, 4096}
	const lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
	rng := rand.New(rand.NewSource(3))

	hp := make([][]float32, len(sizes)) // host params
	hm := make([][]float32, len(sizes)) // host f32 first moments
	hv := make([][]float32, len(sizes)) // host f32 second moments
	dp := make([]*cuda.DeviceF32, len(sizes))
	for i, n := range sizes {
		hp[i] = make([]float32, n)
		hm[i] = make([]float32, n)
		hv[i] = make([]float32, n)
		for j := range hp[i] {
			hp[i][j] = float32(rng.NormFloat64())
		}
		d, _ := cuda.NewDeviceF32(1, n)
		d.UploadF32(hp[i])
		dp[i] = d
		defer d.Free()
	}
	opt, err := cuda.NewDeviceAdamF32(sizes, lr, b1, b2, eps, wd)
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Free()

	sqrtf := func(x float32) float32 { return float32(math.Sqrt(float64(x))) }
	const steps = 15
	for s := 1; s <= steps; s++ {
		grads := make([][]float32, len(sizes))
		dgs := make([]*cuda.DeviceF32, len(sizes))
		for i, n := range sizes {
			grads[i] = make([]float32, n)
			for j := range grads[i] {
				grads[i][j] = float32(rng.NormFloat64() * 0.1)
			}
			g, _ := cuda.NewDeviceF32(1, n)
			g.UploadF32(grads[i])
			dgs[i] = g
		}
		if err := opt.Step(dp, dgs); err != nil {
			t.Fatal(err)
		}
		// host f32 reference (same arithmetic as the kernel, all float32)
		ic1 := float32(1.0 / (1.0 - math.Pow(b1, float64(s))))
		ic2 := float32(1.0 / (1.0 - math.Pow(b2, float64(s))))
		decay := float32(1.0 - lr*wd)
		for i, n := range sizes {
			for j := 0; j < n; j++ {
				gv := grads[i][j]
				hm[i][j] = b1*hm[i][j] + (1-b1)*gv
				hv[i][j] = b2*hv[i][j] + (1-b2)*gv*gv
				mh, vh := hm[i][j]*ic1, hv[i][j]*ic2
				hp[i][j] = hp[i][j]*decay - lr*mh/(sqrtf(vh)+eps)
			}
			dgs[i].Free()
		}
	}
	var maxErr float64
	for i, n := range sizes {
		got := make([]float32, n)
		dp[i].DownloadF32(got)
		for j := 0; j < n; j++ {
			e := math.Abs(float64(got[j]-hp[i][j])) / (math.Abs(float64(hp[i][j])) + 1e-6)
			if e > maxErr {
				maxErr = e
			}
		}
	}
	t.Logf("DeviceAdamF32 vs host-f32 over %d steps: max rel err = %.3g", steps, maxErr)
	if maxErr > 1e-3 {
		t.Fatalf("DeviceAdamF32 diverges from host f32 reference: rel err %g", maxErr)
	}
}
