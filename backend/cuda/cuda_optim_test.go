//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestDeviceAdamMatchesCPU runs the GPU-resident DeviceAdam and a host reference implementing nn.Adam's
// exact f32-parameter / f64-moment AdamW math for the same parameters and gradient sequence, and checks
// they agree. This validates the GPU optimizer step (the first piece of a GPU training path) against the
// established CPU optimizer.
func TestDeviceAdamMatchesCPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	sizes := []int{2048, 257, 4096} // includes a non-warp-multiple size
	const lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
	rng := rand.New(rand.NewSource(3))

	hp := make([][]float32, len(sizes)) // host reference params
	hm := make([][]float64, len(sizes)) // host reference first moments
	hv := make([][]float64, len(sizes)) // host reference second moments
	dp := make([]*cuda.DeviceF32, len(sizes))
	for i, n := range sizes {
		hp[i] = make([]float32, n)
		hm[i] = make([]float64, n)
		hv[i] = make([]float64, n)
		for j := range hp[i] {
			hp[i][j] = float32(rng.NormFloat64())
		}
		d, err := cuda.NewDeviceF32(1, n)
		if err != nil {
			t.Fatal(err)
		}
		if err := d.UploadF32(hp[i]); err != nil {
			t.Fatal(err)
		}
		dp[i] = d
		defer d.Free()
	}
	opt, err := cuda.NewDeviceAdam(sizes, lr, b1, b2, eps, wd)
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Free()

	const steps = 20
	for s := 1; s <= steps; s++ {
		grads := make([][]float32, len(sizes))
		dgs := make([]*cuda.DeviceF32, len(sizes))
		for i, n := range sizes {
			grads[i] = make([]float32, n)
			for j := range grads[i] {
				grads[i][j] = float32(rng.NormFloat64()) * 0.1
			}
			dg, err := cuda.NewDeviceF32(1, n)
			if err != nil {
				t.Fatal(err)
			}
			if err := dg.UploadF32(grads[i]); err != nil {
				t.Fatal(err)
			}
			dgs[i] = dg
		}
		if err := opt.Step(dp, dgs); err != nil {
			t.Fatal(err)
		}
		for _, dg := range dgs {
			dg.Free()
		}
		// Host reference — nn.Adam f32-param/f64-moment fast path.
		c1 := 1 - math.Pow(b1, float64(s))
		c2 := 1 - math.Pow(b2, float64(s))
		ic1, ic2 := 1/c1, 1/c2
		decay := 1 - lr*wd
		for i, n := range sizes {
			for j := 0; j < n; j++ {
				gv := float64(grads[i][j])
				hm[i][j] = b1*hm[i][j] + (1-b1)*gv
				hv[i][j] = b2*hv[i][j] + (1-b2)*gv*gv
				mh := hm[i][j] * ic1
				vh := hv[i][j] * ic2
				hp[i][j] = float32(float64(hp[i][j])*decay - lr*mh/(math.Sqrt(vh)+eps))
			}
		}
	}

	var maxAbs float64
	for i, n := range sizes {
		got := make([]float32, n)
		if err := dp[i].DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < n; j++ {
			if d := math.Abs(float64(got[j] - hp[i][j])); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("DeviceAdam vs CPU nn.Adam math after %d steps: max abs diff %.3e", steps, maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("DeviceAdam diverges from CPU Adam: %.3e", maxAbs)
	}
}
