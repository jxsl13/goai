//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
)

// TestDeviceF32RoPERagged verifies the f32 per-sequence-position RoPE (the primitive a quantized/f32
// continuous-batch decode needs) rotates each row at its OWN position, matching a host reference.
func TestDeviceF32RoPERagged(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const heads, hd, rows = 4, 64, 3
	cols := heads * hd
	attrs := backend.RoPEAttrs{Base: 10000, Heads: heads}
	invD, posDiv := backend.RoPEFreqs(hd, attrs)
	if posDiv == 0 {
		posDiv = 1
	}
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, err := cuda.NewDeviceF32(1, len(inv32))
	if err != nil {
		t.Fatal(err)
	}
	defer invDev.Free()
	if err := invDev.UploadF32(inv32); err != nil {
		t.Fatal(err)
	}

	positions := []int32{5, 128, 1000} // three sequences at very different positions
	data := make([]float32, rows*cols)
	rng := rand.New(rand.NewSource(11))
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	dx, err := cuda.NewDeviceF32(rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	defer dx.Free()
	if err := dx.UploadF32(data); err != nil {
		t.Fatal(err)
	}
	if err := dx.RoPERagged(invDev, positions, attrs); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, rows*cols)
	if err := dx.DownloadF32(got); err != nil {
		t.Fatal(err)
	}

	// Host reference — mirror the kernel: row p rotated for positions[p].
	half := hd / 2
	ref := make([]float32, rows*cols)
	copy(ref, data)
	for p := 0; p < rows; p++ {
		pos := float64(positions[p]) / posDiv
		for h := 0; h < heads; h++ {
			base := p*cols + h*hd
			for i := 0; i < half; i++ {
				ang := pos * invD[i]
				c, s := math.Cos(ang), math.Sin(ang)
				qi, qih := float64(data[base+i]), float64(data[base+i+half])
				ref[base+i] = float32(qi*c - qih*s)
				ref[base+i+half] = float32(qih*c + qi*s)
			}
		}
	}
	var maxAbs float64
	for i := range ref {
		if d := math.Abs(float64(got[i] - ref[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("DeviceF32.RoPERagged max abs diff vs host: %.3e", maxAbs)
	if maxAbs > 1e-3 {
		t.Fatalf("DeviceF32.RoPERagged diverges: %.3e", maxAbs)
	}
}
