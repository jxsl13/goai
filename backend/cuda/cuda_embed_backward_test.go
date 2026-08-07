//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestEmbedBackward validates the embedding VJP: dTable[ids[i]] += dOut[i], with repeated ids summing.
// It uses ids with deliberate repeats to exercise the atomic scatter-add, and checks against a host
// reference.
func TestEmbedBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const V, d, seq = 50, 32, 40
	rng := rand.New(rand.NewSource(23))
	dOut := make([]float32, seq*d)
	for i := range dOut {
		dOut[i] = float32(rng.NormFloat64())
	}
	ids := make([]int32, seq)
	for i := range ids {
		ids[i] = int32(rng.Intn(V / 3)) // small range → many repeats → exercises atomicAdd
	}

	dOutD, err := cuda.NewDeviceF32(seq, d)
	if err != nil {
		t.Fatal(err)
	}
	defer dOutD.Free()
	if err := dOutD.UploadF32(dOut); err != nil {
		t.Fatal(err)
	}
	dTableD, err := cuda.NewDeviceF32(V, d)
	if err != nil {
		t.Fatal(err)
	}
	defer dTableD.Free()
	if err := cuda.EmbedBackward(dTableD, dOutD, ids); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, V*d)
	dTableD.DownloadF32(got)

	// Host reference: scatter-add.
	ref := make([]float64, V*d)
	for i := 0; i < seq; i++ {
		row := int(ids[i])
		for k := 0; k < d; k++ {
			ref[row*d+k] += float64(dOut[i*d+k])
		}
	}
	var maxAbs float64
	for i := range ref {
		if x := math.Abs(ref[i] - float64(got[i])); x > maxAbs {
			maxAbs = x
		}
	}
	t.Logf("embed backward vs host scatter-add: max abs diff %.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("EmbedBackward diverges: %.3e", maxAbs)
	}
}
