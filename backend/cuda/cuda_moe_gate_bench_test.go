//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// benchMoEGate drives cu_moe_gate at a sparse-MoE decode/prefill batch shape (rows tokens,
// E experts, top-K). The kernel is one warp per row; this measures the warp-cooperative
// argmax path vs the prior one-thread-per-row serial scan.
func benchMoEGate(b *testing.B, rows, e, k int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	rng := rand.New(rand.NewSource(17))
	logits := make([]float32, rows*e)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64())
	}
	dlog, err := cuda.NewDeviceBufferF32(logits)
	if err != nil {
		b.Fatal(err)
	}
	defer dlog.Free()
	dw, err := cuda.NewDeviceF32(rows, e)
	if err != nil {
		b.Fatal(err)
	}
	defer dw.Free()
	if err := rec.MoEGate(dlog, dw, rows, e, k, 0, 1); err != nil {
		b.Fatal(err)
	}
	rec.Wait()
	b.ResetTimer()
	for range b.N {
		rec.MoEGate(dlog, dw, rows, e, k, 0, 1)
	}
	rec.Wait()
	b.StopTimer()
}

func BenchmarkMoEGate_512x128k8(b *testing.B) { benchMoEGate(b, 512, 128, 8) }
func BenchmarkMoEGate_512x256k8(b *testing.B) { benchMoEGate(b, 512, 256, 8) }
func BenchmarkMoEGate_256x64k2(b *testing.B)  { benchMoEGate(b, 256, 64, 2) }
