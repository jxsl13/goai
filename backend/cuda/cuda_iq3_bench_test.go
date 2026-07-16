//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// benchIQ3 reports IQ3_XXS GEMV throughput at decode shapes. Tw80: the grid gather was the
// bottleneck (stub-probe: removing it → 2.7×), so the grid now lives in shared memory. Measured on
// RTX 3060: grid-in-shared cut 2048² 32263→21574 ns (~1.5×), 5632×2048 85582→57995 ns — bit-exact.
// The gather (not the byte-assembled block reads) dominated i-quant decode; shared moves it off L2.
func benchIQ3(b *testing.B, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	sbs := k / 256
	raw := make([]byte, n*sbs*98)
	for i := 0; i < n*sbs; i++ {
		off := i * 98
		binary.LittleEndian.PutUint16(raw[off:], 0x2C00)
		for p := 0; p < 64; p++ {
			raw[off+2+p] = byte(rng.Intn(256))
		}
		for g := 0; g < 8; g++ {
			binary.LittleEndian.PutUint32(raw[off+66+g*4:], rng.Uint32())
		}
	}
	rq, err := cuda.NewResidentBIQ3XXSFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	defer rq.Free()
	a := make([]float32, k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	da, _ := cuda.NewDeviceF32(1, k)
	defer da.Free()
	da.UploadF32(a)
	out, _ := cuda.NewDeviceF32(1, n)
	defer out.Free()
	rq.QMatMulInto(da, out)
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		rq.QMatMulInto(da, out)
	}
	cuda.GraphSync()
	b.StopTimer()
}

func BenchmarkGemvIQ3_2048(b *testing.B)      { benchIQ3(b, 2048, 2048) }
func BenchmarkGemvIQ3_5632x2048(b *testing.B) { benchIQ3(b, 5632, 2048) }
func BenchmarkGemvIQ3_2048x256(b *testing.B)  { benchIQ3(b, 2048, 256) }
