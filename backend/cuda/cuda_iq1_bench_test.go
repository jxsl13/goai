//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// benchIQ1 reports IQ1_S GEMV throughput. Tw80 escape route: the ternary {−1,0,+1} grid is packed
// to 2 bits/entry (2048×8 → 4 KB) and gathered from SHARED as one uint16/lane (not 8 f32 from L2),
// then decoded code→{−1,0,+1} in ALU. The tiny 2-byte per-lane read dodges the 8-way bank floor that
// caps the f32-grid path (perf-notes-cuda.md R6). Measured on RTX 3060 (bit-exact): 2048² 22007 →
// 14642 ns (1.5×, ~Q4_K parity), 5632×2048 55451 → 31360 (1.77×).
func benchIQ1(b *testing.B, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	sbs := k / 256
	raw := make([]byte, n*sbs*50)
	for i := 0; i < n*sbs; i++ {
		off := i * 50
		binary.LittleEndian.PutUint16(raw[off:], 0x2C00)
		for p := 0; p < 32; p++ {
			raw[off+2+p] = byte(rng.Intn(256))
		}
		for g := 0; g < 8; g++ {
			binary.LittleEndian.PutUint16(raw[off+34+g*2:], uint16(rng.Intn(65536)))
		}
	}
	rq, err := cuda.NewResidentBIQ1SFromBlocks(raw, k, n)
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

func BenchmarkGemvIQ1_2048(b *testing.B)      { benchIQ1(b, 2048, 2048) }
func BenchmarkGemvIQ1_5632x2048(b *testing.B) { benchIQ1(b, 5632, 2048) }
