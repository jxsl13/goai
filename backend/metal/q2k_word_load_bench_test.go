//go:build darwin && cgo

package metal_test

import (
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

func BenchmarkMetalQ2KWordLoad(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	previousCooperative := metal.SetQ2KCooperative(true)
	previousWord := metal.SetQ2KWordLoad(false)
	defer func() {
		metal.SetQ2KWordLoad(previousWord)
		metal.SetQ2KCooperative(previousCooperative)
	}()
	for _, shape := range []struct {
		name string
		k, n int
	}{{"kv", 2048, 256}, {"square", 2048, 2048}, {"mid_up", 2048, 3072}, {"mid_down", 4096, 2048}, {"gate_up", 2048, 5632}, {"down", 5632, 2048}, {"vocab", 2048, 32000}} {
		blocks := shape.n * (shape.k / 256)
		weight := make([]byte, blocks*84)
		for block := range blocks {
			base := block * 84
			for i := 0; i < 80; i++ {
				weight[base+i] = byte((block*17 + i*29) & 0xff)
			}
			weight[base+80], weight[base+81] = 0, 0x3c
			weight[base+82], weight[base+83] = 0, 0x38
		}
		raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q2_K), shape.n, shape.k)
		if err != nil {
			b.Fatal(err)
		}
		rw := raw.(*metal.ResidentQWeight)
		x, err := metal.NewDeviceBufferF32(make([]float32, shape.k))
		if err != nil {
			b.Fatal(err)
		}
		o, err := metal.NewDeviceBufferF32(make([]float32, shape.n))
		if err != nil {
			b.Fatal(err)
		}
		run := func(tb testing.TB) {
			r, err := metal.NewRecorder()
			if err != nil {
				tb.Fatal(err)
			}
			if err := r.QMatMulResident(x, rw, o, 1); err != nil {
				tb.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				tb.Fatal(err)
			}
			r.Free()
		}
		name := shape.name + "_k" + strconv.Itoa(shape.k) + "_n" + strconv.Itoa(shape.n)
		b.Run(name, func(b *testing.B) {
			for i := range 40 {
				metal.SetQ2KWordLoad(i%2 == 1)
				run(b)
			}
			const measured = 32
			timed := func(word bool) time.Duration {
				metal.SetQ2KWordLoad(word)
				run(b)
				start := time.Now()
				for range measured {
					run(b)
				}
				return time.Since(start)
			}
			var controlTime, wordTime time.Duration
			b.ResetTimer()
			for i := range b.N {
				if i%2 == 0 {
					controlTime += timed(false)
					wordTime += timed(true)
				} else {
					wordTime += timed(true)
					controlTime += timed(false)
				}
			}
			den := float64(b.N * measured)
			controlNS, wordNS := float64(controlTime.Nanoseconds())/den, float64(wordTime.Nanoseconds())/den
			b.ReportMetric(controlNS, "control-ns/op")
			b.ReportMetric(wordNS, "word-ns/op")
			b.ReportMetric(controlNS/wordNS, "speedup-x")
		})
		o.Release()
		x.Release()
		rw.Close()
	}
}
