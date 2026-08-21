//go:build darwin && cgo

package metal_test

import (
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// syntheticKQuant builds a weight blob of the right shape for a K-quant type.
// Only the timing of the resident matvec is read from it; the kernels have no
// data-dependent control flow, so deterministic filler is sufficient. d/dmin are
// pinned to finite f16 values so no NaN/Inf path can be entered.
func syntheticKQuant(n, k, blockBytes, scaleHdr int) []byte {
	blocks := n * (k / 256)
	out := make([]byte, blocks*blockBytes)
	for block := range blocks {
		base := block * blockBytes
		out[base], out[base+1] = 0, 0x3c // f16(1)
		if scaleHdr > 2 {
			out[base+2], out[base+3] = 0, 0x38 // f16(0.5)
		}
		for i := scaleHdr; i < blockBytes; i++ {
			out[base+i] = byte((block*17 + i*29) & 0xff)
		}
	}
	return out
}

// BenchmarkMetalKQuantM1Gap times the M=1 resident decode leaf for every K-quant
// type at one shape. All five K-quant types have cooperative kernels; this tracks
// their relative rates and catches regressions in the decode leaves.
func BenchmarkMetalKQuantM1Gap(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const k, n = 2048, 2048
	for _, tc := range []struct {
		name       string
		qtype      uint32
		blockBytes int
		scaleHdr   int
	}{
		{"Q2K_cooperative", uint32(gguf.Q2_K), 84, 4},
		{"Q3K_cooperative", uint32(gguf.Q3_K), 110, 2},
		{"Q4K_cooperative", uint32(gguf.Q4_K), 144, 4},
		{"Q5K_cooperative", uint32(gguf.Q5_K), 176, 4},
		{"Q6K_cooperative", uint32(gguf.Q6_K), 210, 2},
	} {
		b.Run(tc.name, func(b *testing.B) {
			weight := syntheticKQuant(n, k, tc.blockBytes, tc.scaleHdr)
			raw, err := metal.Backend{}.UploadQuant(weight, tc.qtype, n, k)
			if err != nil {
				b.Skipf("UploadQuant %s: %v", tc.name, err)
			}
			rw := raw.(*metal.ResidentQWeight)
			defer rw.Close()
			x, err := metal.NewDeviceBufferF32(make([]float32, k))
			if err != nil {
				b.Fatal(err)
			}
			defer x.Release()
			o, err := metal.NewDeviceBufferF32(make([]float32, n))
			if err != nil {
				b.Fatal(err)
			}
			defer o.Release()
			run := func() {
				r, err := metal.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if err := r.QMatMulResident(x, rw, o, 1); err != nil {
					r.Free()
					b.Fatal(err)
				}
				if err := r.Finish(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
			}
			for range 20 {
				run()
			}
			b.SetBytes(int64(len(weight)))
			b.ResetTimer()
			for range b.N {
				run()
			}
		})
	}
}

// BenchmarkMetalQ5KWideLoad is the same-binary promotion gate for replacing
// byte-granular Q5_K loads with aligned packed loads on M2. Each shape reuses one
// resident weight and identical device buffers across both arms. Every iteration
// times both arms and reverses their AB/BA order, cancelling the substantial
// warm-state bias seen when Go sub-benchmarks ran control and candidate in blocks.
func BenchmarkMetalQ5KWideLoad(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	previousCooperative := metal.SetQ5KCooperative(true)
	previousWideLoad := metal.SetQ5KWideLoad(false)
	defer func() {
		metal.SetQ5KWideLoad(previousWideLoad)
		metal.SetQ5KCooperative(previousCooperative)
	}()
	for _, shape := range []struct {
		name string
		k    int
		n    int
	}{
		{name: "kv", k: 2048, n: 256},
		{name: "square", k: 2048, n: 2048},
		{name: "mid_up", k: 2048, n: 3072},
		{name: "mid_down", k: 4096, n: 2048},
		{name: "gate_up", k: 2048, n: 5632},
		{name: "down", k: 5632, n: 2048},
	} {
		shape := shape
		weight := syntheticKQuant(shape.n, shape.k, 176, 4)
		raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q5_K), shape.n, shape.k)
		if err != nil {
			b.Fatalf("UploadQuant %s: %v", shape.name, err)
		}
		rw := raw.(*metal.ResidentQWeight)
		x, err := metal.NewDeviceBufferF32(make([]float32, shape.k))
		if err != nil {
			rw.Close()
			b.Fatal(err)
		}
		o, err := metal.NewDeviceBufferF32(make([]float32, shape.n))
		if err != nil {
			x.Release()
			rw.Close()
			b.Fatal(err)
		}
		run := func(tb testing.TB) {
			r, err := metal.NewRecorder()
			if err != nil {
				tb.Fatal(err)
			}
			if err := r.QMatMulResident(x, rw, o, 1); err != nil {
				r.Free()
				tb.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				r.Free()
				tb.Fatal(err)
			}
			r.Free()
		}
		benchmarkName := shape.name + "_k" + strconv.Itoa(shape.k) + "_n" + strconv.Itoa(shape.n)
		b.Run(benchmarkName, func(b *testing.B) {
			for i := range 40 {
				metal.SetQ5KWideLoad(i%2 == 1)
				run(b)
			}
			const measuredPerArm = 32
			timed := func(wide bool) time.Duration {
				metal.SetQ5KWideLoad(wide)
				// Exclude the transition dispatch: production decode repeats one
				// pipeline, while the benchmark deliberately switches pipelines.
				run(b)
				start := time.Now()
				for range measuredPerArm {
					run(b)
				}
				return time.Since(start)
			}
			var controlTime, wideTime time.Duration
			b.ResetTimer()
			for i := range b.N {
				if i%2 == 0 {
					controlTime += timed(false)
					wideTime += timed(true)
				} else {
					wideTime += timed(true)
					controlTime += timed(false)
				}
			}
			denominator := float64(b.N * measuredPerArm)
			controlNS := float64(controlTime.Nanoseconds()) / denominator
			wideNS := float64(wideTime.Nanoseconds()) / denominator
			b.ReportMetric(controlNS, "control-ns/op")
			b.ReportMetric(wideNS, "wide-ns/op")
			b.ReportMetric(controlNS/wideNS, "speedup-x")
		})
		o.Release()
		x.Release()
		rw.Close()
	}
}
