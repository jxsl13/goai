//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// modelLikeSizes returns a parameter-tensor size list resembling a real transformer's per-layer weights
// (a few large projections + several small norms/biases) across `layers` layers — the many-tensor regime
// where DeviceAdam.Step's one-kernel-launch-per-parameter loop pays launch overhead.
func modelLikeSizes(layers int) []int {
	var s []int
	for l := 0; l < layers; l++ {
		s = append(s,
			1024*1024, // wq
			1024*1024, // wo
			1024*256,  // wk
			1024*256,  // wv
			1024*2816, // wgate
			1024*2816, // wup
			2816*1024, // wdown
			1024,      // attn norm
			1024,      // ffn norm
		)
	}
	s = append(s, 32000*1024, 1024, 32000*1024) // embed, final norm, lm head
	return s
}

// BenchmarkDeviceAdamStep measures the AdamW step over a model-like parameter set. Compare ms/step against
// the memory-bandwidth floor (Σ 24 bytes/param / BW): a large gap means the per-parameter kernel-launch
// loop is launch/occupancy-bound — the gate for a multi-tensor (single-launch) fused Adam.
func BenchmarkDeviceAdamStep(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	for _, layers := range []int{4, 12} {
		sizes := modelLikeSizes(layers)
		b.Run(benchAdamName(len(sizes)), func(b *testing.B) {
			params := make([]*cuda.DeviceF32, len(sizes))
			grads := make([]*cuda.DeviceF32, len(sizes))
			var totalParams int64
			for i, n := range sizes {
				p, _ := cuda.NewDeviceF32(1, n)
				g, _ := cuda.NewDeviceF32(1, n)
				params[i], grads[i] = p, g
				totalParams += int64(n)
			}
			defer func() {
				for i := range params {
					params[i].Free()
					grads[i].Free()
				}
			}()
			opt, err := cuda.NewDeviceAdam(sizes, 1e-3, 0.9, 0.999, 1e-8, 0.01)
			if err != nil {
				b.Fatal(err)
			}
			defer opt.Free()
			opt.Step(params, grads) // warm up
			cuda.GraphSync()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := opt.Step(params, grads); err != nil {
					b.Fatal(err)
				}
			}
			cuda.GraphSync()
			b.StopTimer()
			b.ReportMetric(float64(len(sizes)), "tensors")
			b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/step")
			// BW floor: 24 bytes/param (p+g read, m+v read+write ~ f64) — rough lower bound.
			b.ReportMetric(float64(totalParams*24)/(360e9)*1e3, "ms-BW-floor")
		})
	}
}

// BenchmarkDeviceAdamStepF32 is the f32-moment twin — A/B against BenchmarkDeviceAdamStep to measure
// whether dropping f64 (slow sqrt/div on GA106) to incumbent f32 precision speeds the step.
func BenchmarkDeviceAdamStepF32(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	for _, layers := range []int{4, 12} {
		sizes := modelLikeSizes(layers)
		b.Run(benchAdamName(len(sizes)), func(b *testing.B) {
			params := make([]*cuda.DeviceF32, len(sizes))
			grads := make([]*cuda.DeviceF32, len(sizes))
			for i, n := range sizes {
				p, _ := cuda.NewDeviceF32(1, n)
				g, _ := cuda.NewDeviceF32(1, n)
				params[i], grads[i] = p, g
			}
			defer func() {
				for i := range params {
					params[i].Free()
					grads[i].Free()
				}
			}()
			opt, err := cuda.NewDeviceAdamF32(sizes, 1e-3, 0.9, 0.999, 1e-8, 0.01)
			if err != nil {
				b.Fatal(err)
			}
			defer opt.Free()
			opt.Step(params, grads)
			cuda.GraphSync()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := opt.Step(params, grads); err != nil {
					b.Fatal(err)
				}
			}
			cuda.GraphSync()
			b.StopTimer()
			b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/step")
		})
	}
}

func benchAdamName(n int) string {
	return "tensors" + itoaA(n)
}

func itoaA(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
