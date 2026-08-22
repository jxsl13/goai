//go:build darwin && cgo

package metal_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

var metalIQ2TensorSink *tensor.Tensor

// BenchmarkMetalIQ2HostRouteCampaign compares equal host input/output boundaries with
// distinct weights and AB/BA ordering. It decides whether generic QuantLinear may leave ARM64.
func BenchmarkMetalIQ2HostRouteCampaign(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, format := range metalIQ2Cases() {
		b.Run(format.name, func(b *testing.B) {
			for _, shape := range []struct {
				name string
				n, k int
			}{{"kv", 512, 2048}, {"square", 2048, 2048}, {"gate", 5632, 2048}, {"down", 2048, 5632}} {
				b.Run(shape.name, func(b *testing.B) {
					const weightCount = 8
					weights := make([][]byte, 0, weightCount)
					for i := range weightCount {
						weights = append(weights, format.synthetic(shape.n, shape.k, i))
					}
					x := tensor.New(tensor.F32, tensor.Shape{1, shape.k})
					timedCPU := func() time.Duration {
						start := time.Now()
						for _, weight := range weights {
							var err error
							metalIQ2TensorSink, err = gguf.QMatMul(x, weight, format.qt, shape.n, shape.k)
							if err != nil {
								b.Fatal(err)
							}
						}
						return time.Since(start) / weightCount
					}
					timedMetal := func() time.Duration {
						start := time.Now()
						for _, weight := range weights {
							var err error
							metalIQ2TensorSink, err = format.direct(x, weight, shape.n, shape.k)
							if err != nil {
								b.Fatal(err)
							}
						}
						return time.Since(start) / weightCount
					}
					for i := range 8 {
						if i&1 == 0 {
							timedCPU()
							timedMetal()
						} else {
							timedMetal()
							timedCPU()
						}
					}
					var cpuWall, metalWall time.Duration
					b.ResetTimer()
					for i := range b.N {
						if i&1 == 0 {
							cpuWall += timedCPU()
							metalWall += timedMetal()
						} else {
							metalWall += timedMetal()
							cpuWall += timedCPU()
						}
					}
					denominator := float64(b.N)
					cpuNS := float64(cpuWall.Nanoseconds()) / denominator
					metalNS := float64(metalWall.Nanoseconds()) / denominator
					b.ReportMetric(cpuNS, "cpu-wall-ns/op")
					b.ReportMetric(metalNS, "metal-direct-wall-ns/op")
					b.ReportMetric(cpuNS/metalNS, "metal-vs-cpu-wall-x")
				})
			}
		})
	}
}

// BenchmarkMetalIQ2CooperativeCampaign measures the production resident recorder path.
// Sixteen distinct weights defeat cache-hot repetition; scalar A/cooperative/scalar B
// interleaving controls drift while GPU timestamps exclude host submit and wake jitter.
func BenchmarkMetalIQ2CooperativeCampaign(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, format := range metalIQ2Cases() {
		b.Run(format.name, func(b *testing.B) {
			previous := format.toggle(false)
			defer format.toggle(previous)
			for _, shape := range []struct {
				name string
				n, k int
			}{{"kv", 512, 2048}, {"square", 2048, 2048}, {"gate", 5632, 2048}, {"down", 2048, 5632}} {
				b.Run(shape.name, func(b *testing.B) {
					const residentCount = 16
					weights := make([]*metal.ResidentQWeight, 0, residentCount)
					hostWeights := make([][]byte, 0, residentCount)
					for i := range residentCount {
						raw := format.synthetic(shape.n, shape.k, i)
						resident, err := format.upload(raw, shape.n, shape.k)
						if err != nil {
							b.Fatal(err)
						}
						hostWeights = append(hostWeights, raw)
						weights = append(weights, resident)
					}
					defer func() {
						for _, resident := range weights {
							resident.Close()
						}
					}()
					x, err := metal.NewDeviceBufferF32(make([]float32, shape.k))
					if err != nil {
						b.Fatal(err)
					}
					defer x.Release()
					o, err := metal.NewDeviceBufferF32(make([]float32, shape.n))
					if err != nil {
						b.Fatal(err)
					}
					defer o.Release()
					type sample struct{ gpu, wall time.Duration }
					timed := func(cooperative bool) sample {
						format.toggle(cooperative)
						start := time.Now()
						recorder, err := metal.NewRecorder()
						if err != nil {
							b.Fatal(err)
						}
						for _, resident := range weights {
							if err := recorder.QMatMulResident(x, resident, o, 1); err != nil {
								recorder.Free()
								b.Fatal(err)
							}
						}
						if err := recorder.Commit(); err != nil {
							recorder.Free()
							b.Fatal(err)
						}
						if err := recorder.Wait(); err != nil {
							recorder.Free()
							b.Fatal(err)
						}
						recorder.Free()
						return sample{
							gpu:  time.Duration(metal.LastGPUSeconds() * float64(time.Second) / residentCount),
							wall: time.Since(start) / residentCount,
						}
					}
					hostX := tensor.New(tensor.F32, tensor.Shape{1, shape.k})
					timedCPU := func() time.Duration {
						start := time.Now()
						for _, weight := range hostWeights {
							var err error
							metalIQ2TensorSink, err = gguf.QMatMul(hostX, weight, format.qt, shape.n, shape.k)
							if err != nil {
								b.Fatal(err)
							}
						}
						return time.Since(start) / residentCount
					}
					for i := range 8 {
						timed(i&1 == 1)
						timedCPU()
					}
					var scalarGPU, scalarWall, cooperativeGPU, cooperativeWall, cpuWall time.Duration
					var unchangedControl float64
					b.ResetTimer()
					for i := range b.N {
						if i&1 == 0 {
							scalarA, cooperative, scalarB := timed(false), timed(true), timed(false)
							scalarGPU += (scalarA.gpu + scalarB.gpu) / 2
							scalarWall += (scalarA.wall + scalarB.wall) / 2
							cooperativeGPU += cooperative.gpu
							cooperativeWall += cooperative.wall
							ratio := float64(scalarA.gpu) / float64(scalarB.gpu)
							if ratio < 1 {
								ratio = 1 / ratio
							}
							unchangedControl += ratio
						} else {
							cooperative, scalarA, scalarB := timed(true), timed(false), timed(false)
							cooperativeGPU += cooperative.gpu
							cooperativeWall += cooperative.wall
							scalarGPU += (scalarA.gpu + scalarB.gpu) / 2
							scalarWall += (scalarA.wall + scalarB.wall) / 2
							ratio := float64(scalarA.gpu) / float64(scalarB.gpu)
							if ratio < 1 {
								ratio = 1 / ratio
							}
							unchangedControl += ratio
						}
						cpuWall += timedCPU()
					}
					denominator := float64(b.N)
					scalarNS := float64(scalarGPU.Nanoseconds()) / denominator
					cooperativeNS := float64(cooperativeGPU.Nanoseconds()) / denominator
					cooperativeWallNS := float64(cooperativeWall.Nanoseconds()) / denominator
					cpuNS := float64(cpuWall.Nanoseconds()) / denominator
					b.ReportMetric(scalarNS, "scalar-gpu-ns/op")
					b.ReportMetric(float64(scalarWall.Nanoseconds())/denominator, "scalar-wall-ns/op")
					b.ReportMetric(cooperativeNS, "cooperative-gpu-ns/op")
					b.ReportMetric(cooperativeWallNS, "cooperative-wall-ns/op")
					b.ReportMetric(cpuNS, "cpu-wall-ns/op")
					b.ReportMetric(scalarNS/cooperativeNS, "cooperative-vs-scalar-x")
					b.ReportMetric(float64(scalarWall.Nanoseconds())/denominator/cooperativeWallNS, "cooperative-vs-scalar-wall-x")
					b.ReportMetric(cpuNS/cooperativeWallNS, "metal-vs-cpu-wall-x")
					b.ReportMetric(unchangedControl/denominator, "unchanged-control-x")
				})
			}
		})
	}
}
