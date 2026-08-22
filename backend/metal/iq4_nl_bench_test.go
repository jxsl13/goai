//go:build darwin && cgo

package metal_test

import (
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

var metalIQ4NLTensorSink *tensor.Tensor

// BenchmarkMetalIQ4NLLeadership compares equal host input/output boundaries: both candidates
// allocate an F32 result and return it to Go. Metal keeps only the compressed weight resident.
func BenchmarkMetalIQ4NLLeadership(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, shape := range []struct{ n, k int }{{512, 1024}, {2048, 2048}, {4096, 1024}} {
		name := "N" + strconv.Itoa(shape.n) + "_K" + strconv.Itoa(shape.k)
		b.Run(name, func(b *testing.B) {
			x, wq := iq4NLInputs(1, shape.n, shape.k)
			rw, err := metal.UploadQWeightIQ4_NL(wq, shape.n, shape.k)
			if err != nil {
				b.Fatal(err)
			}
			defer rw.Close()
			if _, err := rw.QMatMul(x); err != nil {
				b.Fatal(err)
			}
			b.Run("CPU", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					var err error
					metalIQ4NLTensorSink, err = gguf.QMatMul(x, wq, gguf.IQ4_NL, shape.n, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("MetalResidentHostIO", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					var err error
					metalIQ4NLTensorSink, err = rw.QMatMul(x)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

// BenchmarkMetalIQ4NLCooperativeCampaign measures the production resident recorder path. Sixteen
// distinct weights defeat cache-hot repetition; GPU timestamps remove host submit/wake jitter,
// and AB/BA ordering distributes drift across scalar and cooperative arms.
func BenchmarkMetalIQ4NLCooperativeCampaign(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	previous := metal.SetIQ4NLCooperative(false)
	defer metal.SetIQ4NLCooperative(previous)
	for _, shape := range []struct {
		name string
		n, k int
	}{{"kv", 512, 2048}, {"square", 2048, 2048}, {"gate", 5632, 2048}, {"down", 2048, 5632}} {
		b.Run(shape.name, func(b *testing.B) {
			const residentCount = 16
			weights := make([]*metal.ResidentQWeight, 0, residentCount)
			hostWeights := make([][]byte, 0, residentCount)
			for i := range residentCount {
				raw := syntheticIQ4NL(shape.n, shape.k, i)
				rw, err := metal.UploadQWeightIQ4_NL(raw, shape.n, shape.k)
				if err != nil {
					b.Fatal(err)
				}
				hostWeights = append(hostWeights, raw)
				weights = append(weights, rw)
			}
			defer func() {
				for _, rw := range weights {
					rw.Close()
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
				metal.SetIQ4NLCooperative(cooperative)
				start := time.Now()
				r, err := metal.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				for _, rw := range weights {
					if err := r.QMatMulResident(x, rw, o, 1); err != nil {
						r.Free()
						b.Fatal(err)
					}
				}
				if err := r.Commit(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				if err := r.Wait(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
				return sample{
					gpu:  time.Duration(metal.LastGPUSeconds() * float64(time.Second) / residentCount),
					wall: time.Since(start) / residentCount,
				}
			}
			hostX := tensor.New(tensor.F32, tensor.Shape{1, shape.k})
			timedCPU := func() time.Duration {
				start := time.Now()
				for _, raw := range hostWeights {
					var err error
					metalIQ4NLTensorSink, err = gguf.QMatMul(hostX, raw, gguf.IQ4_NL, shape.n, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
				return time.Since(start) / residentCount
			}
			for i := range 8 {
				timed(i%2 == 1)
				timedCPU()
			}
			var scalarGPU, scalarWall, cooperativeGPU, cooperativeWall, cpuWall time.Duration
			b.ResetTimer()
			for i := range b.N {
				if i%2 == 0 {
					scalar, cooperative := timed(false), timed(true)
					scalarGPU += scalar.gpu
					scalarWall += scalar.wall
					cooperativeGPU += cooperative.gpu
					cooperativeWall += cooperative.wall
				} else {
					cooperative, scalar := timed(true), timed(false)
					cooperativeGPU += cooperative.gpu
					cooperativeWall += cooperative.wall
					scalarGPU += scalar.gpu
					scalarWall += scalar.wall
				}
				cpuWall += timedCPU()
			}
			den := float64(b.N)
			controlNS := float64(scalarGPU.Nanoseconds()) / den
			candidateNS := float64(cooperativeGPU.Nanoseconds()) / den
			candidateWallNS := float64(cooperativeWall.Nanoseconds()) / den
			cpuNS := float64(cpuWall.Nanoseconds()) / den
			b.ReportMetric(controlNS, "scalar-gpu-ns/op")
			b.ReportMetric(float64(scalarWall.Nanoseconds())/den, "scalar-wall-ns/op")
			b.ReportMetric(candidateNS, "cooperative-gpu-ns/op")
			b.ReportMetric(candidateWallNS, "cooperative-wall-ns/op")
			b.ReportMetric(cpuNS, "cpu-wall-ns/op")
			b.ReportMetric(controlNS/candidateNS, "cooperative-vs-scalar-x")
			b.ReportMetric(cpuNS/candidateWallNS, "metal-vs-cpu-wall-x")
		})
	}
}
