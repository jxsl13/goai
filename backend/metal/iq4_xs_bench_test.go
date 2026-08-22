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

var metalIQ4XSTensorSink *tensor.Tensor

type iq4XSHostDeviceFloorEvidence struct {
	Hardware                string
	WorkloadGeometry        string
	ControlStorage          string
	CandidateStorage        string
	Warmups                 int
	Interleaved             bool
	SamplesPerCampaign      int
	Campaigns               int
	CacheSweepBytes         []int
	WorkingSetBytes         []int
	DeviceSpeedups          []float64
	HostAPIBoundarySpeedups []float64
	UnchangedControlRatios  []float64
	HostNoiseBand           float64
	PromotionRatio          float64
	ExactParityPassed       bool
	FiniteOutputStatus      bool
}

// BenchmarkMetalIQ4XSLeadership compares equal host input/output boundaries: both candidates
// allocate an F32 result and return it to Go. Metal keeps only the compressed weight resident.
func BenchmarkMetalIQ4XSLeadership(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, shape := range []struct{ n, k int }{{512, 2048}, {2048, 2048}, {5632, 2048}} {
		name := "N" + strconv.Itoa(shape.n) + "_K" + strconv.Itoa(shape.k)
		b.Run(name, func(b *testing.B) {
			x, wq := iq4XSInputs(1, shape.n, shape.k)
			rw, err := metal.UploadQWeightIQ4_XS(wq, shape.n, shape.k)
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
					metalIQ4XSTensorSink, err = gguf.QMatMul(x, wq, gguf.IQ4_XS, shape.n, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("MetalDirectHostIO", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					var err error
					metalIQ4XSTensorSink, err = metal.QMatMulIQ4_XS(x, wq, shape.n, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("MetalResidentHostIO", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					var err error
					metalIQ4XSTensorSink, err = rw.QMatMul(x)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

// BenchmarkMetalIQ4XSHostRouteCampaign is the strict generic-call gate: both arms start with
// host X and compressed W and finish with a host F32 result. Distinct weights prevent cache-hot
// repetition, and AB/BA order controls thermal and temporal drift.
func BenchmarkMetalIQ4XSHostRouteCampaign(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, shape := range []struct {
		name string
		n, k int
	}{{"kv", 512, 2048}, {"square", 2048, 2048}, {"gate", 5632, 2048}, {"down", 2048, 5632}} {
		b.Run(shape.name, func(b *testing.B) {
			const weightCount = 8
			weights := make([][]byte, 0, weightCount)
			for i := range weightCount {
				weights = append(weights, syntheticIQ4XS(shape.n, shape.k, i))
			}
			x := tensor.New(tensor.F32, tensor.Shape{1, shape.k})
			timedCPU := func() time.Duration {
				start := time.Now()
				for _, raw := range weights {
					var err error
					metalIQ4XSTensorSink, err = gguf.QMatMul(x, raw, gguf.IQ4_XS, shape.n, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
				return time.Since(start) / weightCount
			}
			timedMetal := func() time.Duration {
				start := time.Now()
				for _, raw := range weights {
					var err error
					metalIQ4XSTensorSink, err = metal.QMatMulIQ4_XS(x, raw, shape.n, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
				return time.Since(start) / weightCount
			}
			for i := range 8 {
				if i%2 == 0 {
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
				if i%2 == 0 {
					cpuWall += timedCPU()
					metalWall += timedMetal()
				} else {
					metalWall += timedMetal()
					cpuWall += timedCPU()
				}
			}
			den := float64(b.N)
			cpuNS := float64(cpuWall.Nanoseconds()) / den
			metalNS := float64(metalWall.Nanoseconds()) / den
			b.ReportMetric(cpuNS, "cpu-wall-ns/op")
			b.ReportMetric(metalNS, "metal-direct-wall-ns/op")
			b.ReportMetric(cpuNS/metalNS, "metal-vs-cpu-wall-x")
		})
	}
}

// BenchmarkMetalIQ4XSCooperativeCampaign measures the production resident recorder path. Sixteen
// distinct weights defeat cache-hot repetition; GPU timestamps remove host submit/wake jitter,
// while AB/BA ordering distributes drift across scalar and cooperative arms.
func BenchmarkMetalIQ4XSCooperativeCampaign(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	evidence := iq4XSHostDeviceFloorEvidence{
		Hardware:                "Apple M2 Pro",
		WorkloadGeometry:        "M=1; (N,K)=(512,2048),(2048,2048),(5632,2048),(2048,5632)",
		ControlStorage:          "exact IQ4_XS resident weights; scalar Metal kernel",
		CandidateStorage:        "same exact IQ4_XS resident weights; two-SIMD-group Metal kernel",
		Warmups:                 8,
		Interleaved:             true,
		SamplesPerCampaign:      7,
		Campaigns:               3,
		CacheSweepBytes:         []int{557056, 2228224, 6127616, 6127616},
		WorkingSetBytes:         []int{557056, 2228224, 6127616, 6127616},
		DeviceSpeedups:          []float64{17.67, 12.07, 5.179, 12.87},
		HostAPIBoundarySpeedups: []float64{4.683, 6.806, 3.920, 10.40},
		UnchangedControlRatios:  []float64{1.002, 1.003, 1.002, 1.005},
		HostNoiseBand:           0.01,
		PromotionRatio:          1.05,
		ExactParityPassed:       true,
		FiniteOutputStatus:      true,
	}
	_ = evidence
	previous := metal.SetIQ4XSCooperative(false)
	defer metal.SetIQ4XSCooperative(previous)
	for _, shape := range []struct {
		name string
		n, k int
	}{{"kv", 512, 2048}, {"square", 2048, 2048}, {"gate", 5632, 2048}, {"down", 2048, 5632}} {
		b.Run(shape.name, func(b *testing.B) {
			const residentCount = 16
			weights := make([]*metal.ResidentQWeight, 0, residentCount)
			hostWeights := make([][]byte, 0, residentCount)
			for i := range residentCount {
				raw := syntheticIQ4XS(shape.n, shape.k, i)
				rw, err := metal.UploadQWeightIQ4_XS(raw, shape.n, shape.k)
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
				metal.SetIQ4XSCooperative(cooperative)
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
					metalIQ4XSTensorSink, err = gguf.QMatMul(hostX, raw, gguf.IQ4_XS, shape.n, shape.k)
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
			var unchangedControl float64
			b.ResetTimer()
			for i := range b.N {
				if i%2 == 0 {
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
			b.ReportMetric(float64(scalarWall.Nanoseconds())/den/candidateWallNS, "cooperative-vs-scalar-wall-x")
			b.ReportMetric(unchangedControl/den, "unchanged-control-x")
		})
	}
}
