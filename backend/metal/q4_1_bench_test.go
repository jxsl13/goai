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

var metalQ41TensorSink *tensor.Tensor

func syntheticQ41(n, k, seed int) []byte {
	blocks := n * (k / 32)
	raw := make([]byte, blocks*20)
	for block := range blocks {
		base := block * 20
		raw[base], raw[base+1] = 0, 0x3c   // f16 d=1
		raw[base+2], raw[base+3] = 0, 0x38 // f16 m=0.5
		for i := 4; i < 20; i++ {
			raw[base+i] = byte((block*17 + i*29 + seed) & 0xff)
		}
	}
	return raw
}

// BenchmarkMetalQ4_1Leadership compares equal host input/output boundaries: both candidates
// allocate an F32 result and return it to Go. The Metal candidate keeps only the compressed weight
// resident, which is the normal decode-loop lifetime.
func BenchmarkMetalQ4_1Leadership(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, shape := range []struct{ n, k int }{{512, 1024}, {2048, 2048}, {4096, 1024}} {
		name := "N" + strconv.Itoa(shape.n) + "_K" + strconv.Itoa(shape.k)
		b.Run(name, func(b *testing.B) {
			x, wq := q41Inputs(1, shape.n, shape.k)
			rw, err := metal.UploadQWeightQ4_1(wq, shape.n, shape.k)
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
					metalQ41TensorSink, err = gguf.QMatMul(x, wq, gguf.Q4_1, shape.n, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("MetalResidentHostIO", func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					var err error
					metalQ41TensorSink, err = rw.QMatMul(x)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

// BenchmarkMetalQ4_1Cooperative isolates the production resident recorder path. Both arms keep
// activation, compressed weight, and output resident and differ only in scalar versus SIMD-group
// pipeline selection.
func BenchmarkMetalQ4_1Cooperative(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const n, k = 2048, 2048
	x, wq := q41Inputs(1, n, k)
	rw, err := metal.UploadQWeightQ4_1(wq, n, k)
	if err != nil {
		b.Fatal(err)
	}
	defer rw.Close()
	xb, err := metal.NewDeviceBufferF32(x.Storage().F32())
	if err != nil {
		b.Fatal(err)
	}
	defer xb.Release()
	ob, err := metal.NewDeviceBufferF32(make([]float32, n))
	if err != nil {
		b.Fatal(err)
	}
	defer ob.Release()
	previous := metal.SetQ4_1Cooperative(false)
	defer metal.SetQ4_1Cooperative(previous)
	run := func(b *testing.B) {
		recorder, err := metal.NewRecorder()
		if err != nil {
			b.Fatal(err)
		}
		if err := recorder.QMatMulResident(xb, rw, ob, 1); err != nil {
			recorder.Free()
			b.Fatal(err)
		}
		if err := recorder.Finish(); err != nil {
			recorder.Free()
			b.Fatal(err)
		}
		recorder.Free()
	}
	for _, tc := range []struct {
		name        string
		cooperative bool
	}{{"Scalar", false}, {"Cooperative", true}} {
		b.Run(tc.name, func(b *testing.B) {
			metal.SetQ4_1Cooperative(tc.cooperative)
			for range 20 {
				run(b)
			}
			b.SetBytes(int64(len(wq)))
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				run(b)
			}
		})
	}
}

// BenchmarkMetalQ4_1CooperativeCampaign measures control and candidate in the same command-stream
// shape used by a decoder. Sixteen distinct resident weights defeat cache-hot repetition; GPU
// timestamps remove host submit/wake jitter, and AB/BA ordering distributes drift across arms.
func BenchmarkMetalQ4_1CooperativeCampaign(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	previous := metal.SetQ4_1Cooperative(false)
	defer metal.SetQ4_1Cooperative(previous)
	for _, shape := range []struct {
		name string
		n, k int
	}{{"kv", 512, 2048}, {"square", 2048, 2048}, {"gate", 5632, 2048}, {"down", 2048, 5632}} {
		b.Run(shape.name, func(b *testing.B) {
			const residentCount = 16
			weights := make([]*metal.ResidentQWeight, 0, residentCount)
			hostWeights := make([][]byte, 0, residentCount)
			for i := range residentCount {
				raw := syntheticQ41(shape.n, shape.k, i)
				rw, err := metal.UploadQWeightQ4_1(raw, shape.n, shape.k)
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
				metal.SetQ4_1Cooperative(cooperative)
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
					gpu:  time.Duration(metal.LastGPUSeconds() * float64(time.Second) / float64(residentCount)),
					wall: time.Since(start) / residentCount,
				}
			}
			hostX := tensor.New(tensor.F32, tensor.Shape{1, shape.k})
			timedCPU := func() time.Duration {
				start := time.Now()
				for _, raw := range hostWeights {
					var err error
					metalQ41TensorSink, err = gguf.QMatMul(hostX, raw, gguf.Q4_1, shape.n, shape.k)
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
					scalar := timed(false)
					cooperative := timed(true)
					scalarGPU += scalar.gpu
					scalarWall += scalar.wall
					cooperativeGPU += cooperative.gpu
					cooperativeWall += cooperative.wall
				} else {
					cooperative := timed(true)
					scalar := timed(false)
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
