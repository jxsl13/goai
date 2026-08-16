//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"sort"
	"testing"
)

// TestMeasurementNoiseFloor establishes how precisely this machine can time a GPU kernel, so a
// later A/B can be judged against it instead of being read hopefully.
//
// It matters because wall-clock timing is not good enough here. The same Q4_K workload measured by
// wall clock gave 100.5 / 107.2 / 117.2 GB/s across runs — a 17% spread that swamped every effect
// worth chasing, and produced a NON-MONOTONIC dispatch-size curve that suggested a real effect
// where there was none. Timing the command buffer by its own GPU timestamps
// (GPUEndTime-GPUStartTime, via LastGPUSeconds) removes host submit and wake jitter entirely.
//
// With a warmup to a steady clock and 40 samples, measured cv is 1.1-3.0% within a run and warmed
// runs agree to ~0.2% (127.5 vs 127.3 GB/s). The corrected dispatch-size curve is monotonic:
// N=2048 124.0, 2560 132.5, 4096 144.7, 8192 156.1, 11264 162.7, 16384 169.7 GB/s.
//
// Rule this encodes: quantify the noise floor BEFORE interpreting a difference. An effect smaller
// than the spread is not a result.
func TestMeasurementNoiseFloor(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 2048
	nb := K / 256
	per := N * nb * 144
	count := (420 << 20) / per
	type e struct {
		rq   *ResidentQWeight
		x, o *DeviceBuffer
	}
	var ws []e
	raw := make([]byte, per)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	for range count {
		rw, err := Backend{}.UploadQuant(raw, 12, N, K)
		if err != nil {
			t.Skip(err)
		}
		xb, _ := NewDeviceBufferF32(make([]float32, K))
		ob, _ := NewDeviceBufferF32(make([]float32, N))
		ws = append(ws, e{rw.(*ResidentQWeight), xb, ob})
	}
	defer func() {
		for _, w := range ws {
			w.rq.Close()
			w.x.Release()
			w.o.Release()
		}
	}()
	total := float64(per) * float64(count)

	// Warm the GPU to a steady clock before recording anything.
	for range 10 {
		r, _ := NewRecorder()
		for _, w := range ws {
			r.QMatMulResident(w.x, w.rq, w.o, 1)
		}
		r.Commit()
		r.Wait()
		r.Free()
	}

	const samples = 40
	gpu := make([]float64, 0, samples)
	for range samples {
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range ws {
			if err := r.QMatMulResident(w.x, w.rq, w.o, 1); err != nil {
				t.Fatal(err)
			}
		}
		r.Commit()
		r.Wait()
		gpu = append(gpu, LastGPUSeconds())
		r.Free()
	}
	stat := func(name string, v []float64) {
		s := append([]float64(nil), v...)
		sort.Float64s(s)
		med := s[len(s)/2]
		var mean float64
		for _, x := range s {
			mean += x
		}
		mean /= float64(len(s))
		var sd float64
		for _, x := range s {
			sd += (x - mean) * (x - mean)
		}
		sd = math.Sqrt(sd / float64(len(s)))
		fmt.Printf("NOISE %-4s min=%.1f med=%.1f max=%.1f GB/s  cv=%.2f%%  (min/med spread %.1f%%)\n",
			name, total/s[len(s)-1]/1e9, total/med/1e9, total/s[0]/1e9, 100*sd/mean, 100*(s[len(s)-1]-s[0])/med)
	}
	stat("gpu", gpu)
	// Loose: this guards the INSTRUMENT, not the kernel. A cv this large means the machine is
	// too noisy to trust an A/B, whatever the kernel does.
	var mean float64
	for _, x := range gpu {
		mean += x
	}
	mean /= float64(len(gpu))
	var sd float64
	for _, x := range gpu {
		sd += (x - mean) * (x - mean)
	}
	if cv := 100 * math.Sqrt(sd/float64(len(gpu))) / mean; cv > 12 {
		// A DIAGNOSTIC, not a correctness failure. This measures the MACHINE, not the code: it is
		// telling you that a kernel A/B run right now would be unreliable. Under -short (what CI and
		// make preflight-metal use) that must not redden the lane — it failed roughly one run in
		// three on a thermally loaded machine while every other test passed, which makes a gate
		// useless. Full runs still fail so the warning is not lost.
		if testing.Short() {
			t.Logf("GPU-timestamp cv %.1f%% — machine too noisy for a kernel A/B right now "+
				"(not failing under -short: this measures the host, not the code)", cv)
			return
		}
		t.Errorf("GPU-timestamp cv %.1f%% — too noisy to judge a kernel A/B on this machine right now", cv)
	}
}
