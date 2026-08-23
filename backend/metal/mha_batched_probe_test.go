//go:build darwin && cgo

package metal

import (
	"math"
	"sort"
	"testing"
	"time"
)

func probeMedian(xs []float64) float64 {
	y := append([]float64(nil), xs...)
	sort.Float64s(y)
	return y[len(y)/2]
}

func TestMetalMPSGraphBatchedAttentionProbe(t *testing.T) {
	if !Available() {
		t.Skip("metal unavailable")
	}
	const batch, seq, dm, heads, dk, kvHeads = 8, 65, 128, 4, 32, 4
	const pairs, repeats = 7, 12
	scale := float32(1 / math.Sqrt(dk))
	q := make([]float32, batch*seq*dm)
	k := make([]float32, batch*seq*dm)
	v := make([]float32, batch*seq*dm)
	for i := range q {
		q[i] = float32(math.Sin(float64(i+1)*0.013)) * 0.2
		k[i] = float32(math.Cos(float64(i+3)*0.017)) * 0.2
		v[i] = float32(math.Sin(float64(i+7)*0.019)) * 0.3
	}
	controlOut := make([]float32, len(q))
	candidateOut := make([]float32, len(q))
	for range 3 {
		if _, err := probeMHAMPSGraphBatch(q, k, v, controlOut, batch, seq, dm, heads, dk, kvHeads, scale, false); err != nil {
			t.Fatal(err)
		}
		if _, err := probeMHAMPSGraphBatch(q, k, v, candidateOut, batch, seq, dm, heads, dk, kvHeads, scale, true); err != nil {
			t.Fatal(err)
		}
	}
	var maxAbs, maxRel float64
	for i, got := range candidateOut {
		want := float64(controlOut[i])
		d := math.Abs(float64(got) - want)
		if d > maxAbs {
			maxAbs = d
		}
		r := d / math.Max(1e-8, math.Abs(want))
		if r > maxRel {
			maxRel = r
		}
	}
	if maxAbs > 2e-5 {
		t.Fatalf("batched attention parity maxAbs=%g maxRel=%g", maxAbs, maxRel)
	}

	type sample struct{ gpu, wall float64 }
	controls := make([]sample, pairs)
	candidates := make([]sample, pairs)
	run := func(batched bool, dst []float32) sample {
		start := time.Now()
		var gpu float64
		for range repeats {
			g, err := probeMHAMPSGraphBatch(q, k, v, dst, batch, seq, dm, heads, dk, kvHeads, scale, batched)
			if err != nil {
				t.Fatal(err)
			}
			gpu += g
		}
		return sample{gpu: gpu / repeats, wall: time.Since(start).Seconds() / repeats}
	}
	for i := range pairs {
		if i%2 == 0 {
			controls[i] = run(false, controlOut)
			candidates[i] = run(true, candidateOut)
		} else {
			candidates[i] = run(true, candidateOut)
			controls[i] = run(false, controlOut)
		}
	}
	gpuSpeedups := make([]float64, pairs)
	wallSpeedups := make([]float64, pairs)
	minGPU, minWall := math.Inf(1), math.Inf(1)
	for i := range pairs {
		gpuSpeedups[i] = controls[i].gpu / candidates[i].gpu
		wallSpeedups[i] = controls[i].wall / candidates[i].wall
		minGPU = math.Min(minGPU, gpuSpeedups[i])
		minWall = math.Min(minWall, wallSpeedups[i])
		t.Logf("pair=%d control_gpu_us=%.3f candidate_gpu_us=%.3f gpu_speedup=%.4fx control_wall_us=%.3f candidate_wall_us=%.3f wall_speedup=%.4fx",
			i+1, controls[i].gpu*1e6, candidates[i].gpu*1e6, gpuSpeedups[i], controls[i].wall*1e6, candidates[i].wall*1e6, wallSpeedups[i])
	}
	t.Logf("summary gpu_median=%.4fx gpu_min=%.4fx wall_median=%.4fx wall_min=%.4fx maxAbs=%g maxRel=%g",
		probeMedian(gpuSpeedups), minGPU, probeMedian(wallSpeedups), minWall, maxAbs, maxRel)
}
