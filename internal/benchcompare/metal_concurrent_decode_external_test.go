//go:build darwin && cgo && vulkan

package benchcompare

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestProdMetalConcurrentDecodeCampaign is the same-binary M2 promotion campaign for the
// dependency-tracked concurrent recorder. It alternates arm order, uses command-buffer GPU time
// to isolate host contention, and verifies a raw-bit final-logit digest on every arm.
func TestProdMetalConcurrentDecodeCampaign(t *testing.T) {
	path := os.Getenv("TINYLLAMA_GGUF")
	if path == "" {
		t.Skip("set TINYLLAMA_GGUF to a quantized Llama GGUF")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	raw, err := gguf.ReadRaw(f)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	dec, err := llamagpu.NewQuantF16KV(model)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	pairs := 7
	if value := os.Getenv("GOAI_CONCURRENT_PAIRS"); value != "" {
		pairs, err = strconv.Atoi(value)
		if err != nil || pairs < 3 {
			t.Fatalf("GOAI_CONCURRENT_PAIRS=%q must be an integer >=3", value)
		}
	}
	previous := metal.SetConcurrentDecodeRecorder(false)
	defer metal.SetConcurrentDecodeRecorder(previous)

	type sample struct {
		wall   time.Duration
		gpu    time.Duration
		digest [sha256.Size]byte
	}
	measure := func(concurrent bool) sample {
		metal.SetConcurrentDecodeRecorder(concurrent)
		started := time.Now()
		var gpu time.Duration
		var logits []float32
		for pos := range 64 {
			logits, err = dec.Step(1+(pos*17)%1000, pos)
			if err != nil {
				t.Fatal(err)
			}
			gpu += time.Duration(metal.LastGPUSeconds() * float64(time.Second))
		}
		bytes := make([]byte, 4*len(logits))
		for i, value := range logits {
			binary.LittleEndian.PutUint32(bytes[4*i:], math.Float32bits(value))
		}
		return sample{wall: time.Since(started), gpu: gpu, digest: sha256.Sum256(bytes)}
	}

	// Fill all lazy pipelines and discard one observation of each arm.
	_ = measure(false)
	_ = measure(true)
	gpuRatios := make([]float64, 0, pairs)
	wallRatios := make([]float64, 0, pairs)
	var digest [sha256.Size]byte
	for pair := range pairs {
		var control, candidate sample
		if pair%2 == 0 {
			control, candidate = measure(false), measure(true)
		} else {
			candidate, control = measure(true), measure(false)
		}
		if pair == 0 {
			digest = control.digest
		}
		if control.digest != digest || candidate.digest != digest {
			t.Fatalf("pair %d final-logit digest changed: control=%x candidate=%x want=%x", pair+1, control.digest[:8], candidate.digest[:8], digest[:8])
		}
		gpuRatio := float64(control.gpu) / float64(candidate.gpu)
		wallRatio := float64(control.wall) / float64(candidate.wall)
		gpuRatios = append(gpuRatios, gpuRatio)
		wallRatios = append(wallRatios, wallRatio)
		t.Logf("pair=%d gpu control=%s candidate=%s speedup=%.4fx wall control=%s candidate=%s speedup=%.4fx digest=%x",
			pair+1, control.gpu, candidate.gpu, gpuRatio, control.wall, candidate.wall, wallRatio, digest[:8])
	}
	sort.Float64s(gpuRatios)
	sort.Float64s(wallRatios)
	median := func(values []float64) float64 { return values[len(values)/2] }
	gpuMedian, wallMedian := median(gpuRatios), median(wallRatios)
	gpuSpread := gpuRatios[len(gpuRatios)-1] / gpuRatios[0]
	t.Logf("aggregate pairs=%d GPU median=%.4fx spread=%.4fx wall median=%.4fx digest=%x",
		pairs, gpuMedian, gpuSpread, wallMedian, digest[:8])

	gate, err := strconv.ParseBool(os.Getenv("GOAI_CONCURRENT_GATE"))
	if err != nil && os.Getenv("GOAI_CONCURRENT_GATE") != "" {
		t.Fatalf("GOAI_CONCURRENT_GATE must be a boolean: %v", err)
	}
	if gate && (gpuMedian < 1.03 || gpuSpread > 1.05 || wallMedian < 1.0) {
		t.Fatalf("concurrent recorder missed promotion gate: GPU median %.4fx spread %.4fx wall median %.4fx", gpuMedian, gpuSpread, wallMedian)
	}
	fmt.Printf("GOAI_CONCURRENT pairs=%d gpu_median=%.4fx gpu_spread=%.4fx wall_median=%.4fx digest=%x\n",
		pairs, gpuMedian, gpuSpread, wallMedian, digest[:8])
}
