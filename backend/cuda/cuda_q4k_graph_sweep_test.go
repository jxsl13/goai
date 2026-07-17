//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
)

// The definitive decode scoreboard: Q4_K on the CUDA-GRAPH path (fixed buffers +
// device-pos + on-device argmax) across every eligible model. The earlier sweep numbers
// (96-171 tok/s) were EAGER-path; the serving capstone showed the graph adds a large
// launch-overhead win on top (TinyLlama 199→249). This test produces the current
// authoritative tok/s per model for docs/benchmarking.md, against the recorded
// llama.cpp Vulkan Q8 / Q4_K_M baselines.
func TestCUDAQ4KGraphDecodeSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	skipNoGPU(t)
	for _, tc := range []struct {
		path, arch, label string
	}{
		{"../../models/tinyllama-1.1b-chat-q8_0.gguf", "llama", "TinyLlama-1.1B"},
		{"../../models/qwen2.5-1.5b-instruct-q8_0.gguf", "qwen2", "Qwen2.5-1.5B"},
		{"../../models/qwen2.5-3b-instruct-q8_0.gguf", "qwen2", "Qwen2.5-3B"},
		{"../../models/mistral-7b-instruct-v0.2.Q8_0.gguf", "llama", "Mistral-7B"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("model not present (%s)", tc.path)
			}
			f, err := os.Open(tc.path)
			must(t, err)
			rf, err := gguf.ReadRaw(f)
			f.Close()
			must(t, err)

			const maxSeq = 160
			gd := buildRawGraphDecoder(t, rf, tc.arch, maxSeq, fromF32(quantQ4K))
			defer gd.free()

			// short eager prefill to populate real state, then capture
			const prefill = 8
			var last int
			for p := 0; p < prefill; p++ {
				last = gd.step(t, int32((p*131+7)%1000+1), p)
			}
			runtime.LockOSThread()
			must(t, cuda.CaptureBegin())
			gd.forwardBody(t)
			g, err := cuda.CaptureEnd()
			runtime.UnlockOSThread()
			must(t, err)
			gd.graph = g

			// warm + timed window (tg128-equivalent)
			for w := 0; w < 8; w++ {
				last = gd.stepGraph(t, int32(last), prefill+w)
			}
			const steps = 128
			t0 := time.Now()
			pos := prefill + 8
			for d := 0; d < steps; d++ {
				last = gd.stepGraph(t, int32(last), pos+d)
			}
			tps := float64(steps) / time.Since(t0).Seconds()
			t.Logf("%s Q4_K GRAPH decode: %.1f tok/s", tc.label, tps)
		})
	}
}
