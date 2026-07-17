//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDARWKVMatchesReference is the parity anchor for the RWKV-4 GPU decoder — GoAI's third
// recurrent family after the Mamba SSMs. On CUDA it must reproduce the reference nlp.RWKV.Forward
// logits at EVERY prompt position. RWKV has no attention, no KV cache and no positional encoding —
// decode is an O(1) recurrence (per-block token-shift + WKV state), so the GPU path records
// cu_wkv_step for the WKV time-mix alongside token-shift lerps, sigmoid gates and a squared-ReLU
// channel-mix, all on the shared Decoder core. Comparing every position validates the recurrence is
// exact across the whole run; the StepN path (which loops Step for RWKV) is checked too.
func TestCUDARWKVMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/rwkv_hf.safetensors")
	if err != nil {
		t.Skipf("rwkv testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.RWKVFromHF(ts, nlp.RWKVConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("RWKVFromHF: %v", err)
	}

	dec, err := llamagpu.NewRWKVCUDA(m)
	if err != nil {
		t.Fatalf("NewRWKVCUDA: %v", err)
	}
	defer dec.Release()

	prompt := []int{3, 7, 1, 9, 4, 2, 8}
	refT, err := m.Forward(backend.NewContext().WithBackend(backend.Reference()), prompt)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}
	vocab := refT.Shape()[1]

	// Sequential decode: Step carries the token-shift + WKV state (resetting at pos==0); row i must
	// match the reference logits after token i.
	for pos, tok := range prompt {
		got, err := dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("cuda Step pos %d: %v", pos, err)
		}
		if len(got) != vocab {
			t.Fatalf("pos %d: got %d logits, want vocab %d", pos, len(got), vocab)
		}
		for j := range got {
			want := refT.AtF64(pos, j)
			if math.IsNaN(float64(got[j])) || math.Abs(float64(got[j])-want) > 2e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("rwkv Step pos %d logit[%d]: cuda %v vs reference %v", pos, j, got[j], want)
			}
		}
	}

	// StepN prefill (loops Step for RWKV): every row must match too.
	all, err := dec.StepN(prompt, 0)
	if err != nil {
		t.Fatalf("cuda StepN: %v", err)
	}
	if len(all) != len(prompt)*vocab {
		t.Fatalf("StepN returned %d logits, want %d", len(all), len(prompt)*vocab)
	}
	for i := range prompt {
		for j := 0; j < vocab; j++ {
			want := refT.AtF64(i, j)
			got := all[i*vocab+j]
			if math.IsNaN(float64(got)) || math.Abs(float64(got)-want) > 2e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("rwkv StepN row %d logit[%d]: cuda %v vs reference %v", i, j, got, want)
			}
		}
	}
	t.Logf("llamagpu NewRWKVCUDA matches reference RWKV.Forward at all %d positions (Step + StepN); WKV recurrent GPU decoder", len(prompt))
}
