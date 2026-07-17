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

// TestCUDAMambaMatchesReference is the parity anchor for GoAI's FIRST non-transformer GPU decoder:
// an nlp.Mamba (selective state-space) run on CUDA must reproduce the reference Mamba.Forward logits
// at EVERY prompt position. Mamba has no attention and no KV cache — decode is a linear-time
// recurrence — so the GPU path records the per-timestep conv/SSM decode kernels (cu_conv1d_step /
// cu_ssm_step) carrying per-block conv-window and SSM state across Step calls. Comparing every
// position (not just the last) validates that the state recurrence is exact across the whole run:
// row i is the prediction after consuming token i, which the sequential Step-loop must match the
// full-sequence reference on. The StepN path (which loops Step internally for Mamba) is checked too.
func TestCUDAMambaMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/mamba_hf.safetensors")
	if err != nil {
		t.Skipf("mamba testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("MambaFromHF: %v", err)
	}

	dec, err := llamagpu.NewMambaCUDA(m)
	if err != nil {
		t.Fatalf("NewMambaCUDA: %v", err)
	}
	defer dec.Release()

	prompt := []int{3, 7, 1, 9, 4, 2, 8}
	refT, err := m.Forward(backend.NewContext().WithBackend(backend.Reference()), prompt)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}
	vocab := refT.Shape()[1]

	// Sequential decode: Step carries the conv/SSM state (resetting it at pos==0). Row i must match
	// the reference logits after token i.
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
				t.Fatalf("mamba Step pos %d logit[%d]: cuda %v vs reference %v", pos, j, got[j], want)
			}
		}
	}

	// StepN prefill (loops Step for Mamba): every row must match the reference too.
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
				t.Fatalf("mamba StepN row %d logit[%d]: cuda %v vs reference %v", i, j, got, want)
			}
		}
	}
	t.Logf("llamagpu NewMambaCUDA matches reference Mamba.Forward at all %d positions (Step + StepN); first non-transformer GPU decoder", len(prompt))
}
