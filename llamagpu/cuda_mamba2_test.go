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

// TestCUDAMamba2MatchesReference is the parity anchor for the Mamba-2 (state-space-duality) GPU
// decoder — the SSD sibling of the Mamba decoder. On CUDA it must reproduce the reference
// nlp.Mamba2.Forward logits at EVERY prompt position. Mamba-2 keeps Mamba's no-attention/no-KV-cache
// linear-time recurrence but restricts A to a scalar per head and shares B/C across a group, so the
// GPU path records cu_ssd_step (scalar-decay, grouped) alongside the shared conv1d/softplus
// primitives + gated RMSNorm, carrying per-block conv/SSD state across Step calls. Comparing every
// position validates the recurrence is exact across the whole run; the StepN path is checked too.
// The golden uses num_heads=4, head_dim=8, n_groups=2, state_size=8 (heads_per_group=2 exercised).
func TestCUDAMamba2MatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/mamba2_hf.safetensors")
	if err != nil {
		t.Skipf("mamba2 testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatalf("Mamba2FromHF: %v", err)
	}

	dec, err := llamagpu.NewMamba2CUDA(m)
	if err != nil {
		t.Fatalf("NewMamba2CUDA: %v", err)
	}
	defer dec.Release()

	prompt := []int{3, 7, 1, 9, 4, 2, 8}
	refT, err := m.Forward(backend.NewContext().WithBackend(backend.Reference()), prompt)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}
	vocab := refT.Shape()[1]

	// Sequential decode: Step carries the conv/SSD state (resetting at pos==0); row i must match the
	// reference logits after token i.
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
				t.Fatalf("mamba2 Step pos %d logit[%d]: cuda %v vs reference %v", pos, j, got[j], want)
			}
		}
	}

	// StepN prefill (loops Step for Mamba-2): every row must match too.
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
				t.Fatalf("mamba2 StepN row %d logit[%d]: cuda %v vs reference %v", i, j, got, want)
			}
		}
	}
	t.Logf("llamagpu NewMamba2CUDA matches reference Mamba2.Forward at all %d positions (Step + StepN); SSD GPU decoder", len(prompt))
}
