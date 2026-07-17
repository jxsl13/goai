//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// gpuStepper is the subset of *llamagpu.Decoder the recurrent-Generate check needs.
type gpuStepper interface {
	Step(token, pos int) ([]float32, error)
	Generate(prompt []int, maxNew int, s nlp.TokenSampler) ([]int, error)
	Release()
}

// checkRecurrentGenerate validates that Decoder.Generate works end-to-end for a recurrent decoder
// (Mamba/Mamba-2/RWKV): its greedy output must equal a manual greedy Step-loop on an independent
// decoder instance. Generate prefills with StepN (which loops Step for the recurrent families) then
// decodes with Step; the manual reference prefills+decodes with Step alone. Both are GPU, so the
// logits are identical — any divergence would be a bug in Generate's orchestration of the recurrent
// state (prefill/decode boundary, position tracking, reset-at-pos-0). newDec builds a fresh decoder
// (independent recurrent state) each call.
func checkRecurrentGenerate(t *testing.T, newDec func() (gpuStepper, error), prompt []int, maxNew int) {
	t.Helper()
	argmax := func(v []float32) int {
		bi := 0
		for i, x := range v {
			if x > v[bi] {
				bi = i
			}
		}
		return bi
	}

	// Reference: manual greedy Step-loop on a fresh decoder (state resets at pos 0).
	ref, err := newDec()
	if err != nil {
		t.Fatalf("build reference decoder: %v", err)
	}
	defer ref.Release()
	want := append([]int(nil), prompt...)
	var logits []float32
	for i, tok := range prompt {
		if logits, err = ref.Step(tok, i); err != nil {
			t.Fatalf("ref Step prefill pos %d: %v", i, err)
		}
	}
	for pos := len(prompt); pos < len(prompt)+maxNew; pos++ {
		next := argmax(logits)
		want = append(want, next)
		if logits, err = ref.Step(next, pos); err != nil {
			t.Fatalf("ref Step decode pos %d: %v", pos, err)
		}
	}

	// Generate on a fresh decoder must match token-for-token.
	gen, err := newDec()
	if err != nil {
		t.Fatalf("build generate decoder: %v", err)
	}
	defer gen.Release()
	got, err := gen.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("Generate produced %d tokens, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: Generate %d vs manual Step-loop %d (%v vs %v)", i, got[i], want[i], got, want)
		}
	}
	t.Logf("Decoder.Generate matches manual greedy Step-loop across %d prompt + %d generated tokens: %v", len(prompt), maxNew, got)
}

// checkRecurrentReuse validates that a recurrent decoder RESETS its state between sequences: running
// Generate twice on the same decoder (a served, reused instance) must give — for the second prompt —
// the same tokens as Generate on a FRESH decoder. Generate prefills with StepN(_, 0), which resets the
// conv/SSM/WKV state at pos 0; a state leak from the first sequence into the second would corrupt the
// output. This is the correctness property that lets one decoder serve many independent requests.
func checkRecurrentReuse(t *testing.T, newDec func() (gpuStepper, error), promptA, promptB []int, maxNew int) {
	t.Helper()
	reused, err := newDec()
	if err != nil {
		t.Fatalf("build reused decoder: %v", err)
	}
	defer reused.Release()
	if _, err := reused.Generate(promptA, maxNew, nlp.Greedy()); err != nil {
		t.Fatalf("first Generate (promptA): %v", err)
	}
	gotB, err := reused.Generate(promptB, maxNew, nlp.Greedy()) // must be independent of promptA
	if err != nil {
		t.Fatalf("second Generate (promptB) on reused decoder: %v", err)
	}

	fresh, err := newDec()
	if err != nil {
		t.Fatalf("build fresh decoder: %v", err)
	}
	defer fresh.Release()
	wantB, err := fresh.Generate(promptB, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate (promptB) on fresh decoder: %v", err)
	}
	if len(gotB) != len(wantB) {
		t.Fatalf("reused %d tokens vs fresh %d (%v vs %v)", len(gotB), len(wantB), gotB, wantB)
	}
	for i := range wantB {
		if gotB[i] != wantB[i] {
			t.Fatalf("state leaked across sequences: token %d reused %d vs fresh %d (%v vs %v)", i, gotB[i], wantB[i], gotB, wantB)
		}
	}
	t.Logf("recurrent state resets on reuse: 2nd Generate on a reused decoder == Generate on a fresh one (%v)", gotB)
}

// TestCUDAMambaGenerate exercises end-to-end greedy Generate for the Mamba (selective-scan) decoder —
// the prefill(StepN→Step-loop)+decode(Step) generation path over the recurrent conv/SSM state.
func TestCUDAMambaGenerate(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/mamba_hf.safetensors")
	if err != nil {
		t.Skipf("mamba testdata unavailable: %v", err)
	}
	m, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("MambaFromHF: %v", err)
	}
	nd := func() (gpuStepper, error) { return llamagpu.NewMambaCUDA(m) }
	checkRecurrentGenerate(t, nd, []int{3, 7, 1, 9}, 5)
	checkRecurrentReuse(t, nd, []int{5, 8, 2}, []int{3, 7, 1, 9}, 5)
}

// TestCUDAMamba2Generate exercises end-to-end greedy Generate for the Mamba-2 (SSD) decoder.
func TestCUDAMamba2Generate(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/mamba2_hf.safetensors")
	if err != nil {
		t.Skipf("mamba2 testdata unavailable: %v", err)
	}
	m, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatalf("Mamba2FromHF: %v", err)
	}
	nd := func() (gpuStepper, error) { return llamagpu.NewMamba2CUDA(m) }
	checkRecurrentGenerate(t, nd, []int{3, 7, 1, 9}, 5)
	checkRecurrentReuse(t, nd, []int{5, 8, 2}, []int{3, 7, 1, 9}, 5)
}

// TestCUDARWKVGenerate exercises end-to-end greedy Generate for the RWKV-4 decoder — the token-shift +
// WKV recurrent state carried across the prefill/decode boundary.
func TestCUDARWKVGenerate(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/rwkv_hf.safetensors")
	if err != nil {
		t.Skipf("rwkv testdata unavailable: %v", err)
	}
	m, err := nlp.RWKVFromHF(ts, nlp.RWKVConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("RWKVFromHF: %v", err)
	}
	nd := func() (gpuStepper, error) { return llamagpu.NewRWKVCUDA(m) }
	checkRecurrentGenerate(t, nd, []int{3, 7, 1, 9}, 5)
	checkRecurrentReuse(t, nd, []int{5, 8, 2}, []int{3, 7, 1, 9}, 5)
}
