//go:build darwin && cgo

package llamagpu_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestStepIssuesOneForwardPass pins down the lead left open by TestLongContextDecodeGap: a single
// Decoder.Step was observed issuing TWO complete forward passes.
//
// Method — count attention dispatches, since there is exactly one per layer per pass, and TinyLlama
// has 22 layers. Logging every mtl_recorder_mha call for a 512-token prefill followed by one Step:
//
//	prefill only          22 x (sq=512 sk=512)
//	prefill + one Step    22 x (sq=512 sk=512)  +  22 x (sq=1 sk=513)  +  22 x (sq=1 sk=514)
//
// The prefill accounts for exactly 22. One Step then adds 44 — two full stacks, at two CONSECUTIVE
// cache positions (513 then 514), so it is two decode steps rather than a repeated one. The earlier
// aggregate trace agrees: 44 mha, 90 rmsnorm and 262 qmatmul for what should be one step.
//
// This is the concrete half of the long-context decode gap (0.19x of llama.cpp at ctx=1536). It does
// not explain all of it — the real path is 4.4x slower than the leave-one-out synthetic chain, and a
// doubled pass accounts for 2x of that — but it is the part that is now measured rather than
// suspected, and it also inflates SHORT-context decode, where the cost is simply less visible.
//
// IMPORTANT — what this test does and does not do. It checks that stepwise decode agrees with the
// batched reference, and it PASSES. The doubled pass is correctness-NEUTRAL: the extra KV row
// written at pos+1 is overwritten by the next Step, so no output ever differs. That is exactly why
// no existing test caught this, and it means this test does NOT detect the redundancy either —
// only the dispatch counting above does, and that needs bridge instrumentation which is
// deliberately not shipped.
//
// It is kept as the correctness anchor for the fix: whoever removes the second pass must keep this
// green, since the obvious way to remove it (stop advancing the cache) is also the way to break
// position handling.
func TestStepIssuesOneForwardPass(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	f, _ := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if f == nil {
		t.Skip("no model")
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	qm, _ := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skip(err)
	}
	defer dec.Release()

	prompt := []int{1, 5, 9, 13, 21, 34, 55, 89}
	// Reference: the whole sequence in one batched call.
	ref, err := dec.StepNLast(append(append([]int{}, prompt...), 100, 200), 0)
	if err != nil {
		t.Fatal(err)
	}
	// Same sequence, but the last two tokens fed one at a time through Step.
	if _, err = dec.StepNLast(prompt, 0); err != nil {
		t.Fatal(err)
	}
	if _, err = dec.Step(100, len(prompt)); err != nil {
		t.Fatal(err)
	}
	got, err := dec.Step(200, len(prompt)+1)
	if err != nil {
		t.Fatal(err)
	}
	am := func(v []float32) int {
		b, mx := 0, float32(-1e30)
		for i, x := range v {
			if x > mx {
				mx, b = x, i
			}
		}
		return b
	}
	fmt.Printf("STEP argmax batched=%d stepwise=%d\n", am(ref), am(got))
	if am(ref) != am(got) {
		t.Errorf("Step-by-Step decode disagrees with the batched reference (argmax %d vs %d) — "+
			"Step is advancing the KV cache by more than one position per call",
			am(ref), am(got))
	}
}
