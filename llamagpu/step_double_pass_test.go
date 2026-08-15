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

// TestStepIssuesOneForwardPass is the correctness anchor for single-token decode, and the record of
// a WRONG conclusion I published about it.
//
// Counting mtl_recorder_mha calls (one per layer per pass, 22 layers) for a 512-token prefill plus
// one Step showed:
//
//	prefill only          22 x (sq=512 sk=512)
//	prefill + one Step    22 x (sq=512 sk=512) + 22 x (sq=1 sk=513) + 22 x (sq=1 sk=514)
//
// I read the 44 as two forward passes per token and reported a 2x redundancy. That was wrong.
//
// mtl_recorder_mha is the RECORDING entry point, not execution. The second group is stepInto's
// encode-overlap (SPEC T614): after committing pos it pre-ENCODES pos+1 into d.pending while the GPU
// is busy, and the next Step consumes that buffer instead of encoding fresh. Two buffers are
// recorded; exactly one is committed per Step. In steady state each Step encodes one and executes
// one, which is the point of the optimisation — it hides encode latency behind GPU execution.
//
// So there is no doubled GPU work, the 0.97x short-context decode figure was never inflated, and the
// long-context gap (0.19x of llama.cpp at ctx=1536) remains UNEXPLAINED — the 4.4x discrepancy
// against the leave-one-out synthetic chain still has no identified cause.
//
// The methodological error is worth keeping: instrumenting a recorder counts what is ENQUEUED, and
// any code that speculatively records work will double that count without doing double work.
// Dispatch counts from a record-mode API are not execution counts. Measuring GPU time, or counting
// at commit, would not have made this mistake.
//
// What this test does: it checks stepwise decode against the batched reference and passes. It never
// detected anything about pass counts and was not able to — it is kept purely as the correctness
// anchor for the position/KV-append handling that encode-overlap depends on.
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
