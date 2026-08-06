//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

const b4ModelPath = "../models/tinyllama-1.1b-chat-q8_0.gguf"

// TestBatchedDecoderF16Ragged exercises the production BatchedDecoderF16 end-to-end (construct from a
// real model, NewSequence, ragged Decode with mid-stream admit, Close) and proves it's per-sequence
// correct: sequence A decoded SOLO produces the identical greedy stream to A decoded in a BATCH where a
// second sequence is admitted mid-stream (A and B then sit at different positions every step). This is
// the public-API twin of backend/cuda's TestBatchedDecodeRaggedPositions.
func TestBatchedDecoderF16Ragged(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	f, err := gguf.ReadFile(b4ModelPath)
	if err != nil {
		t.Skipf("gguf: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewBatchedDecoderF16(m, 4, 32)
	if err != nil {
		t.Skipf("decoder: %v", err)
	}
	defer dec.Close()

	const promptLen, gen, admitDelay = 8, 8, 3
	vocab := m.Config.Vocab
	promptA := make([]int32, promptLen)
	promptB := make([]int32, promptLen)
	for i := range promptA {
		promptA[i] = int32((i*2129 + 41) % vocab)
		promptB[i] = int32((i*1327 + 613) % vocab)
	}

	solo := func(prompt []int32) []int32 {
		seq := dec.NewSequence()
		defer seq.Release()
		out := make([]int32, 0, gen)
		var last int32
		for pos := 0; pos < promptLen+gen; pos++ {
			tok := last
			if pos < promptLen {
				tok = prompt[pos]
			}
			nx, err := dec.Decode([]*llamagpu.BatchedSeq{seq}, []int32{tok})
			if err != nil {
				t.Fatal(err)
			}
			last = nx[0]
			if pos >= promptLen-1 && len(out) < gen {
				out = append(out, last)
			}
		}
		return out
	}
	soloA := solo(promptA)

	seqA := dec.NewSequence()
	defer seqA.Release()
	seqB := dec.NewSequence()
	defer seqB.Release()
	batchedA := make([]int32, 0, gen)
	var lastA, lastB int32
	started := false
	bStep := 0
	for pos := 0; pos < promptLen+gen; pos++ {
		tokA := lastA
		if pos < promptLen {
			tokA = promptA[pos]
		}
		if pos >= admitDelay {
			started = true
		}
		if !started {
			nx, err := dec.Decode([]*llamagpu.BatchedSeq{seqA}, []int32{tokA})
			if err != nil {
				t.Fatal(err)
			}
			lastA = nx[0]
		} else {
			tokB := lastB
			if bStep < promptLen {
				tokB = promptB[bStep]
			}
			nx, err := dec.Decode([]*llamagpu.BatchedSeq{seqA, seqB}, []int32{tokA, tokB})
			if err != nil {
				t.Fatal(err)
			}
			lastA, lastB = nx[0], nx[1]
			bStep++
		}
		if pos >= promptLen-1 && len(batchedA) < gen {
			batchedA = append(batchedA, lastA)
		}
	}

	if len(soloA) != len(batchedA) {
		t.Fatalf("length mismatch solo %d batched %d", len(soloA), len(batchedA))
	}
	for i := range soloA {
		if soloA[i] != batchedA[i] {
			t.Fatalf("BatchedDecoderF16 ragged mismatch at %d: solo=%d batched=%d (solo=%v batched=%v)",
				i, soloA[i], batchedA[i], soloA, batchedA)
		}
	}
	t.Logf("BatchedDecoderF16 ragged parity OK via public API: %v", soloA)
}
