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

// BenchmarkBatchedDecoderF16Throughput measures the aggregate serving throughput (tokens/sec across the
// batch) of the PUBLIC API's real decode step — Decode, which includes the on-device greedy sampling
// (BatchArgmax) and its per-step token round-trip, i.e. exactly what a serving loop runs. Aggregate
// tok/s should scale ~linearly with batch size B on a bandwidth-bound decode (the weight reads amortize
// across B tokens), confirming BatchedDecoderF16 delivers the ~30× headroom over batch-1 through the
// production interface, not just the raw forward.
func BenchmarkBatchedDecoderF16Throughput(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	f, err := gguf.ReadFile(b4ModelPath)
	if err != nil {
		b.Skipf("gguf: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		b.Fatal(err)
	}
	for _, B := range []int{1, 8, 32} {
		b.Run(b4BenchName(B), func(b *testing.B) {
			dec, err := llamagpu.NewBatchedDecoderF16(m, B, 96+b.N)
			if err != nil {
				b.Skipf("decoder: %v", err)
			}
			defer dec.Close()
			seqs := make([]*llamagpu.BatchedSeq, B)
			for i := range seqs {
				seqs[i] = dec.NewSequence()
			}
			defer func() {
				for _, s := range seqs {
					s.Release()
				}
			}()
			toks := make([]int32, B)
			// Seed 64 tokens of KV per sequence for realistic attention reads.
			for pos := 0; pos < 64; pos++ {
				for i := range toks {
					toks[i] = int32((pos*131 + i*911 + 7) % m.Config.Vocab)
				}
				if _, err := dec.Decode(seqs, toks); err != nil {
					b.Fatal(err)
				}
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := range toks {
					toks[j] = int32((i*577 + j*911 + 3) % m.Config.Vocab)
				}
				if _, err := dec.Decode(seqs, toks); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(B)*float64(b.N)/b.Elapsed().Seconds(), "tok/s-agg")
			b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/step")
		})
	}
}

func b4BenchName(B int) string {
	switch B {
	case 1:
		return "B1"
	case 8:
		return "B8"
	default:
		return "B32"
	}
}
