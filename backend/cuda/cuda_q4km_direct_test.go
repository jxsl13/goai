//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// quantDirect keeps a Q4_K_M file's tensors in their NATIVE encodings: Q4_K and Q6_K
// blocks upload AS-IS (gguf's [out,in] row-major block order is exactly the GEMV
// layout — zero requantization, inheriting llama.cpp's iterative-fit encoder quality;
// Q6_K covers the mix's minority tensors: v/down in half the layers + the output
// head). Anything else dequantizes into resident Q8. Native Q6_K (0.82 B/w, Tw54)
// replaced the earlier dequant→Q8 detour (1.06 B/w, requant loss) — fewer bytes AND
// bit-native.
func quantDirect(qt gguf.QuantTensor) (qProj, error) {
	n, k := qt.Shape[0], qt.Shape[1]
	switch qt.GGType {
	case 12: // Q4_K
		return cuda.NewResidentBQ4KFromBlocks(qt.Data, k, n)
	case 13: // Q5_K — bulk tensors of a Q5_K_M/Q5_K_S mix (Tw66), native like Q4_K/Q6_K
		return cuda.NewResidentBQ5KFromBlocks(qt.Data, k, n)
	case 14: // Q6_K
		if os.Getenv("GOAI_CUDA_Q6K") == "q8" { // A/B: the old dequant→Q8 detour
			break
		}
		return cuda.NewResidentBQ6KFromBlocks(qt.Data, k, n)
	}
	w, err := qt.Dequantize()
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ8(transpose2D(w))
}

// Loading llama.cpp's own Q4_K_M files directly (native Q4_K blocks + Q8 for the Q6_K
// minority) should close the encoder-quality gap of the simple Go Q4_K encoder — the
// requantized path agreed with Q8 on only 2/24 (3B) and 7/24 (1.5B) greedy tokens.
// Gate: ≥16/24 agreement with the Q8-from-Q8-file reference OR the literal answer
// (agreement is the primary signal — a single diverged token can be the answer word
// itself, as the first Mistral run showed: 23/24 with exactly "Paris" swapped).
func TestCUDAQ4KMDirect(t *testing.T) {
	skipNoGPU(t)
	for _, tc := range []struct {
		label, arch, q8Path, kmPath string
	}{
		{"Mistral-7B", "llama", mistral7BPath, "../../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"},
		{"Qwen2.5-1.5B", "qwen2", "../../models/qwen2.5-1.5b-instruct-q8_0.gguf", "../../models/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
		{"Qwen2.5-3B", "qwen2", "../../models/qwen2.5-3b-instruct-q8_0.gguf", "../../models/qwen2.5-3b-instruct-q4_k_m.gguf"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			if _, err := os.Stat(tc.kmPath); err != nil {
				t.Skipf("model not present (%s)", tc.kmPath)
			}
			if _, err := os.Stat(tc.q8Path); err != nil {
				t.Skipf("model not present (%s)", tc.q8Path)
			}

			// Q8 reference tokens from the Q8_0 file.
			f, err := os.Open(tc.q8Path)
			must(t, err)
			rfQ8, err := gguf.ReadRaw(f)
			f.Close()
			must(t, err)
			var ids []int
			var decode func([]int) string
			if tc.arch == "llama" {
				tok, err := nlp.SPMFromGGUF(rfQ8.Metadata)
				must(t, err)
				ids = append([]int{1}, tok.Encode("The capital of France is")...)
				decode = tok.Decode
			} else {
				tok, err := nlp.BPEFromGGUF(rfQ8.Metadata)
				must(t, err)
				ids = tok.Encode("The capital of France is")
				decode = tok.Decode
			}
			const gen = 24
			maxSeq := len(ids) + gen + 40
			q8 := fromF32(func(w *tensor.Tensor) (qProj, error) { return cuda.NewResidentBQ8(w) })
			q8toks, _ := runRawDecode(t, rfQ8, tc.arch, ids, gen, maxSeq, q8)
			rfQ8 = nil // release the raw file before loading the next

			// Direct native-mix decode from the Q4_K_M file.
			f, err = os.Open(tc.kmPath)
			must(t, err)
			rfKM, err := gguf.ReadRaw(f)
			f.Close()
			must(t, err)
			kmToks, kmTps := runRawDecode(t, rfKM, tc.arch, ids, gen, maxSeq, quantDirect)

			match := 0
			for i := range q8toks {
				if kmToks[i] == q8toks[i] {
					match++
				}
			}
			kmText := strings.TrimSpace(decode(kmToks))
			t.Logf("%s Q4_K_M-direct text: %q", tc.label, kmText)
			t.Logf("%s Q4_K_M-direct vs Q8: %d/%d tokens agree; DECODE %.1f tok/s", tc.label, match, gen, kmTps)

			if match < 16 && !strings.Contains(strings.ToLower(kmText), "paris") {
				t.Errorf("%s Q4_K_M-direct quality gate failed: %d/24 agreement and no literal answer: %q",
					tc.label, match, kmText)
			}
		})
	}
}
