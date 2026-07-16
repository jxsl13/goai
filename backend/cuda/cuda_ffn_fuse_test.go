//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

// TestCUDAFFNFuseSpeedAB measures the SwiGLU-epilogue fusion's decode throughput
// vs the 3-op chain, interleaved A,B,A,B across reps so a cold-start outlier can
// never land entirely on one arm (§V22 measurement discipline). The fusion removes
// one SwiGLU launch + a hidden-vector (ffn-wide) round-trip per layer per token;
// on a launch-bound decode that is a per-token, per-layer saving. Reports mean tok/s
// for each arm and the delta — informational (t.Log), never a hard gate, since
// absolute tok/s is host/thermal-sensitive.
func TestCUDAFFNFuseSpeedAB(t *testing.T) {
	skipNoGPU(t)
	const path = "../../models/tinyllama-1.1b-chat-q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present (%s)", path)
	}
	if testing.Short() {
		t.Skip("A/B decode measurement is slow; -short")
	}
	f, err := os.Open(path)
	must(t, err)
	rf, err := gguf.ReadRaw(f)
	f.Close()
	must(t, err)

	ids := []int{1, 450, 7483, 310, 3444, 338} // BOS + "The capital of France is"
	const gen, maxSeq, reps = 8, 64, 5

	var fusedSum, plainSum float64
	for r := 0; r < reps; r++ {
		t.Setenv("GOAI_CUDA_FFN_FUSE", "1")
		_, ftps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		t.Setenv("GOAI_CUDA_FFN_FUSE", "0")
		_, ptps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		fusedSum += ftps
		plainSum += ptps
		t.Logf("rep %d: fused %.1f tok/s  chain %.1f tok/s", r, ftps, ptps)
	}
	fused, plain := fusedSum/reps, plainSum/reps
	t.Logf("SwiGLU-epilogue fusion: fused %.1f vs chain %.1f tok/s (%+.1f%%, mean of %d interleaved reps)",
		fused, plain, 100*(fused-plain)/plain, reps)
}

// The SwiGLU-epilogue fusion must be TOKEN-FOR-TOKEN identical to the 3-op
// chain it replaces: the epilogue computes silu with the exact arithmetic of
// the standalone swiglu kernel (stable two-branch expf sigmoid), so the only
// difference is where the multiply happens — same values, same order, no
// tolerance needed. 24 greedy TinyLlama-Q4_K tokens, fused vs GOAI_CUDA_FFN_FUSE=0.
func TestCUDAFFNFuseTokenParity(t *testing.T) {
	skipNoGPU(t)
	const path = "../../models/tinyllama-1.1b-chat-q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present (%s)", path)
	}
	f, err := os.Open(path)
	must(t, err)
	rf, err := gguf.ReadRaw(f)
	f.Close()
	must(t, err)

	ids := []int{1, 450, 7483, 310, 3444, 338} // BOS + "The capital of France is"
	const gen, maxSeq = 24, 64

	t.Setenv("GOAI_CUDA_FFN_FUSE", "0")
	plain, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))

	t.Setenv("GOAI_CUDA_FFN_FUSE", "1")
	fused, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))

	if len(plain) != len(fused) {
		t.Fatalf("length mismatch: %d vs %d", len(plain), len(fused))
	}
	for i := range plain {
		if plain[i] != fused[i] {
			t.Fatalf("token %d diverged: plain %d, fused %d (full: %v vs %v)", i, plain[i], fused[i], plain, fused)
		}
	}
	t.Logf("FFN fuse: %d/%d tokens identical", len(fused), len(fused))
}
