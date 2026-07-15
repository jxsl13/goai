//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

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
