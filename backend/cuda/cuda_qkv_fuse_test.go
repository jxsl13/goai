//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

// TestCUDAQKVFuseTokenParity: the fused-QKV weight (wq|wk|wv → one N=(heads+2·kv)·hd
// Q4_K GEMV, Tw55(b)) must be TOKEN-FOR-TOKEN identical to the three separate GEMVs.
// It is bit-exact by construction — stacking the dequantized rows and requantizing
// once yields Q4_K blocks whose first Nq rows equal encoding wq alone, so each output
// row is computed from identical weight bytes; only the launch is merged. 24 greedy
// TinyLlama-Q4_K tokens, fused vs GOAI_CUDA_QKV_FUSE unset.
func TestCUDAQKVFuseTokenParity(t *testing.T) {
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

	t.Setenv("GOAI_CUDA_QKV_FUSE", "0")
	plain, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))

	t.Setenv("GOAI_CUDA_QKV_FUSE", "1")
	fused, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))

	if len(plain) != len(fused) {
		t.Fatalf("length mismatch: %d vs %d", len(plain), len(fused))
	}
	for i := range plain {
		if plain[i] != fused[i] {
			t.Fatalf("token %d diverged: plain %d, fused %d (full: %v vs %v)", i, plain[i], fused[i], plain, fused)
		}
	}
	t.Logf("QKV fuse: %d/%d tokens identical", len(fused), len(fused))
}

// TestCUDAQKVFuseSpeedAB measures fused-QKV decode throughput vs three separate GEMVs,
// interleaved A,B,A,B across reps (§V22). The floor measurement predicts a gain: the
// GQA k/v proj (N=256) runs at only 17% of peak, and folding it into the N=(heads+2·kv)·hd
// launch lifts those starved rows into the q proj's ~46% regime. Informational (t.Log).
func TestCUDAQKVFuseSpeedAB(t *testing.T) {
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

	ids := []int{1, 450, 7483, 310, 3444, 338}
	const gen, maxSeq, reps = 8, 64, 5

	var fusedSum, plainSum float64
	for r := 0; r < reps; r++ {
		t.Setenv("GOAI_CUDA_QKV_FUSE", "1")
		_, ftps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		t.Setenv("GOAI_CUDA_QKV_FUSE", "0")
		_, ptps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		fusedSum += ftps
		plainSum += ptps
		t.Logf("rep %d: fused %.1f tok/s  separate %.1f tok/s", r, ftps, ptps)
	}
	fused, plain := fusedSum/reps, plainSum/reps
	t.Logf("fused-QKV: fused %.1f vs separate %.1f tok/s (%+.1f%%, mean of %d interleaved reps)",
		fused, plain, 100*(fused-plain)/plain, reps)
}

// TestCUDAQKVFuseTokenParityQ8: the Q8 twin of TestCUDAQKVFuseTokenParity (Tw57). A Q8
// decoder's fused QKV (fuseRowsQ8 via GOAI_CUDA_FUSE_FMT=q8) must be token-for-token identical
// to three separate Q8 GEMVs — bit-exact by construction (per-output-column Q8_0 blocks are
// unchanged by the row-stack). GUARDS the format downgrade: before the format-aware dispatch,
// fusing a Q8 run silently requantized QKV to Q4_K, which would diverge here.
func TestCUDAQKVFuseTokenParityQ8(t *testing.T) {
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

	t.Setenv("GOAI_CUDA_FUSE_FMT", "q8")
	t.Setenv("GOAI_CUDA_QKV_FUSE", "0")
	plain, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ8))

	t.Setenv("GOAI_CUDA_QKV_FUSE", "1")
	fused, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ8))

	if len(plain) != len(fused) {
		t.Fatalf("length mismatch: %d vs %d", len(plain), len(fused))
	}
	for i := range plain {
		if plain[i] != fused[i] {
			t.Fatalf("token %d diverged: plain %d, fused %d (full: %v vs %v)", i, plain[i], fused[i], plain, fused)
		}
	}
	t.Logf("Q8 fused-QKV token-parity OK: %d tokens identical (no Q4_K downgrade)", len(plain))
}

// TestCUDAQKVFuseSpeedABQ8 is the Q8 counterpart of TestCUDAQKVFuseSpeedAB (Tw57). Open
// question the occupancy-cliff law raises: a Q8 k/v proj reads 2× the weight bytes of Q4_K
// at the same N=256, so it is LESS latency-starved → the law (gain ∝ starvation) predicts a
// SMALLER fusion gain than Q4_K's +3.7%. Informational (t.Log), interleaved A,B per rep.
func TestCUDAQKVFuseSpeedABQ8(t *testing.T) {
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

	ids := []int{1, 450, 7483, 310, 3444, 338}
	const gen, maxSeq, reps = 8, 64, 5

	t.Setenv("GOAI_CUDA_FUSE_FMT", "q8")
	var fusedSum, plainSum float64
	for r := 0; r < reps; r++ {
		t.Setenv("GOAI_CUDA_QKV_FUSE", "1")
		_, ftps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ8))
		t.Setenv("GOAI_CUDA_QKV_FUSE", "0")
		_, ptps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ8))
		fusedSum += ftps
		plainSum += ptps
		t.Logf("rep %d: fused %.1f tok/s  separate %.1f tok/s", r, ftps, ptps)
	}
	fused, plain := fusedSum/reps, plainSum/reps
	t.Logf("Q8 fused-QKV: fused %.1f vs separate %.1f tok/s (%+.1f%%, mean of %d interleaved reps)",
		fused, plain, 100*(fused-plain)/plain, reps)
}

// TestCUDAGateUpFuseTokenParity: fused gate+up (ffn_gate|ffn_up → one N=2·hidden Q4_K
// GEMV, then SwiGLU over the halves) must be token-for-token identical to the separate
// gate/up GEMVs — bit-exact by construction (same per-row weight bytes).
func TestCUDAGateUpFuseTokenParity(t *testing.T) {
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

	ids := []int{1, 450, 7483, 310, 3444, 338}
	const gen, maxSeq = 24, 64

	t.Setenv("GOAI_CUDA_GATEUP_FUSE", "0")
	plain, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
	t.Setenv("GOAI_CUDA_GATEUP_FUSE", "1")
	fused, _ := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))

	if len(plain) != len(fused) {
		t.Fatalf("length mismatch: %d vs %d", len(plain), len(fused))
	}
	for i := range plain {
		if plain[i] != fused[i] {
			t.Fatalf("token %d diverged: plain %d, fused %d (full: %v vs %v)", i, plain[i], fused[i], plain, fused)
		}
	}
	t.Logf("gate+up fuse: %d/%d tokens identical", len(fused), len(fused))
}

// TestCUDAGateUpFuseSpeedAB: the occupancy-cliff counterpart to the QKV A/B. gate/up are
// NOT starved (N=5632 = 55% of peak), so the theory predicts a SMALL gain vs the QKV
// win (+3.7%, from lifting the 17%-starved k/v). Confirms fusion pays in proportion to
// how latency-bound the folded shapes are. Informational (t.Log).
func TestCUDAGateUpFuseSpeedAB(t *testing.T) {
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

	ids := []int{1, 450, 7483, 310, 3444, 338}
	const gen, maxSeq, reps = 8, 64, 5

	var fusedSum, plainSum float64
	for r := 0; r < reps; r++ {
		t.Setenv("GOAI_CUDA_GATEUP_FUSE", "1")
		_, ftps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		t.Setenv("GOAI_CUDA_GATEUP_FUSE", "0")
		_, ptps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		fusedSum += ftps
		plainSum += ptps
		t.Logf("rep %d: fused %.1f tok/s  separate %.1f tok/s", r, ftps, ptps)
	}
	fused, plain := fusedSum/reps, plainSum/reps
	t.Logf("fused gate+up: fused %.1f vs separate %.1f tok/s (%+.1f%%, mean of %d interleaved reps)",
		fused, plain, 100*(fused-plain)/plain, reps)
}

// TestCUDAFusionStackSpeedAB measures the full decode weight-fusion stack — QKV AND
// gate+up fused together — vs the fully-separate baseline. The two are independent GEMVs,
// so the combined win should be ≈ additive (~+4.8%); this also verifies they don't
// contend (extra VRAM / occupancy) into a negative interaction. Interleaved A,B,A,B (§V22).
func TestCUDAFusionStackSpeedAB(t *testing.T) {
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

	ids := []int{1, 450, 7483, 310, 3444, 338}
	const gen, maxSeq, reps = 8, 64, 5

	setStack := func(on string) {
		t.Setenv("GOAI_CUDA_QKV_FUSE", on)
		t.Setenv("GOAI_CUDA_GATEUP_FUSE", on)
	}
	var fusedSum, plainSum float64
	for r := 0; r < reps; r++ {
		setStack("1")
		_, ftps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		setStack("0")
		_, ptps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, fromF32(quantQ4K))
		fusedSum += ftps
		plainSum += ptps
		t.Logf("rep %d: stack %.1f tok/s  separate %.1f tok/s", r, ftps, ptps)
	}
	fused, plain := fusedSum/reps, plainSum/reps
	t.Logf("fusion stack (QKV+gate+up): %.1f vs separate %.1f tok/s (%+.1f%%, mean of %d interleaved reps)",
		fused, plain, 100*(fused-plain)/plain, reps)
}
