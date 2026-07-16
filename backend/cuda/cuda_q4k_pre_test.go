//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAQ4KPreMatMulParity: the pre-decoded-scale Q4_K GEMV (perf-notes-cuda.md R5) must be
// BIT-EXACT vs the standard Q4_K GEMV — both form the same f32 d·sc / dmin·min products and
// accumulate in the same order; the pre path just moves the scale decode to upload. Same blocks,
// same activation, compared elementwise.
func TestCUDAQ4KPreMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(61))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q4_K) // [N,K] out-major, native block order
	must(t, err)

	std, err := cuda.NewResidentBQ4KFromBlocks(blocks, K, N)
	must(t, err)
	defer std.Free()
	pre, err := cuda.NewResidentBQ4KPreFromBlocks(blocks, K, N)
	must(t, err)
	defer pre.Free()

	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	da, err := cuda.NewDeviceF32(1, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	o1, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer o1.Free()
	o2, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer o2.Free()
	must(t, std.QMatMulInto(da, o1))
	must(t, pre.QMatMulInto(da, o2))
	h1, err := o1.ToHost()
	must(t, err)
	h2, err := o2.ToHost()
	must(t, err)

	var maxAbs, maxMag float64
	for n := 0; n < N; n++ {
		if d := math.Abs(h1.AtF64(0, n) - h2.AtF64(0, n)); d > maxAbs {
			maxAbs = d
		}
		if m := math.Abs(h1.AtF64(0, n)); m > maxMag {
			maxMag = m
		}
	}
	rel := maxAbs / math.Max(maxMag, 1e-9)
	t.Logf("Q4_K-pre vs Q4_K: maxAbs %.3e (maxMag %.3e, rel %.2e)", maxAbs, maxMag, rel)
	if rel > 1e-6 {
		t.Fatalf("Q4_K-pre deviates from standard Q4_K: rel %.3e (maxAbs %.3e)", rel, maxAbs)
	}
}

// benchQ4KPre mirrors benchQ4K for the pre-decoded-scale variant. The GB/s uses the pre layout's
// actual 0.75 B/weight (192/144 × 0.5625) so it is comparable in *latency* (ns/op) to benchQ4K but
// honest about the +33% bytes moved. The A/B question (R5): does dropping the scale-decode ALU beat
// the extra bandwidth on this bandwidth-bound path?
func benchQ4KPre(b *testing.B, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	w := tensor.New(tensor.F32, tensor.Shape{k, n})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q4_K)
	if err != nil {
		b.Fatal(err)
	}
	rq, err := cuda.NewResidentBQ4KPreFromBlocks(blocks, k, n)
	if err != nil {
		b.Fatal(err)
	}
	defer rq.Free()
	da, _ := cuda.NewDeviceF32(1, k)
	defer da.Free()
	da.UploadF32(a)
	out, _ := cuda.NewDeviceF32(1, n)
	defer out.Free()
	rq.QMatMulInto(da, out)
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		rq.QMatMulInto(da, out)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(k)*float64(n)*0.75/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkGemvQ4KPre_2048x256(b *testing.B)  { benchQ4KPre(b, 2048, 256) }  // GQA k/v (small-N)
func BenchmarkGemvQ4KPre_2048(b *testing.B)      { benchQ4KPre(b, 2048, 2048) } // q/o proj
func BenchmarkGemvQ4KPre_2048x5632(b *testing.B) { benchQ4KPre(b, 2048, 5632) } // gate/up
func BenchmarkGemvQ4KPre_5632x2048(b *testing.B) { benchQ4KPre(b, 5632, 2048) } // down

// TestCUDAQ4KPreDecodeAB measures the END-TO-END decode speedup from selectively routing the Q4_K
// projections the pre-decode wins on (k/v + ffn_down, via GOAI_CUDA_Q4KPRE=1) vs all-standard, on a
// real Q4_K_M model. The pre-decode is bit-exact, so the decoded tokens MUST be identical either way
// (asserted). Confirms the microbench's ~-2.5% per-layer projection end-to-end.
func TestCUDAQ4KPreDecodeAB(t *testing.T) {
	skipNoGPU(t)
	const path = "../../models/tinyllama-1.1b-chat-q4_k_m.gguf"
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
	t.Setenv("GOAI_CUDA_QKV_FUSE", "0") // route k/v through q() (unfused) so the pre-routing applies
	t.Setenv("GOAI_CUDA_GATEUP_FUSE", "0")

	var preSum, stdSum float64
	for r := 0; r < reps; r++ {
		t.Setenv("GOAI_CUDA_Q4KPRE", "1")
		preToks, ptps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, quantDirect)
		t.Setenv("GOAI_CUDA_Q4KPRE", "0")
		stdToks, stps := runRawDecode(t, rf, "llama", ids, gen, maxSeq, quantDirect)
		if len(preToks) != len(stdToks) {
			t.Fatalf("length mismatch %d vs %d", len(preToks), len(stdToks))
		}
		for i := range preToks {
			if preToks[i] != stdToks[i] {
				t.Fatalf("rep %d token %d diverged: pre %d std %d (pre is bit-exact → must match)", r, i, preToks[i], stdToks[i])
			}
		}
		preSum += ptps
		stdSum += stps
		t.Logf("rep %d: pre %.1f  std %.1f tok/s", r, ptps, stps)
	}
	pre, std := preSum/reps, stdSum/reps
	t.Logf("Q4_K selective-pre (k/v+down): pre %.1f vs std %.1f tok/s (%+.1f%%, mean of %d reps)",
		pre, std, 100*(pre-std)/std, reps)
}
