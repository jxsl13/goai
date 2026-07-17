//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// minCosStep drives both decoders over the prompt and returns the minimum per-position cosine similarity
// of the logit vectors. Shared by the dense-arch Q8 parity tests: Q8 is not bit-exact, so a directional
// (cosine) match at every position is the right gate — a wrong-orientation/dequant bug in the shared
// mkLin/mkFused Q8 path collapses it far below the caller's threshold.
func minCosStep(t *testing.T, f32Dec, q8Dec *llamagpu.Decoder, prompt []int) float64 {
	t.Helper()
	minCos := 1.0
	for pos, tok := range prompt {
		fRow, err := f32Dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("f32 Step pos %d: %v", pos, err)
		}
		qRow, err := q8Dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("q8 Step pos %d: %v", pos, err)
		}
		var dot, nf, nq float64
		for j := range fRow {
			f, q := float64(fRow[j]), float64(qRow[j])
			if math.IsNaN(q) || math.IsInf(q, 0) {
				t.Fatalf("q8 pos %d logit[%d] non-finite: %v", pos, j, q)
			}
			dot += f * q
			nf += f * f
			nq += q * q
		}
		if cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30); cos < minCos {
			minCos = cos
		}
	}
	return minCos
}

// TestCUDAStableLMQ8CloseToF32 validates the shared-helper Q8 path (mkLin + mkFused via NewStableLMQ8CUDA)
// on a dedicated-constructor dense arch with partial rotary + LayerNorm-bias — features nlp.QuantLlama
// cannot represent, so StableLM had no quantized decode before. The fused QKV and every projection go Q8;
// LN bias / partial-RoPE stay f32.
func TestCUDAStableLMQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/stablelm_hf.safetensors")
	if err != nil {
		t.Skipf("stablelm testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.StableLMFromHF(ts, nlp.StableLMConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatalf("StableLMFromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewStableLMCUDA(m)
	if err != nil {
		t.Fatalf("NewStableLMCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewStableLMQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewStableLMQ8CUDA: %v", err)
	}
	defer q8Dec.Release()

	minCos := minCosStep(t, f32Dec, q8Dec, []int{3, 7, 1, 9, 4})
	if minCos < 0.999 {
		t.Fatalf("StableLM Q8 min cosine %.5f < 0.999 vs f32", minCos)
	}
	t.Logf("NewStableLMQ8CUDA tracks f32 CUDA decode: min cosine %.6f (shared mkLin/mkFused Q8, partial-rotary + LN-bias)", minCos)
}

// TestCUDAStarCoder2Q8CloseToF32 is the bias-carrying twin: StarCoder2 has projection biases (kept f32,
// added after the Q8 matmul), exercising that the shared Q8 fused-QKV composes with AddBias.
func TestCUDAStarCoder2Q8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Skipf("starcoder2 testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.StarCoder2FromHF(ts, nlp.StarCoder2Config{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("StarCoder2FromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewStarCoder2CUDA(m)
	if err != nil {
		t.Fatalf("NewStarCoder2CUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewStarCoder2Q8CUDA(m)
	if err != nil {
		t.Fatalf("NewStarCoder2Q8CUDA: %v", err)
	}
	defer q8Dec.Release()

	minCos := minCosStep(t, f32Dec, q8Dec, []int{3, 7, 1, 9, 4})
	if minCos < 0.999 {
		t.Fatalf("StarCoder2 Q8 min cosine %.5f < 0.999 vs f32", minCos)
	}
	t.Logf("NewStarCoder2Q8CUDA tracks f32 CUDA decode: min cosine %.6f (shared Q8 fused-QKV composes with f32 projection bias)", minCos)
}

// TestCUDAMixtralQ8CloseToF32 validates the sparse-MoE Q8 path (NewMixtralQ8CUDA). Every expert's
// gate/up/down and the attention projections go resident Q8_0, but the top-k ROUTER stays f32 — so the
// expert SELECTION is identical to the f32 decode and only the selected experts' arithmetic rounds. This
// is the strongest router-carve-out check: had the router been quantized, a flipped top-k pick would
// collapse the cosine. Sparse MoE holds the most weight per token, so it is the biggest Q8 decode win.
func TestCUDAMixtralQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/mixtral_hf.safetensors")
	if err != nil {
		t.Skipf("mixtral testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.MixtralFromHF(ts, nlp.MixtralConfig{Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("MixtralFromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewMixtralCUDA(m)
	if err != nil {
		t.Fatalf("NewMixtralCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewMixtralQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewMixtralQ8CUDA: %v", err)
	}
	defer q8Dec.Release()

	minCos := minCosStep(t, f32Dec, q8Dec, []int{3, 7, 1, 9, 4})
	if minCos < 0.999 {
		t.Fatalf("Mixtral Q8 min cosine %.5f < 0.999 vs f32 — router carve-out may have failed (flipped expert selection)", minCos)
	}
	t.Logf("NewMixtralQ8CUDA tracks f32 CUDA decode: min cosine %.6f (Q8 experts + attention, f32 router preserves top-k selection)", minCos)
}

// TestCUDADeepSeekV2Q8CloseToF32 validates Q8 on the flagship MLA + DeepSeekMoE decoder (the last Q8
// gap). The low-rank MLA projections (q_a/q_b/kv_a/kv_b, o_proj) and the routed + shared expert matrices
// go Q8_0; the latent RMSNorms, decoupled RoPE and the top-k router stay f32. Uses the MoE fixture so the
// router carve-out is exercised — a quantized router would flip expert selection and collapse the cosine.
func TestCUDADeepSeekV2Q8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Skipf("deepseekv2moe testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.DeepSeekV2FromHF(ts, nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		FirstKDense: 1, NRoutedExperts: 4, NSharedExperts: 1,
		TopK: 2, MoEHidden: 32, RoutedScale: 1.0,
	})
	if err != nil {
		t.Fatalf("DeepSeekV2FromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewDeepSeekV2CUDA(m)
	if err != nil {
		t.Fatalf("NewDeepSeekV2CUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewDeepSeekV2Q8CUDA(m)
	if err != nil {
		t.Fatalf("NewDeepSeekV2Q8CUDA: %v", err)
	}
	defer q8Dec.Release()

	minCos := minCosStep(t, f32Dec, q8Dec, []int{3, 7, 1, 9, 4})
	if minCos < 0.999 {
		t.Fatalf("DeepSeek-V2 Q8 min cosine %.5f < 0.999 vs f32 (MLA/expert Q8 or router carve-out failed)", minCos)
	}
	t.Logf("NewDeepSeekV2Q8CUDA tracks f32 CUDA decode: min cosine %.6f (MLA low-rank + MoE experts Q8, f32 router — Q8 catalogue complete)", minCos)
}
