//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// synthMamba builds a randomly-initialized Mamba of realistic decode dimensions (the golden fixture is
// DModel=16 — far too small to expose weight bandwidth, which is what Q8 attacks). Only used for the
// *_Large benchmarks: it is NOT checked for correctness (parity uses the real golden weights above), so
// the tied head is an independent random [DModel,Vocab] tensor rather than Embedᵀ.
func synthMamba(dModel, n, dConv, expand, layers, vocab int) *nlp.Mamba {
	dt := tensor.F32
	dtRank := (dModel + 15) / 16
	m := &nlp.Mamba{
		Config: nlp.MambaConfig{DModel: dModel, N: n, DConv: dConv, Expand: expand, DtRank: dtRank, Layers: layers, Vocab: vocab, Eps: 1e-5},
		Embed:  tensor.New(dt, tensor.Shape{vocab, dModel}),
		Norm:   nn.NewRMSNorm(dt, dModel),
		Head:   tensor.New(dt, tensor.Shape{dModel, vocab}),
	}
	nn.XavierUniform(m.Embed, vocab, dModel, 1)
	nn.XavierUniform(m.Head, dModel, vocab, 2)
	for i := 0; i < layers; i++ {
		blk, err := nn.NewMambaBlock(dt, dModel, n, dConv, expand, uint64(100+i*20))
		if err != nil {
			panic(err)
		}
		m.Layers = append(m.Layers, nlp.MambaLayer{Norm: nn.NewRMSNorm(dt, dModel), Mixer: blk})
	}
	return m
}

// loadMambaTD loads the shared Mamba test fixture, skipping if the golden weights aren't present.
func loadMambaTD(t testing.TB) *nlp.Mamba {
	t.Helper()
	ts, _, err := safetensors.LoadFile("../nlp/testdata/mamba_hf.safetensors")
	if err != nil {
		t.Skipf("mamba testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("MambaFromHF: %v", err)
	}
	return m
}

// TestCUDAMambaQ8CloseToF32 validates the Q8_0-quantized Mamba decoder (NewMambaQ8CUDA) against the
// f32 CUDA decoder. Q8 is NOT bit-exact — it trades ~0.4%/weight rounding for a ~4× smaller GEMV on the
// bandwidth-bound recurrent decode — so parity is measured as directional agreement (cosine similarity)
// of the logit vectors at EVERY prompt position, not element equality. The recurrence compounds the
// per-matmul rounding across timesteps, so holding a high cosine at the last position is the real
// guarantee that Q8 preserves the model's predictions, not just the first step.
func TestCUDAMambaQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	m := loadMambaTD(t)

	f32Dec, err := llamagpu.NewMambaCUDA(m)
	if err != nil {
		t.Fatalf("NewMambaCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewMambaQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewMambaQ8CUDA: %v", err)
	}
	defer q8Dec.Release()

	prompt := []int{3, 7, 1, 9, 4, 2, 8}
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
		if len(qRow) != len(fRow) {
			t.Fatalf("pos %d: q8 %d logits vs f32 %d", pos, len(qRow), len(fRow))
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
		cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30)
		if cos < minCos {
			minCos = cos
		}
		// A well-behaved Q8 GEMV keeps the logit direction essentially unchanged. 0.999 leaves margin for
		// the recurrence's compounded rounding while still failing loudly on a wrong-orientation/dequant bug
		// (those collapse cosine far below this).
		if cos < 0.999 {
			t.Fatalf("q8 pos %d: cosine similarity %.5f < 0.999 vs f32 — Q8 decode diverged", pos, cos)
		}
	}
	t.Logf("NewMambaQ8CUDA tracks f32 CUDA decode: min cosine %.6f over %d positions (Q8_0 projections, ~4× less weight bandwidth)", minCos, len(prompt))
}

// TestCUDARWKVQ8CloseToF32 is the RWKV twin of TestCUDAMambaQ8CloseToF32: the Q8_0 time-mix/channel-mix
// GEMVs must track the f32 CUDA RWKV decode directionally (cosine) at every position. RWKV's WKV
// recurrence carries state across Step like Mamba's, so a Q8 dequant/orientation slip compounds and
// collapses the cosine — this gates against that.
func TestCUDARWKVQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/rwkv_hf.safetensors")
	if err != nil {
		t.Skipf("rwkv testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.RWKVFromHF(ts, nlp.RWKVConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("RWKVFromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewRWKVCUDA(m)
	if err != nil {
		t.Fatalf("NewRWKVCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewRWKVQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewRWKVQ8CUDA: %v", err)
	}
	defer q8Dec.Release()

	prompt := []int{3, 7, 1, 9, 4, 2, 8}
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
		cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30)
		if cos < minCos {
			minCos = cos
		}
		if cos < 0.999 {
			t.Fatalf("q8 pos %d: cosine similarity %.5f < 0.999 vs f32 — RWKV Q8 decode diverged", pos, cos)
		}
	}
	t.Logf("NewRWKVQ8CUDA tracks f32 CUDA decode: min cosine %.6f over %d positions", minCos, len(prompt))
}

// TestCUDAMamba2Q8CloseToF32 gates the Mamba-2 Q8 path, whose extra risk is the [out,in]→[in,out]
// pre-transpose before quantization (Mamba/RWKV weights are already [in,out]). A transpose slip would
// quantize the wrong orientation and collapse the cosine, so tracking f32 at every position confirms the
// transposed Q8 GEMV is oriented correctly and the SSD recurrence still composes.
func TestCUDAMamba2Q8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/mamba2_hf.safetensors")
	if err != nil {
		t.Skipf("mamba2 testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatalf("Mamba2FromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewMamba2CUDA(m)
	if err != nil {
		t.Fatalf("NewMamba2CUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewMamba2Q8CUDA(m)
	if err != nil {
		t.Fatalf("NewMamba2Q8CUDA: %v", err)
	}
	defer q8Dec.Release()

	prompt := []int{3, 7, 1, 9, 4, 2, 8}
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
		cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30)
		if cos < minCos {
			minCos = cos
		}
		if cos < 0.999 {
			t.Fatalf("q8 pos %d: cosine similarity %.5f < 0.999 vs f32 — Mamba-2 Q8 decode diverged (transpose?)", pos, cos)
		}
	}
	t.Logf("NewMamba2Q8CUDA tracks f32 CUDA decode: min cosine %.6f over %d positions (transposed Q8 projections)", minCos, len(prompt))
}

// benchMambaStep drives a hot decode loop at a fixed position (state carried, no reset) so the timing
// reflects one steady-state token: the recurrence kernels plus every projection GEMV.
func benchMambaStep(b *testing.B, dec *llamagpu.Decoder) {
	b.Helper()
	if _, err := dec.Step(1, 0); err != nil { // prime state at pos 0
		b.Fatalf("prime Step: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dec.Step(2, 1); err != nil {
			b.Fatalf("Step: %v", err)
		}
	}
}

// BenchmarkMambaDecodeStepF32 / _Q8 quantify the Q8_0 decode win. Mamba decode streams the full f32
// projection weights per token (bandwidth-bound), so the Q8 variant should cut per-step latency toward
// the ~4× the 8-bit weights imply. Run: go test -tags 'cuda cgo' -run x -bench MambaDecodeStep ./llamagpu/
func BenchmarkMambaDecodeStepF32(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	m := loadMambaTD(b)
	dec, err := llamagpu.NewMambaCUDA(m)
	if err != nil {
		b.Fatalf("NewMambaCUDA: %v", err)
	}
	defer dec.Release()
	benchMambaStep(b, dec)
}

func BenchmarkMambaDecodeStepQ8(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	m := loadMambaTD(b)
	dec, err := llamagpu.NewMambaQ8CUDA(m)
	if err != nil {
		b.Fatalf("NewMambaQ8CUDA: %v", err)
	}
	defer dec.Release()
	benchMambaStep(b, dec)
}

// The *Large pair runs a realistic Mamba (DModel=1024, DInner=2048, 8 layers ≈ Mamba-370M geometry) —
// ~200 MB of f32 projection weights streamed per token. Here the decode is genuinely weight-bandwidth-
// bound, so Q8_0 (~50 MB) should show the multiplicative win the tiny fixture cannot.
func BenchmarkMambaDecodeStepF32Large(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewMambaCUDA(synthMamba(1024, 16, 4, 2, 8, 8192))
	if err != nil {
		b.Fatalf("NewMambaCUDA: %v", err)
	}
	defer dec.Release()
	benchMambaStep(b, dec)
}

func BenchmarkMambaDecodeStepQ8Large(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewMambaQ8CUDA(synthMamba(1024, 16, 4, 2, 8, 8192))
	if err != nil {
		b.Fatalf("NewMambaQ8CUDA: %v", err)
	}
	defer dec.Release()
	benchMambaStep(b, dec)
}
