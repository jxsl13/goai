//go:build cuda && cgo && linux

package cuda

import (
	"math/rand"

	"github.com/jxsl13/goai/tensor"
	"testing"
)

// TestSwiGLUHalvesParity validates the fused-gate|up SwiGLU (SwiGLUHalves) is BIT-EXACT to the
// existing Binary SwiGLU applied to the two halves separately — same GPU kernel formula, so any
// difference would be a layout/indexing bug. Covers rows>1 (batched prefill stride) and rows=1.
func TestSwiGLUHalvesParity(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const rows, hidden = 3, 128
	rng := rand.New(rand.NewSource(5))
	gate := make([]float32, rows*hidden)
	up := make([]float32, rows*hidden)
	gu := make([]float32, rows*2*hidden) // [rows, 2*hidden]: row r = [gate row r | up row r]
	for r := 0; r < rows; r++ {
		for i := 0; i < hidden; i++ {
			g := float32(rng.NormFloat64() * 2)
			u := float32(rng.NormFloat64() * 2)
			gate[r*hidden+i] = g
			up[r*hidden+i] = u
			gu[r*2*hidden+i] = g
			gu[r*2*hidden+hidden+i] = u
		}
	}
	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Free()

	dgu, _ := NewDeviceF32(rows, 2*hidden)
	must(t, dgu.UploadF32(gu))
	defer dgu.Free()
	dout, _ := NewDeviceF32(rows, hidden)
	defer dout.Free()
	must(t, rec.SwiGLUHalves(dgu, dout, rows, hidden))
	must(t, rec.Wait())
	got := make([]float32, rows*hidden)
	dout.DownloadF32(got)

	// reference: Binary SwiGLU on separate gate/up buffers (same swiglu_f32 formula)
	dg, _ := NewDeviceF32(rows, hidden)
	must(t, dg.UploadF32(gate))
	defer dg.Free()
	du, _ := NewDeviceF32(rows, hidden)
	must(t, du.UploadF32(up))
	defer du.Free()
	must(t, rec.Binary(dg, du, dg, recBinarySwiGLU))
	must(t, rec.Wait())
	ref := make([]float32, rows*hidden)
	dg.DownloadF32(ref)

	for i := range got {
		if got[i] != ref[i] {
			t.Fatalf("SwiGLUHalves not bit-exact at %d: fused %g vs separate %g", i, got[i], ref[i])
		}
	}
	t.Logf("SwiGLUHalves bit-exact vs separate Binary SwiGLU across rows=%d hidden=%d", rows, hidden)
}

// benchFFNGateUp measures the fused ffn_gate|ffn_up decode path (one wGU GEMV + one SwiGLUHalves)
// against the separate path (wG GEMV + wU GEMV + Binary SwiGLU) at TinyLlama FFN dims. Fused saves
// one GEMV launch per layer. Batched 64 ops/token (one Wait) — the production recorder pattern.
func benchFFNGateUpDecode(b *testing.B, d, hidden int, fused bool) {
	if !Available() {
		b.Skip("no gpu")
	}
	mkW := func(k, n int) *ResidentBQ8 {
		t := tensor.New(tensor.F32, tensor.Shape{k, n})
		tf := t.Storage().F32()
		for i := range tf {
			tf[i] = float32(rand.NormFloat64() * 0.05)
		}
		w, err := NewResidentBQ8(t)
		if err != nil {
			b.Fatal(err)
		}
		return w
	}
	wGU := mkW(d, 2*hidden)
	wG, wU := mkW(d, hidden), mkW(d, hidden)
	rec, _ := NewRecorder()
	in, _ := NewDeviceF32(1, d)
	gu, _ := NewDeviceF32(1, 2*hidden)
	gate, _ := NewDeviceF32(1, hidden)
	up, _ := NewDeviceF32(1, hidden)
	defer func() {
		wGU.Free()
		wG.Free()
		wU.Free()
		rec.Free()
		in.Free()
		gu.Free()
		gate.Free()
		up.Free()
	}()
	const opsPerToken = 64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < opsPerToken; j++ {
			if fused {
				rec.QMatMulResident(in, wGU, gu, 1)
				rec.SwiGLUHalves(gu, gate, 1, hidden)
			} else {
				rec.QMatMulResident(in, wG, gate, 1)
				rec.QMatMulResident(in, wU, up, 1)
				rec.Binary(gate, up, gate, recBinarySwiGLU)
			}
		}
		if err := rec.Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

// TinyLlama dims: d=2048, hidden=5632
func BenchmarkFFNGateUpDecodeFused(b *testing.B)    { benchFFNGateUpDecode(b, 2048, 5632, true) }
func BenchmarkFFNGateUpDecodeSeparate(b *testing.B) { benchFFNGateUpDecode(b, 2048, 5632, false) }

// benchQ8GemvBW measures the ACHIEVED weight-read bandwidth of the resident-Q8 decode GEMV (m=1)
// at a realistic projection size. Q8 weight bytes = k*n (int8) + k*n/32*4 (per-32-block f32 scale).
// Decode is weight-bandwidth-bound, so this vs the RTX-3060 ~360 GB/s peak tells us whether the
// GoAI-vs-llama.cpp decode gap is GEMV efficiency (BW << peak) or per-token overhead (BW ~ peak).
func benchQ8GemvBW(b *testing.B, k, n int) {
	if !Available() {
		b.Skip("no gpu")
	}
	t := tensor.New(tensor.F32, tensor.Shape{k, n})
	tf := t.Storage().F32()
	for i := range tf {
		tf[i] = float32(rand.NormFloat64() * 0.05)
	}
	w, err := NewResidentBQ8(t)
	if err != nil {
		b.Fatal(err)
	}
	rec, _ := NewRecorder()
	in, _ := NewDeviceF32(1, k)
	out, _ := NewDeviceF32(1, n)
	defer func() { w.Free(); rec.Free(); in.Free(); out.Free() }()
	const opsPerBatch = 64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < opsPerBatch; j++ {
			rec.QMatMulResident(in, w, out, 1)
		}
		rec.Wait()
	}
	b.StopTimer()
	wbytes := float64(k*n) + float64(k*n/32)*4 // int8 weights + f32 block scales
	secsPerOp := b.Elapsed().Seconds() / float64(b.N) / opsPerBatch
	b.ReportMetric(wbytes/secsPerOp/1e9, "GB/s")
}

// ffn_down [5632,2048], gate/up-fused [2048,11264], attn qkv-fused [2048,~2560]
func BenchmarkQ8GemvBW_down(b *testing.B) { benchQ8GemvBW(b, 5632, 2048) }
func BenchmarkQ8GemvBW_gu(b *testing.B)   { benchQ8GemvBW(b, 2048, 11264) }

// TestGeGLUHalvesParity validates the fused GeGLU (GeGLUHalves) is BIT-EXACT to the separate
// Unary-GELU + Binary-mul on the two halves — same GPU GELU (erff) formula. Covers rows>1.
func TestGeGLUHalvesParity(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const rows, hidden = 3, 128
	rng := rand.New(rand.NewSource(6))
	gate := make([]float32, rows*hidden)
	up := make([]float32, rows*hidden)
	gu := make([]float32, rows*2*hidden)
	for r := 0; r < rows; r++ {
		for i := 0; i < hidden; i++ {
			g := float32(rng.NormFloat64() * 2)
			u := float32(rng.NormFloat64() * 2)
			gate[r*hidden+i], up[r*hidden+i] = g, u
			gu[r*2*hidden+i], gu[r*2*hidden+hidden+i] = g, u
		}
	}
	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Free()
	dgu, _ := NewDeviceF32(rows, 2*hidden)
	must(t, dgu.UploadF32(gu))
	defer dgu.Free()
	dout, _ := NewDeviceF32(rows, hidden)
	defer dout.Free()
	must(t, rec.GeGLUHalves(dgu, dout, rows, hidden))
	must(t, rec.Wait())
	got := make([]float32, rows*hidden)
	dout.DownloadF32(got)

	dg, _ := NewDeviceF32(rows, hidden)
	must(t, dg.UploadF32(gate))
	defer dg.Free()
	du, _ := NewDeviceF32(rows, hidden)
	must(t, du.UploadF32(up))
	defer du.Free()
	must(t, rec.Unary(dg, dg, recUnaryGELU))
	must(t, rec.Binary(dg, du, dg, recBinaryMul))
	must(t, rec.Wait())
	ref := make([]float32, rows*hidden)
	dg.DownloadF32(ref)

	for i := range got {
		if got[i] != ref[i] {
			t.Fatalf("GeGLUHalves not bit-exact at %d: fused %g vs separate %g", i, got[i], ref[i])
		}
	}
	t.Logf("GeGLUHalves bit-exact vs separate GELU+mul across rows=%d hidden=%d", rows, hidden)
}

// benchFFNGeGLUDecode: fused GeGLU (1 GEMV + GeGLUHalves) vs separate (2 GEMV + Unary GELU + Binary
// mul) at Gemma FFN dims. GeGLU fuses 4 ops→2 (vs SwiGLU's 3→2), so a larger relative saving.
func benchFFNGeGLUDecode(b *testing.B, d, hidden int, fused bool) {
	if !Available() {
		b.Skip("no gpu")
	}
	mkW := func(k, n int) *ResidentBQ8 {
		t := tensor.New(tensor.F32, tensor.Shape{k, n})
		tf := t.Storage().F32()
		for i := range tf {
			tf[i] = float32(rand.NormFloat64() * 0.05)
		}
		w, err := NewResidentBQ8(t)
		if err != nil {
			b.Fatal(err)
		}
		return w
	}
	wGU := mkW(d, 2*hidden)
	wG, wU := mkW(d, hidden), mkW(d, hidden)
	rec, _ := NewRecorder()
	in, _ := NewDeviceF32(1, d)
	gu, _ := NewDeviceF32(1, 2*hidden)
	gate, _ := NewDeviceF32(1, hidden)
	up, _ := NewDeviceF32(1, hidden)
	defer func() {
		wGU.Free()
		wG.Free()
		wU.Free()
		rec.Free()
		in.Free()
		gu.Free()
		gate.Free()
		up.Free()
	}()
	const opsPerToken = 64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < opsPerToken; j++ {
			if fused {
				rec.QMatMulResident(in, wGU, gu, 1)
				rec.GeGLUHalves(gu, gate, 1, hidden)
			} else {
				rec.QMatMulResident(in, wG, gate, 1)
				rec.QMatMulResident(in, wU, up, 1)
				rec.Unary(gate, gate, recUnaryGELU)
				rec.Binary(gate, up, gate, recBinaryMul)
			}
		}
		if err := rec.Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

// Gemma-2B FFN dims: d=2048, hidden=16384
func BenchmarkFFNGeGLUDecodeFused(b *testing.B)    { benchFFNGeGLUDecode(b, 2048, 16384, true) }
func BenchmarkFFNGeGLUDecodeSeparate(b *testing.B) { benchFFNGeGLUDecode(b, 2048, 16384, false) }
