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
