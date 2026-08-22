package nlp_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func readOnlyQuantBytes(shape tensor.Shape, qt gguf.QuantType, seed int) []byte {
	blocks := shape.Numel() / 32
	blockBytes := 18
	if qt == gguf.IQ4_XS {
		blocks = shape.Numel() / 256
		blockBytes = 136
	} else if qt == gguf.Q4_1 {
		blockBytes = 20
	}
	raw := make([]byte, blocks*blockBytes)
	for block := range blocks {
		base := block * blockBytes
		binary.LittleEndian.PutUint16(raw[base:], 0x3000) // f16 d=0.125
		start := 2
		if qt == gguf.IQ4_XS {
			binary.LittleEndian.PutUint16(raw[base:], 0x1000) // small f16 d
			binary.LittleEndian.PutUint16(raw[base+2:], 0xaaaa)
			for i := 4; i < 8; i++ {
				raw[base+i] = 0x11 // every signed 6-bit sub-scale is one
			}
			start = 8
		} else if qt == gguf.Q4_1 {
			binary.LittleEndian.PutUint16(raw[base+2:], 0) // f16 m=0
			start = 4
		}
		for i := start; i < blockBytes; i++ {
			raw[base+i] = byte((block*17 + i*29 + seed) & 0xff)
		}
	}
	return raw
}

// The real raw-GGUF loader admits modern read-only quant formats without eagerly expanding their
// projection weights, including IQ4_XS's 256-value nonlinear super-blocks.
func TestQuantLlamaFromGGUFAdmitsModernIQ4(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 256, Heads: 8, KVHeads: 2, Layers: 1, Hidden: 256, Eps: 1e-5, RopeBase: 10000,
	}, 17)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		qt   gguf.QuantType
	}{{"Q4_1", gguf.Q4_1}, {"IQ4_NL", gguf.IQ4_NL}, {"IQ4_XS", gguf.IQ4_XS}} {
		t.Run(tc.name, func(t *testing.T) {
			qt := tc.qt
			tensors := make(map[string]gguf.QuantTensor, len(rf.Tensors))
			for name, tt := range rf.Tensors {
				if len(tt.Shape) == 2 && name != "token_embd.weight" {
					tt.Data = readOnlyQuantBytes(tt.Shape, qt, len(name))
					tt.GGType = uint32(qt)
				}
				tensors[name] = tt
			}
			loaded, err := nlp.QuantLlamaFromGGUF(rf.Metadata, tensors)
			if err != nil {
				t.Fatal(err)
			}
			defer loaded.Close()
			want := tensors["blk.0.attn_output.weight"]
			if loaded.Blocks[0].Wo.QT != qt {
				t.Fatalf("Wo quant type = %v, want %v", loaded.Blocks[0].Wo.QT, qt)
			}
			if !bytes.Equal(loaded.Blocks[0].Wo.Weight, want.Data) {
				t.Fatal("loader changed compressed attn_output bytes")
			}
			logits, err := loaded.Forward(backend.NewContext(), []int{2, 7, 0, 4})
			if err != nil {
				t.Fatal(err)
			}
			for i, value := range logits.Storage().F32() {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("logit %d is non-finite: %g", i, value)
				}
			}
		})
	}
}

// §V15 / E2E (§T151): a float Llama written as a quantized GGUF (projections Q8_0, norms +
// embedding F32) and loaded back with QuantLlamaFromGGUF — via ReadRaw, keeping the projections
// quantized — reconstructs a QuantLlama whose forward is EXACTLY that of QuantizeLlama on the same
// model (both wrap byte-identical Q8_0 weights; the GGUF loader just avoids the float round-trip),
// and still tracks the float model closely (cosine ≈ 1). This is the real "load a quantized model
// straight onto the GPU" path.
func TestQuantLlamaFromGGUF(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(m)
	// quantize every 2-D projection to Q8_0; keep token_embd + the 1-D norms F32.
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}

	fromGGUF, err := nlp.QuantLlamaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	direct, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}

	toks := []int{2, 7, 0, 4}
	lg, err := fromGGUF.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	// TIGHT ANCHOR: the GGUF-loaded model equals the directly-quantized model exactly (same bytes).
	if !lg.Shape().Equal(ld.Shape()) {
		t.Fatalf("shape %v != %v", lg.Shape(), ld.Shape())
	}
	for i := range lg.Numel() {
		idx := tensor.Unravel(i, lg.Shape())
		if lg.AtF64(idx...) != ld.AtF64(idx...) {
			t.Fatalf("[%v] QuantLlamaFromGGUF %v != QuantizeLlama %v", idx, lg.AtF64(idx...), ld.AtF64(idx...))
		}
	}
	// SANITY: still close to the float model.
	lf, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lf, lg); cos < 0.999 {
		t.Errorf("GGUF-quantized cosine %.5f < 0.999 vs float model", cos)
	}
}

// §T151: a "projection" that is F32 in the GGUF (not a quantized-matmul format) is rejected with
// a clear error rather than failing opaquely at forward.
func TestQuantLlamaFromGGUFRejectsFloatProjection(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(m)
	// quantize everything EXCEPT blk.0.attn_q → it stays F32, which QuantLinear cannot run.
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" && name != "blk.0.attn_q.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf, _ := gguf.ReadRaw(&buf)
	if _, err := nlp.QuantLlamaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantLlamaFromGGUF should reject an F32 attn_q projection")
	}
}

// §R93: a tied LM head (no output.weight) wraps the quantized token_embd as the output projection.
func TestQuantLlamaFromGGUFTiedHead(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 5)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(m)
	delete(ts, "output.weight") // tie the head
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 { // includes token_embd (the tied head must be quantized)
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf, _ := gguf.ReadRaw(&buf)
	q, err := nlp.QuantLlamaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	lg, err := q.Forward(backend.NewContext(), []int{1, 3, 5})
	if err != nil {
		t.Fatal(err)
	}
	if !lg.Shape().Equal(tensor.Shape{3, 12}) {
		t.Errorf("tied-head logits shape %v, want (3, 12)", lg.Shape())
	}
	for i := range lg.Numel() {
		idx := tensor.Unravel(i, lg.Shape())
		if v := lg.AtF64(idx...); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("tied-head logit %v not finite", idx)
		}
	}
}
