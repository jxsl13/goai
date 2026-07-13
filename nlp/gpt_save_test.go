package nlp_test

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §T435/§V15: GPT.Safetensors is the exact inverse of FromSafetensors — a model round-trips
// through safetensors.Save/Load bit-identically (same tensor names, shapes, dtypes, values), and
// the reloaded model produces bit-identical logits. This is the checkpoint path for models trained
// with the library's own loop (§T434).
func TestGPTSafetensorsRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gpt.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.GPTConfig{Vocab: 17, Ctx: 8, Dim: 12, Heads: 2, Layers: 2, Eps: 1e-5}
	shape := ts["tok_emb"].Shape()
	cfg.Vocab, cfg.Dim = shape[0], shape[1]
	cfg.Ctx = ts["pos_emb"].Shape()[0]
	layers := 0
	for name := range ts {
		if len(name) > 7 && name[:7] == "blocks." && int(name[7]-'0') >= layers {
			layers = int(name[7]-'0') + 1
		}
	}
	cfg.Layers = layers

	m, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		t.Fatal(err)
	}

	// the export names/values must match the loaded map exactly (live tensors, not copies).
	out := m.Safetensors()
	if len(out) != len(ts) {
		t.Fatalf("exported %d tensors, loaded %d", len(out), len(ts))
	}
	for name, want := range ts {
		got := out[name]
		if got == nil {
			t.Fatalf("export missing %q", name)
		}
		if got != want {
			t.Fatalf("%q: export is not the model's live tensor", name)
		}
	}

	// full serialize → parse → rebuild → identical logits, bit for bit.
	var buf bytes.Buffer
	if err := safetensors.Save(&buf, out, nil); err != nil {
		t.Fatal(err)
	}
	ts2, _, err := safetensors.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	m2, err := nlp.FromSafetensors(cfg, ts2)
	if err != nil {
		t.Fatal(err)
	}

	tokens := make([]int, cfg.Ctx)
	for i := range tokens {
		tokens[i] = (i * 5) % cfg.Vocab
	}
	ctx := backend.NewContext()
	l1, err := m.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := m2.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if !l1.Shape().Equal(l2.Shape()) {
		t.Fatalf("logit shapes differ: %v vs %v", l1.Shape(), l2.Shape())
	}
	for i := range l1.Numel() {
		idx := tensor.Unravel(i, l1.Shape())
		a, b := l1.AtF64(idx...), l2.AtF64(idx...)
		if a != b {
			t.Fatalf("logit[%v]: %.17g != %.17g after round-trip", idx, a, b)
		}
	}
	t.Logf("round-trip: %d tensors, %d logits bit-identical", len(out), l1.Numel())
}
