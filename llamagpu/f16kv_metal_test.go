//go:build darwin && cgo

package llamagpu

import (
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

func f16KVTestModel(t *testing.T, cfg nlp.LlamaConfig) *nlp.QuantLlama {
	t.Helper()
	m, err := nlp.NewLlama(cfg, 17)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = qm.Close() })
	return qm
}

func decoderCacheBytes(d *Decoder) int {
	n := 0
	for _, b := range d.blocks {
		n += mb(b.kC).ByteLen() + mb(b.vC).ByteLen()
	}
	return n
}

func f16KVArgmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

func TestQuantF16KVDecoderStoragePathAndQuality(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 128, Ctx: 96, Dim: 256, Heads: 4, KVHeads: 1, Layers: 2,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	qm := f16KVTestModel(t, cfg)
	f32, err := NewQuant(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer f32.Release()
	f16, err := NewQuantF16KV(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer f16.Release()

	if f32.f16KV || !f16.f16KV {
		t.Fatalf("cache mode default=%v opt-in=%v, want false/true", f32.f16KV, f16.f16KV)
	}
	if got, want := decoderCacheBytes(f16), decoderCacheBytes(f32)/2; got != want {
		t.Fatalf("f16 KV cache bytes=%d, want exactly half of f32 (%d)", got, want)
	}

	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = 1 + (i*29)%100
	}
	logits32, err := f32.StepNLast(prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	logits16, err := f16.StepNLast(prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	compare := func(pos int, got, want []float32) {
		t.Helper()
		var diff2, ref2 float64
		for i := range got {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("pos %d logit[%d] is non-finite: %g", pos, i, got[i])
			}
			d := float64(got[i] - want[i])
			diff2 += d * d
			ref2 += float64(want[i]) * float64(want[i])
		}
		nrmse := math.Sqrt(diff2 / math.Max(ref2, 1e-30))
		if nrmse > 2e-3 {
			t.Fatalf("pos %d normalized logit RMSE %.6g exceeds 2e-3", pos, nrmse)
		}
		if a, b := f16KVArgmax(got), f16KVArgmax(want); a != b {
			t.Fatalf("pos %d greedy token changed: f16=%d f32=%d (normalized RMSE %.6g)", pos, a, b, nrmse)
		}
	}
	compare(len(prompt)-1, logits16, logits32)

	// Non-initial StepN is the verification path used by speculative decoding. Unlike prompt
	// prefill, it must read the retained half cache while appending a multi-token window.
	verify := []int{11, 37, 59, 83}
	logits32, err = f32.StepNLast(verify, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	logits16, err = f16.StepNLast(verify, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	compare(len(prompt)+len(verify)-1, logits16, logits32)

	decodeStart := len(prompt) + len(verify)
	for pos := decodeStart; pos < decodeStart+8; pos++ {
		tok := 1 + (pos*31)%100
		logits32, err = f32.Step(tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		logits16, err = f16.Step(tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		compare(pos, logits16, logits32)
	}

	if !metal.RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	_, profile, err := f16.ProfileMetalStep(7, decodeStart+8, 256)
	if err != nil {
		t.Fatal(err)
	}
	foundConvert, foundAttention := false, false
	for _, event := range profile.Events {
		foundConvert = foundConvert || event.Label == "kv.f32_to_f16_pair"
		foundAttention = foundAttention || strings.HasPrefix(event.Label, "mha.f16kv.")
	}
	if !foundConvert || !foundAttention {
		t.Fatalf("end-to-end profile did not prove f16 append+attention: convert=%v attention=%v events=%+v", foundConvert, foundAttention, profile.Events)
	}
}

func TestQuantF16KVRejectsUnsupportedHeadDimension(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 64, Ctx: 16, Dim: 64, Heads: 8, KVHeads: 2, Layers: 1,
		Hidden: 192, Eps: 1e-5, RopeBase: 10000,
	}
	qm := f16KVTestModel(t, cfg)
	dec, err := NewQuantF16KV(qm)
	if dec != nil {
		dec.Release()
	}
	if err == nil || !strings.Contains(err.Error(), "head dimension 64") {
		t.Fatalf("NewQuantF16KV error=%v, want explicit head-dimension rejection", err)
	}
}
