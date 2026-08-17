//go:build darwin && cgo

package benchcompare

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestProdMetalEncoderProfile loads the same real GGUF used by the production leadership matrix
// and attributes one ordinary llamagpu decode step to explicit Metal encoder labels. It is gated
// by TINYLLAMA_GGUF because the 638 MiB model is intentionally not a repository fixture.
func TestProdMetalEncoderProfile(t *testing.T) {
	path := os.Getenv("TINYLLAMA_GGUF")
	if path == "" {
		t.Skip("set TINYLLAMA_GGUF to a quantized Llama GGUF")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	raw, err := gguf.ReadRaw(f)
	if err != nil {
		t.Fatalf("gguf.ReadRaw: %v", err)
	}
	model, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Fatalf("QuantLlamaFromGGUF: %v", err)
	}
	dec, err := llamagpu.NewQuant(model)
	if err != nil {
		t.Fatalf("llamagpu.NewQuant: %v", err)
	}
	defer dec.Release()

	// Warm pipeline caches and establish one KV row. ProfileMetalStep discards Step's pending
	// pre-encode so the trace contains exactly the requested position-1 decode command buffer.
	if _, err := dec.Step(1, 0); err != nil {
		t.Fatalf("warm Step: %v", err)
	}
	logits, profile, err := dec.ProfileMetalStep(2, 1, 512)
	if err != nil {
		t.Fatalf("ProfileMetalStep: %v", err)
	}
	controlLogits, err := dec.Step(2, 1)
	if err != nil {
		t.Fatalf("unprofiled control Step: %v", err)
	}
	if !slices.Equal(logits, controlLogits) {
		t.Fatalf("profiled logits digest=%s differs from unprofiled digest=%s",
			metalFloat32RowsDigest(logits), metalFloat32RowsDigest(controlLogits))
	}
	if len(logits) != model.Config.Vocab {
		t.Fatalf("logits=%d want vocab=%d", len(logits), model.Config.Vocab)
	}
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logit[%d]=%v is not finite", i, value)
		}
	}
	if len(profile.Events) == 0 || profile.TimestampFrequency == 0 {
		t.Fatalf("empty or uncalibrated profile: %+v", profile)
	}
	if profile.OmittedMPS != 0 || profile.OmittedOverflow != 0 || profile.OmittedUnsupported != 0 {
		t.Fatalf("profile has omissions: %+v", profile)
	}

	type aggregate struct {
		label string
		count int
		total time.Duration
	}
	byLabel := make(map[string]*aggregate)
	var total time.Duration
	quantEvents := 0
	for _, event := range profile.Events {
		a := byLabel[event.Label]
		if a == nil {
			a = &aggregate{label: event.Label}
			byLabel[event.Label] = a
		}
		a.count++
		a.total += event.Duration
		total += event.Duration
		if strings.HasPrefix(event.Label, "qmatmul.") {
			quantEvents++
		}
	}
	coverage := float64(total) / float64(profile.CommandDuration)
	if coverage < 0.70 || coverage > 1.05 {
		t.Fatalf("encoder busy-time share %.4f outside [0.70,1.05]: events=%s command=%s",
			coverage, total, profile.CommandDuration)
	}
	spanDelta := profile.CommandDuration - profile.EventSpan
	if spanDelta < 0 {
		spanDelta = -spanDelta
	}
	if spanDelta > profile.CommandDuration/10 {
		t.Fatalf("event span %s differs from command duration %s by more than 10%%",
			profile.EventSpan, profile.CommandDuration)
	}
	if quantEvents == 0 {
		t.Fatal("real quantized decode profile contains no qmatmul events")
	}
	ranked := make([]*aggregate, 0, len(byLabel))
	for _, a := range byLabel {
		ranked = append(ranked, a)
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].total == ranked[j].total {
			return ranked[i].label < ranked[j].label
		}
		return ranked[i].total > ranked[j].total
	})
	t.Logf("METAL_ENCODER_PROFILE model=%s dim=%d hidden=%d layers=%d events=%d frequency_hz=%d event_total=%s event_span=%s command=%s coverage=%.4f logits_sha256=%s",
		path, model.Config.Dim, model.Config.Hidden, model.Config.Layers, len(profile.Events),
		profile.TimestampFrequency, total, profile.EventSpan, profile.CommandDuration, coverage,
		metalFloat32RowsDigest(logits))
	for rank, a := range ranked {
		t.Logf("METAL_ENCODER_PROFILE rank=%d label=%s count=%d total=%s share=%.4f",
			rank+1, a.label, a.count, a.total, float64(a.total)/float64(total))
	}
}

func metalFloat32RowsDigest(rows ...[]float32) string {
	h := sha256.New()
	var encoded [4]byte
	for _, row := range rows {
		for _, value := range row {
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			_, _ = h.Write(encoded[:])
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}
