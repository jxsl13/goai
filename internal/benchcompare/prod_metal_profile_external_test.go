//go:build darwin && cgo

package benchcompare

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

type prodMetalStageProfile struct {
	Version            int                          `json:"version"`
	LogitsSHA256       string                       `json:"logits_sha256"`
	Dim                int                          `json:"dim"`
	Hidden             int                          `json:"hidden"`
	Layers             int                          `json:"layers"`
	CommandDurationNS  int64                        `json:"command_duration_ns"`
	EventSpanNS        int64                        `json:"event_span_ns"`
	OmittedMPS         int                          `json:"omitted_mps"`
	OmittedOverflow    int                          `json:"omitted_overflow"`
	OmittedUnsupported int                          `json:"omitted_unsupported"`
	Events             []prodMetalStageProfileEvent `json:"events"`
}

type prodMetalStageProfileEvent struct {
	Label         string `json:"label"`
	StartOffsetNS int64  `json:"start_offset_ns"`
	DurationNS    int64  `json:"duration_ns"`
	Ticks         uint64 `json:"ticks"`
}

func loadProfileQuantLlama(t testing.TB) *nlp.QuantLlama {
	t.Helper()
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
	return model
}

// BenchmarkProdMetalProfiledDecodeGGUF is the stage-correlation target for metalcounters. Model
// loading, resident upload, pipeline compilation, and the position-0 warm step are outside the
// timed region; the timed region contains exactly one profiled position-1 command buffer.
func BenchmarkProdMetalProfiledDecodeGGUF(b *testing.B) {
	path := os.Getenv("GOAI_METAL_STAGE_PROFILE")
	if path == "" {
		b.Skip("set GOAI_METAL_STAGE_PROFILE to a temporary JSON sidecar path")
	}
	if b.N != 1 {
		b.Fatalf("stage profile requires exactly one fixed iteration, got %d", b.N)
	}
	model := loadProfileQuantLlama(b)
	dec, err := llamagpu.NewQuant(model)
	if err != nil {
		b.Fatalf("llamagpu.NewQuant: %v", err)
	}
	defer dec.Release()
	warmLogits, err := dec.Step(1, 0)
	if err != nil {
		b.Fatalf("warm Step: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	logits, profile, err := dec.ProfileMetalStep(2, 1, 512)
	b.StopTimer()
	if err != nil {
		b.Fatalf("ProfileMetalStep: %v", err)
	}
	if profile.OmittedMPS != 0 || profile.OmittedOverflow != 0 || profile.OmittedUnsupported != 0 {
		b.Fatalf("profile omissions: %+v", profile)
	}
	if len(profile.Events) == 0 || profile.EventSpan <= 0 || profile.CommandDuration <= 0 {
		b.Fatalf("incomplete stage profile: %+v", profile)
	}

	report := prodMetalStageProfile{
		Version:            1,
		LogitsSHA256:       metalFloat32RowsDigest(warmLogits, logits),
		Dim:                model.Config.Dim,
		Hidden:             model.Config.Hidden,
		Layers:             model.Config.Layers,
		CommandDurationNS:  profile.CommandDuration.Nanoseconds(),
		EventSpanNS:        profile.EventSpan.Nanoseconds(),
		OmittedMPS:         profile.OmittedMPS,
		OmittedOverflow:    profile.OmittedOverflow,
		OmittedUnsupported: profile.OmittedUnsupported,
		Events:             make([]prodMetalStageProfileEvent, len(profile.Events)),
	}
	for i, event := range profile.Events {
		report.Events[i] = prodMetalStageProfileEvent{
			Label:         event.Label,
			StartOffsetNS: event.StartOffset.Nanoseconds(),
			DurationNS:    event.Duration.Nanoseconds(),
			Ticks:         event.Ticks,
		}
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		b.Fatalf("create stage profile: %v", err)
	}
	encodeErr := json.NewEncoder(f).Encode(report)
	closeErr := f.Close()
	if encodeErr != nil {
		b.Fatalf("encode stage profile: %v", encodeErr)
	}
	if closeErr != nil {
		b.Fatalf("close stage profile: %v", closeErr)
	}
}
