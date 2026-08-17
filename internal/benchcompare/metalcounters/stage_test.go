package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestDecodeAndAlignStageProfile(t *testing.T) {
	profile := stageProfileFile{
		Version: 1, LogitsSHA256: "digest", Dim: 2048, Hidden: 5632, Layers: 22,
		CommandDurationNS: 100, EventSpanNS: 100,
		Events: []stageProfileFileEvent{
			{Label: "qmatmul.q4_k", StartOffsetNS: 0, DurationNS: 40, Ticks: 40},
			{Label: "mha.decode", StartOffsetNS: 50, DurationNS: 50, Ticks: 50},
		},
	}
	data, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeStageProfile(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	aligned, start, end, scale, err := alignStageIntervals(decoded, []gpuInterval{{StartNS: 1000, EndNS: 1100}})
	if err != nil {
		t.Fatal(err)
	}
	if start != 1000 || end != 1100 || scale != 1 || len(aligned) != 2 || aligned[1].StartNS != 1050 || aligned[1].EndNS != 1100 {
		t.Fatalf("aligned=%+v start=%d end=%d scale=%g", aligned, start, end, scale)
	}

	profile.Events[1].StartOffsetNS = 39
	data, _ = json.Marshal(profile)
	if _, err := decodeStageProfile(bytes.NewReader(data)); err == nil || !strings.Contains(err.Error(), "overlaps") {
		t.Fatalf("err=%v, want overlap rejection", err)
	}
	profile.Events[1].StartOffsetNS = 50
	if _, _, _, _, err := alignStageIntervals(profile, []gpuInterval{{StartNS: 1000, EndNS: 1200}}); err == nil || !strings.Contains(err.Error(), "more than 5%") {
		t.Fatalf("err=%v, want span rejection", err)
	}
}

func TestDecodeStageProfileRejectsOmissionsAndTrailingValue(t *testing.T) {
	profile := stageProfileFile{
		Version: 1, LogitsSHA256: "digest", Dim: 1, Hidden: 1, Layers: 1,
		CommandDurationNS: 10, EventSpanNS: 10, OmittedMPS: 1,
		Events: []stageProfileFileEvent{{Label: "stage", DurationNS: 10, Ticks: 10}},
	}
	data, _ := json.Marshal(profile)
	if _, err := decodeStageProfile(bytes.NewReader(data)); err == nil || !strings.Contains(err.Error(), "omissions") {
		t.Fatalf("err=%v, want omissions rejection", err)
	}
	profile.OmittedMPS = 0
	data, _ = json.Marshal(profile)
	data = append(data, []byte(` {"extra":true}`)...)
	if _, err := decodeStageProfile(bytes.NewReader(data)); err == nil || !strings.Contains(err.Error(), "trailing") {
		t.Fatalf("err=%v, want trailing-value rejection", err)
	}
}

func TestParseGPUIntervalsResolvesReferencesAndFiltersCommandBuffer(t *testing.T) {
	xml := `<trace-query-result>` +
		`<row><start-time id="s">100</start-time><duration id="d">20</duration><gpu-channel-name id="c">Compute</gpu-channel-name><sentinel/><duration>1</duration><metal-nesting-level>0</metal-nesting-level><formatted-label>target</formatted-label><gpu-state>Active</gpu-state><connection-uuid64>1</connection-uuid64><render-buffer-depth>0</render-buffer-depth><process>target</process><metal-device-name>M2 Pro</metal-device-name><metal-object-label></metal-object-label><formatted-label></formatted-label><size-in-bytes>0</size-in-bytes><metal-command-buffer-id id="cb">7</metal-command-buffer-id><metal-command-buffer-id>8</metal-command-buffer-id><uint64>9</uint64></row>` +
		`<row><start-time ref="s"/><duration ref="d"/><gpu-channel-name ref="c"/><sentinel/><duration>1</duration><metal-nesting-level>0</metal-nesting-level><formatted-label>other</formatted-label><gpu-state>Active</gpu-state><connection-uuid64>2</connection-uuid64><render-buffer-depth>0</render-buffer-depth><process>other</process><metal-device-name>M2 Pro</metal-device-name><metal-object-label></metal-object-label><formatted-label></formatted-label><size-in-bytes>0</size-in-bytes><metal-command-buffer-id>99</metal-command-buffer-id><metal-command-buffer-id>10</metal-command-buffer-id><uint64>11</uint64></row>` +
		`</trace-query-result>`
	intervals, err := parseGPUIntervals(strings.NewReader(xml), commandBuffer{ID: 7, Process: "target", GPU: "M2 Pro"})
	if err != nil {
		t.Fatal(err)
	}
	if len(intervals) != 1 || intervals[0].StartNS != 100 || intervals[0].EndNS != 120 || intervals[0].Channel != "Compute" {
		t.Fatalf("intervals=%+v", intervals)
	}
	if _, err := parseGPUIntervals(strings.NewReader(xml), commandBuffer{ID: 7, Process: "other", GPU: "M2 Pro"}); err == nil || !strings.Contains(err.Error(), "no GPU intervals") {
		t.Fatalf("err=%v, want process mismatch rejection", err)
	}
}

func TestParseGPUIntervalsReportsForeignOverlap(t *testing.T) {
	xml := `<trace-query-result>` +
		gpuIntervalRow(100, 50, "Compute", "target", "M2 Pro", 7) +
		gpuIntervalRow(150, 50, "Compute", "target", "M2 Pro", 7) +
		gpuIntervalRow(90, 20, "Render", "other-a", "M2 Pro", 8) +
		gpuIntervalRow(105, 20, "Compute", "other-a", "M2 Pro", 9) +
		gpuIntervalRow(120, 30, "Compute", "other-b", "M2 Pro", 10) +
		gpuIntervalRow(200, 10, "Render", "touches-boundary", "M2 Pro", 11) +
		gpuIntervalRow(130, 10, "Compute", "other-gpu", "M3", 12) +
		`</trace-query-result>`
	intervals, overlap, err := parseGPUIntervalsWithOverlap(strings.NewReader(xml), commandBuffer{ID: 7, Process: "target", GPU: "M2 Pro"})
	if err != nil {
		t.Fatal(err)
	}
	if len(intervals) != 2 || intervals[0].StartNS != 100 || intervals[1].EndNS != 200 {
		t.Fatalf("target intervals=%+v", intervals)
	}
	if overlap.IntervalCount != 3 || overlap.DurationNS != 50 || len(overlap.Processes) != 2 ||
		overlap.Processes[0] != "other-a" || overlap.Processes[1] != "other-b" {
		t.Fatalf("overlap=%+v", overlap)
	}
	if err := validateGPUOverlap(overlap, false); err != nil {
		t.Fatalf("default behavior rejected overlap: %v", err)
	}
	if err := validateGPUOverlap(overlap, true); err == nil || !strings.Contains(err.Error(), "3 foreign intervals for 50 ns") {
		t.Fatalf("err=%v, want exclusive-window rejection", err)
	}
	encoded, err := json.Marshal(stageReport{GPUOverlap: overlap})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"interval_count":3`) || !strings.Contains(string(encoded), `"duration_ns":50`) {
		t.Fatalf("encoded report=%s", encoded)
	}
}

func gpuIntervalRow(start, duration int, channel, process, gpu string, commandID int) string {
	return `<row><start-time>` + stringInt(start) + `</start-time><duration>` + stringInt(duration) +
		`</duration><gpu-channel-name>` + channel + `</gpu-channel-name><sentinel/><duration>1</duration>` +
		`<metal-nesting-level>0</metal-nesting-level><formatted-label>work</formatted-label><gpu-state>Active</gpu-state>` +
		`<connection-uuid64>1</connection-uuid64><render-buffer-depth>0</render-buffer-depth><process>` + process +
		`</process><metal-device-name>` + gpu + `</metal-device-name><metal-object-label></metal-object-label>` +
		`<formatted-label></formatted-label><size-in-bytes>0</size-in-bytes><metal-command-buffer-id>` + stringInt(commandID) +
		`</metal-command-buffer-id><metal-command-buffer-id>0</metal-command-buffer-id><uint64>0</uint64></row>`
}

func TestAnalyzeStageValues(t *testing.T) {
	infos := map[uint32]counterInfo{
		0: {ID: 0, Name: "ALU Limiter", Type: "Percentage", MaxValue: 100, SampleIntervalRaw: 10},
		1: {ID: 1, Name: "Buffer Read Limiter", Type: "Percentage", MaxValue: 100, SampleIntervalRaw: 10},
	}
	intervals := []alignedStageInterval{
		{StartNS: 100, EndNS: 150, Label: "a"},
		{StartNS: 200, EndNS: 250, Label: "b"},
	}
	xml := `<trace-query-result>` +
		`<row><event-time>110</event-time><uint32>0</uint32><fixed-decimal>10</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>` +
		`<row><event-time>110</event-time><uint32>1</uint32><fixed-decimal>20</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>` +
		`<row><event-time>210</event-time><uint32>0</uint32><fixed-decimal>30</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>` +
		`<row><event-time>210</event-time><uint32>1</uint32><fixed-decimal>40</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>` +
		`</trace-query-result>`
	stages, err := analyzeStageValues(strings.NewReader(xml), infos, intervals)
	if err != nil {
		t.Fatal(err)
	}
	if len(stages) != 2 || stages[0].Label != "a" || stages[0].Counters[0].Active.Mean != 10 || stages[1].Counters[1].Active.Mean != 40 {
		t.Fatalf("stages=%+v", stages)
	}
}

func TestAnalyzeStageValuesPolicyResolutionFloor(t *testing.T) {
	infos := map[uint32]counterInfo{
		0: {ID: 0, Name: "ALU Limiter", Type: "Percentage", MaxValue: 100, SampleIntervalRaw: 10},
		1: {ID: 1, Name: "GPU Read Bandwidth", Type: "Rate", SampleIntervalRaw: 10},
	}
	intervals := []alignedStageInterval{
		{StartNS: 100, EndNS: 150, Label: "short"},
		{StartNS: 200, EndNS: 400, Label: "long"},
	}
	xml := `<trace-query-result>` +
		`<row><event-time>250</event-time><uint32>0</uint32><fixed-decimal>25</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>` +
		`</trace-query-result>`
	policy, err := parseCounterPolicy("ALU Limiter", "GPU Read Bandwidth", "ALU Limiter", 100, nil)
	if err != nil {
		t.Fatal(err)
	}
	stages, err := analyzeStageValuesWithPolicy(strings.NewReader(xml), infos, intervals, policy)
	if err != nil {
		t.Fatal(err)
	}
	if len(stages) != 2 || stages[0].Label != "long" || stages[1].Label != "short" {
		t.Fatalf("stages=%+v", stages)
	}
	if stages[0].Counters[0].Policy != "required" || stages[0].Counters[0].Missing ||
		stages[0].Counters[1].Policy != "capability" || !stages[0].Counters[1].Missing {
		t.Fatalf("long stage=%+v", stages[0])
	}
	for _, counter := range stages[1].Counters {
		if counter.Policy != "sparse-stage" || !counter.Missing || counter.Active.Mean != 0 {
			t.Fatalf("short-stage counter=%+v", counter)
		}
	}
	if _, err := json.Marshal(stages); err != nil {
		t.Fatalf("marshal finite missing-stage report: %v", err)
	}
	if _, err := analyzeStageValues(strings.NewReader(xml), infos, intervals); err == nil {
		t.Fatal("strict compatibility mode unexpectedly accepted sparse counters")
	}
	required, err := parseCounterPolicy("ALU Limiter", "", "GPU Read Bandwidth", 100, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := analyzeStageValuesWithPolicy(strings.NewReader(xml), infos, intervals, required); err == nil || !strings.Contains(err.Error(), "GPU Read Bandwidth") {
		t.Fatalf("err=%v, want resolved-stage required-counter rejection", err)
	}
}

func TestCanonicalStageLabel(t *testing.T) {
	if got := canonicalStageLabel("rope_pair"); got != "short.support" {
		t.Fatalf("canonical rope_pair=%q", got)
	}
	if got := canonicalStageLabel("blit"); got != "short.support" {
		t.Fatalf("canonical blit=%q", got)
	}
	if got := canonicalStageLabel("qmatmul.q4_k"); got != "qmatmul.q4_k" {
		t.Fatalf("canonical qmatmul=%q", got)
	}
}
