package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
)

const stageReportVersion = 4

type stageProfileFile struct {
	Version            int                     `json:"version"`
	LogitsSHA256       string                  `json:"logits_sha256"`
	Dim                int                     `json:"dim"`
	Hidden             int                     `json:"hidden"`
	Layers             int                     `json:"layers"`
	CommandDurationNS  int64                   `json:"command_duration_ns"`
	EventSpanNS        int64                   `json:"event_span_ns"`
	OmittedMPS         int                     `json:"omitted_mps"`
	OmittedOverflow    int                     `json:"omitted_overflow"`
	OmittedUnsupported int                     `json:"omitted_unsupported"`
	Events             []stageProfileFileEvent `json:"events"`
}

type stageProfileFileEvent struct {
	Label         string `json:"label"`
	StartOffsetNS int64  `json:"start_offset_ns"`
	DurationNS    int64  `json:"duration_ns"`
	Ticks         uint64 `json:"ticks"`
}

type gpuInterval struct {
	StartNS         uint64
	EndNS           uint64
	CommandBufferID uint64
	Channel         string
	Process         string
	GPU             string
}

type gpuOverlapReport struct {
	IntervalCount int      `json:"interval_count"`
	DurationNS    uint64   `json:"duration_ns"`
	Processes     []string `json:"processes"`
}

type alignedStageInterval struct {
	StartNS uint64
	EndNS   uint64
	Label   string
}

type stageReport struct {
	SourceVersion     int                  `json:"source_version"`
	LogitsSHA256      string               `json:"logits_sha256"`
	Dim               int                  `json:"dim"`
	Hidden            int                  `json:"hidden"`
	Layers            int                  `json:"layers"`
	CommandDurationNS int64                `json:"command_duration_ns"`
	EventSpanNS       int64                `json:"event_span_ns"`
	GPUStartNS        uint64               `json:"gpu_start_ns"`
	GPUEndNS          uint64               `json:"gpu_end_ns"`
	GPUSpanNS         uint64               `json:"gpu_span_ns"`
	AlignmentScale    float64              `json:"alignment_scale"`
	GPUIntervalCount  int                  `json:"gpu_interval_count"`
	GPUOverlap        gpuOverlapReport     `json:"gpu_overlap"`
	EventCount        int                  `json:"event_count"`
	Stages            []stageCounterReport `json:"stages"`
}

type stageCounterReport struct {
	Label      string             `json:"label"`
	EventCount int                `json:"event_count"`
	DurationNS uint64             `json:"duration_ns"`
	Counters   []stageCounterStat `json:"counters"`
}

type stageCounterStat struct {
	ID                uint32      `json:"id"`
	Name              string      `json:"name"`
	Type              string      `json:"type"`
	MaxValue          uint64      `json:"max_value"`
	SampleIntervalRaw uint32      `json:"sample_interval_raw"`
	Policy            string      `json:"policy"`
	Missing           bool        `json:"missing"`
	Active            sampleStats `json:"active_intervals"`
}

func analyzeStageProfile(paths map[string]string, infos map[uint32]counterInfo, selected []commandBuffer, policy counterPolicy, requireExclusiveGPU bool) (stageReport, error) {
	if len(selected) != 1 {
		return stageReport{}, fmt.Errorf("stage profile requires one selected command buffer, got %d", len(selected))
	}
	profileFile, err := os.Open(paths["stage-profile"])
	if err != nil {
		return stageReport{}, fmt.Errorf("open stage profile: %w", err)
	}
	profile, err := decodeStageProfile(profileFile)
	profileFile.Close()
	if err != nil {
		return stageReport{}, err
	}

	gpuFile, err := os.Open(paths["gpu-intervals"])
	if err != nil {
		return stageReport{}, fmt.Errorf("open GPU intervals: %w", err)
	}
	gpuIntervals, overlap, err := parseGPUIntervalsWithOverlap(gpuFile, selected[0])
	gpuFile.Close()
	if err != nil {
		return stageReport{}, err
	}
	if err := validateGPUOverlap(overlap, requireExclusiveGPU); err != nil {
		return stageReport{}, err
	}
	aligned, gpuStart, gpuEnd, scale, err := alignStageIntervals(profile, gpuIntervals)
	if err != nil {
		return stageReport{}, err
	}

	valueFile, err := os.Open(paths["counter-value"])
	if err != nil {
		return stageReport{}, fmt.Errorf("open counter values for stages: %w", err)
	}
	stages, err := analyzeStageValuesWithPolicy(valueFile, infos, aligned, policy)
	valueFile.Close()
	if err != nil {
		return stageReport{}, err
	}
	return stageReport{
		SourceVersion:     profile.Version,
		LogitsSHA256:      profile.LogitsSHA256,
		Dim:               profile.Dim,
		Hidden:            profile.Hidden,
		Layers:            profile.Layers,
		CommandDurationNS: profile.CommandDurationNS,
		EventSpanNS:       profile.EventSpanNS,
		GPUStartNS:        gpuStart,
		GPUEndNS:          gpuEnd,
		GPUSpanNS:         gpuEnd - gpuStart,
		AlignmentScale:    scale,
		GPUIntervalCount:  len(gpuIntervals),
		GPUOverlap:        overlap,
		EventCount:        len(profile.Events),
		Stages:            stages,
	}, nil
}

func decodeStageProfile(r io.Reader) (stageProfileFile, error) {
	var profile stageProfileFile
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&profile); err != nil {
		return stageProfileFile{}, fmt.Errorf("decode stage profile: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return stageProfileFile{}, errors.New("stage profile has trailing JSON value")
		}
		return stageProfileFile{}, fmt.Errorf("decode stage profile trailer: %w", err)
	}
	if profile.Version != 1 {
		return stageProfileFile{}, fmt.Errorf("stage profile version=%d, want 1", profile.Version)
	}
	if profile.LogitsSHA256 == "" || profile.Dim <= 0 || profile.Hidden <= 0 || profile.Layers <= 0 {
		return stageProfileFile{}, errors.New("stage profile has incomplete model identity")
	}
	if profile.OmittedMPS != 0 || profile.OmittedOverflow != 0 || profile.OmittedUnsupported != 0 {
		return stageProfileFile{}, fmt.Errorf("stage profile has omissions: mps=%d overflow=%d unsupported=%d",
			profile.OmittedMPS, profile.OmittedOverflow, profile.OmittedUnsupported)
	}
	if profile.CommandDurationNS <= 0 || profile.EventSpanNS <= 0 || len(profile.Events) == 0 {
		return stageProfileFile{}, errors.New("stage profile has no positive command/event span")
	}
	var previousEnd int64
	for i, event := range profile.Events {
		if event.Label == "" || event.StartOffsetNS < 0 || event.DurationNS <= 0 || event.Ticks == 0 {
			return stageProfileFile{}, fmt.Errorf("stage event %d is incomplete: %+v", i, event)
		}
		if i == 0 && event.StartOffsetNS != 0 {
			return stageProfileFile{}, fmt.Errorf("first stage event offset=%d, want 0", event.StartOffsetNS)
		}
		if event.StartOffsetNS < previousEnd {
			return stageProfileFile{}, fmt.Errorf("stage event %d overlaps its predecessor", i)
		}
		if event.StartOffsetNS > math.MaxInt64-event.DurationNS {
			return stageProfileFile{}, fmt.Errorf("stage event %d end overflows", i)
		}
		previousEnd = event.StartOffsetNS + event.DurationNS
	}
	if previousEnd != profile.EventSpanNS {
		return stageProfileFile{}, fmt.Errorf("stage event span=%d, declared=%d", previousEnd, profile.EventSpanNS)
	}
	if !withinRelative(profile.EventSpanNS, profile.CommandDurationNS, 0.05) {
		return stageProfileFile{}, fmt.Errorf("stage span=%d and command duration=%d differ by more than 5%%", profile.EventSpanNS, profile.CommandDurationNS)
	}
	return profile, nil
}

func parseGPUIntervals(r io.Reader, command commandBuffer) ([]gpuInterval, error) {
	intervals, _, err := parseGPUIntervalsWithOverlap(r, command)
	return intervals, err
}

func parseGPUIntervalsWithOverlap(r io.Reader, command commandBuffer) ([]gpuInterval, gpuOverlapReport, error) {
	if command.ID == 0 || command.Process == "" || command.GPU == "" {
		return nil, gpuOverlapReport{}, errors.New("selected command buffer has incomplete target identity")
	}
	dec := newRowDecoder(r)
	var sameGPU []gpuInterval
	for {
		row, err := dec.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, gpuOverlapReport{}, fmt.Errorf("GPU interval XML: %w", err)
		}
		if len(row) != 18 {
			return nil, gpuOverlapReport{}, fmt.Errorf("GPU interval row has %d columns, want 18", len(row))
		}
		id, err := parseUint64(row[15], "GPU interval command-buffer ID")
		if err != nil {
			return nil, gpuOverlapReport{}, err
		}
		if row[11] != command.GPU {
			continue
		}
		start, err := parseUint64(row[0], "GPU interval start")
		if err != nil {
			return nil, gpuOverlapReport{}, err
		}
		duration, err := parseUint64(row[1], "GPU interval duration")
		if err != nil {
			return nil, gpuOverlapReport{}, err
		}
		if duration == 0 || start > math.MaxUint64-duration {
			return nil, gpuOverlapReport{}, fmt.Errorf("invalid GPU interval start=%d duration=%d", start, duration)
		}
		sameGPU = append(sameGPU, gpuInterval{
			StartNS: start, EndNS: start + duration, CommandBufferID: id,
			Channel: row[2], Process: row[10], GPU: row[11],
		})
	}
	intervals := make([]gpuInterval, 0, len(sameGPU))
	for _, interval := range sameGPU {
		if interval.CommandBufferID == command.ID && interval.Process == command.Process {
			intervals = append(intervals, interval)
		}
	}
	if len(intervals) == 0 {
		return nil, gpuOverlapReport{}, fmt.Errorf("no GPU intervals found for process %q GPU %q command buffer %d", command.Process, command.GPU, command.ID)
	}
	sort.Slice(intervals, func(i, j int) bool { return intervals[i].StartNS < intervals[j].StartNS })
	targetStart, targetEnd := intervals[0].StartNS, intervals[0].EndNS
	for _, interval := range intervals[1:] {
		if interval.StartNS < targetStart {
			targetStart = interval.StartNS
		}
		if interval.EndNS > targetEnd {
			targetEnd = interval.EndNS
		}
	}
	overlap := summarizeGPUOverlap(sameGPU, command, targetStart, targetEnd)
	return intervals, overlap, nil
}

func summarizeGPUOverlap(intervals []gpuInterval, command commandBuffer, targetStart, targetEnd uint64) gpuOverlapReport {
	type span struct{ start, end uint64 }
	spans := make([]span, 0)
	processes := make(map[string]struct{})
	count := 0
	for _, interval := range intervals {
		if interval.GPU != command.GPU || (interval.CommandBufferID == command.ID && interval.Process == command.Process) {
			continue
		}
		start := max(interval.StartNS, targetStart)
		end := min(interval.EndNS, targetEnd)
		if start >= end {
			continue
		}
		count++
		spans = append(spans, span{start: start, end: end})
		process := interval.Process
		if process == "" {
			process = "<unknown>"
		}
		processes[process] = struct{}{}
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	var duration uint64
	if len(spans) != 0 {
		start, end := spans[0].start, spans[0].end
		for _, current := range spans[1:] {
			if current.start <= end {
				end = max(end, current.end)
				continue
			}
			duration += end - start
			start, end = current.start, current.end
		}
		duration += end - start
	}
	names := make([]string, 0, len(processes))
	for process := range processes {
		names = append(names, process)
	}
	sort.Strings(names)
	return gpuOverlapReport{IntervalCount: count, DurationNS: duration, Processes: names}
}

func validateGPUOverlap(overlap gpuOverlapReport, required bool) error {
	if required && overlap.DurationNS != 0 {
		return fmt.Errorf("target GPU window overlaps %d foreign intervals for %d ns from processes %q",
			overlap.IntervalCount, overlap.DurationNS, overlap.Processes)
	}
	return nil
}

func alignStageIntervals(profile stageProfileFile, gpu []gpuInterval) ([]alignedStageInterval, uint64, uint64, float64, error) {
	if len(gpu) == 0 {
		return nil, 0, 0, 0, errors.New("GPU interval set is empty")
	}
	start, end := gpu[0].StartNS, gpu[0].EndNS
	for _, interval := range gpu[1:] {
		if interval.StartNS < start {
			start = interval.StartNS
		}
		if interval.EndNS > end {
			end = interval.EndNS
		}
	}
	if end <= start {
		return nil, 0, 0, 0, errors.New("GPU interval span is empty")
	}
	gpuSpan := end - start
	if !withinRelative(int64(gpuSpan), profile.CommandDurationNS, 0.05) {
		return nil, 0, 0, 0, fmt.Errorf("GPU span=%d and profiled command duration=%d differ by more than 5%%", gpuSpan, profile.CommandDurationNS)
	}
	scale := float64(gpuSpan) / float64(profile.EventSpanNS)
	aligned := make([]alignedStageInterval, len(profile.Events))
	for i, event := range profile.Events {
		alignedStart := start + uint64(math.Round(float64(event.StartOffsetNS)*scale))
		alignedEnd := start + uint64(math.Round(float64(event.StartOffsetNS+event.DurationNS)*scale))
		if alignedEnd <= alignedStart || alignedEnd > end {
			return nil, 0, 0, 0, fmt.Errorf("aligned stage event %d is outside GPU span", i)
		}
		if i > 0 && alignedStart < aligned[i-1].EndNS {
			return nil, 0, 0, 0, fmt.Errorf("aligned stage event %d overlaps its predecessor", i)
		}
		aligned[i] = alignedStageInterval{StartNS: alignedStart, EndNS: alignedEnd, Label: canonicalStageLabel(event.Label)}
	}
	return aligned, start, end, scale, nil
}

func canonicalStageLabel(label string) string {
	switch label {
	case "blit", "rope", "rope_pair":
		return "short.support"
	}
	return label
}

func withinRelative(a, b int64, tolerance float64) bool {
	if a <= 0 || b <= 0 {
		return false
	}
	delta := math.Abs(float64(a - b))
	return delta/float64(max(a, b)) <= tolerance
}

func analyzeStageValues(r io.Reader, infos map[uint32]counterInfo, intervals []alignedStageInterval) ([]stageCounterReport, error) {
	return analyzeStageValuesWithPolicy(r, infos, intervals, strictCounterPolicy())
}

func analyzeStageValuesWithPolicy(r io.Reader, infos map[uint32]counterInfo, intervals []alignedStageInterval, policy counterPolicy) ([]stageCounterReport, error) {
	if len(intervals) == 0 {
		return nil, errors.New("aligned stage interval set is empty")
	}
	type stageAccumulator struct {
		count    int
		duration uint64
		stats    map[uint32]*sampleStats
	}
	accumulators := make(map[string]*stageAccumulator)
	for _, interval := range intervals {
		acc := accumulators[interval.Label]
		if acc == nil {
			acc = &stageAccumulator{stats: make(map[uint32]*sampleStats, len(infos))}
			for id := range infos {
				stats := emptyStats()
				acc.stats[id] = &stats
			}
			accumulators[interval.Label] = acc
		}
		acc.count++
		acc.duration += interval.EndNS - interval.StartNS
	}

	dec := newRowDecoder(r)
	intervalIndex := 0
	var previousTimestamp uint64
	var sawTimestamp bool
	for {
		row, err := dec.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("stage counter value XML: %w", err)
		}
		if len(row) != 6 {
			return nil, fmt.Errorf("stage counter value row has %d columns, want 6", len(row))
		}
		timestamp, err := parseUint64(row[0], "stage counter timestamp")
		if err != nil {
			return nil, err
		}
		if sawTimestamp && timestamp < previousTimestamp {
			return nil, fmt.Errorf("stage counter timestamps regress from %d to %d", previousTimestamp, timestamp)
		}
		previousTimestamp, sawTimestamp = timestamp, true
		if timestamp < intervals[0].StartNS {
			continue
		}
		if timestamp >= intervals[len(intervals)-1].EndNS {
			break
		}
		for intervalIndex < len(intervals) && timestamp >= intervals[intervalIndex].EndNS {
			intervalIndex++
		}
		if intervalIndex >= len(intervals) || timestamp < intervals[intervalIndex].StartNS {
			continue
		}
		id, err := parseUint32(row[1], "stage counter sample ID")
		if err != nil {
			return nil, err
		}
		value, err := strconv.ParseFloat(row[2], 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("invalid stage counter %d value %q", id, row[2])
		}
		acc := accumulators[intervals[intervalIndex].Label]
		stats, ok := acc.stats[id]
		if !ok {
			return nil, fmt.Errorf("stage counter sample references unknown counter ID %d", id)
		}
		stats.add(value)
	}

	labels := make([]string, 0, len(accumulators))
	for label := range accumulators {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	result := make([]stageCounterReport, 0, len(labels))
	for _, label := range labels {
		acc := accumulators[label]
		ids := make([]int, 0, len(infos))
		for id := range infos {
			ids = append(ids, int(id))
		}
		sort.Ints(ids)
		stage := stageCounterReport{Label: label, EventCount: acc.count, DurationNS: acc.duration, Counters: make([]stageCounterStat, 0, len(ids))}
		for _, rawID := range ids {
			id := uint32(rawID)
			stats := acc.stats[id]
			class := policy.stageClass(infos[id].Name, acc.duration)
			missing := stats.Samples == 0
			if missing && class == "required" {
				return nil, fmt.Errorf("stage %q counter %q has no samples", label, infos[id].Name)
			}
			stats.finish()
			info := infos[id]
			stage.Counters = append(stage.Counters, stageCounterStat{
				ID: info.ID, Name: info.Name, Type: info.Type, MaxValue: info.MaxValue,
				SampleIntervalRaw: info.SampleIntervalRaw, Policy: class, Missing: missing, Active: *stats,
			})
		}
		result = append(result, stage)
	}
	return result, nil
}
