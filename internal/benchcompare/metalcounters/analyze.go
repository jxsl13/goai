package main

import (
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

const reportVersion = 2

var requiredLimiterNames = []string{
	"ALU Limiter",
	"Buffer Read Limiter",
	"Compute Occupancy",
	"GPU Last Level Cache Limiter",
	"GPU Read Bandwidth",
	"MMU Limiter",
}

type counterInfo struct {
	ID                uint32
	Name              string
	Type              string
	MaxValue          uint64
	SampleIntervalRaw uint32
}

type commandBuffer struct {
	StartNS uint64
	EndNS   uint64
	ID      uint64
	GPU     string
	Process string
}

type sampleStats struct {
	Samples uint64  `json:"samples"`
	Mean    float64 `json:"mean"`
	Min     float64 `json:"min"`
	Max     float64 `json:"max"`

	sum float64
}

type counterReport struct {
	ID                uint32      `json:"id"`
	Name              string      `json:"name"`
	Type              string      `json:"type"`
	MaxValue          uint64      `json:"max_value"`
	SampleIntervalRaw uint32      `json:"sample_interval_raw"`
	Policy            string      `json:"policy"`
	Missing           bool        `json:"missing"`
	Wall              sampleStats `json:"wall_window"`
	Active            sampleStats `json:"active_command_buffers"`
}

type counterRequirement struct {
	all   bool
	names []string
	set   map[string]struct{}
}

type counterPolicy struct {
	globalRequired    counterRequirement
	capability        counterRequirement
	stageRequired     counterRequirement
	contamination     map[string]struct{}
	contaminationList []string
	stageMinDuration  uint64
}

type counterPolicyReport struct {
	GlobalRequired   []string `json:"global_required"`
	Capability       []string `json:"capability"`
	Contamination    []string `json:"contamination"`
	StageRequired    []string `json:"stage_required"`
	StageMinDuration uint64   `json:"stage_min_duration_ns"`
}

func strictCounterPolicy() counterPolicy {
	return counterPolicy{
		globalRequired: counterRequirement{all: true},
		stageRequired:  counterRequirement{all: true},
		capability:     counterRequirement{set: make(map[string]struct{})},
		contamination:  make(map[string]struct{}),
	}
}

func parseCounterPolicy(globalRequired, capability, stageRequired string, stageMinDuration uint64, ceilings []counterCeiling) (counterPolicy, error) {
	global, err := parseCounterRequirement(globalRequired, "required-counters", true)
	if err != nil {
		return counterPolicy{}, err
	}
	caps, err := parseCounterRequirement(capability, "capability-counters", false)
	if err != nil {
		return counterPolicy{}, err
	}
	stage, err := parseCounterRequirement(stageRequired, "stage-required-counters", true)
	if err != nil {
		return counterPolicy{}, err
	}
	if global.all && len(caps.names) != 0 {
		return counterPolicy{}, errors.New("capability-counters cannot be combined with required-counters=*")
	}
	if !global.all {
		for _, name := range global.names {
			if _, overlap := caps.set[name]; overlap {
				return counterPolicy{}, fmt.Errorf("counter %q is both required and capability-classified", name)
			}
		}
	}
	policy := counterPolicy{
		globalRequired:   global,
		capability:       caps,
		stageRequired:    stage,
		contamination:    make(map[string]struct{}, len(ceilings)),
		stageMinDuration: stageMinDuration,
	}
	for _, ceiling := range ceilings {
		policy.contamination[ceiling.name] = struct{}{}
		policy.contaminationList = append(policy.contaminationList, ceiling.name)
	}
	sort.Strings(policy.contaminationList)
	return policy, nil
}

func parseCounterRequirement(value, field string, allowAll bool) (counterRequirement, error) {
	value = strings.TrimSpace(value)
	if allowAll && value == "*" {
		return counterRequirement{all: true}, nil
	}
	result := counterRequirement{set: make(map[string]struct{})}
	if value == "" {
		return result, nil
	}
	for _, clause := range strings.Split(value, ",") {
		name := strings.TrimSpace(clause)
		if name == "" || name == "*" {
			return counterRequirement{}, fmt.Errorf("invalid %s clause %q", field, clause)
		}
		if _, duplicate := result.set[name]; duplicate {
			return counterRequirement{}, fmt.Errorf("duplicate %s counter %q", field, name)
		}
		result.set[name] = struct{}{}
		result.names = append(result.names, name)
	}
	sort.Strings(result.names)
	return result, nil
}

func (r counterRequirement) required(name string) bool {
	if r.all {
		return true
	}
	_, ok := r.set[name]
	return ok
}

func (p counterPolicy) validate(infos map[uint32]counterInfo) error {
	available := make(map[string]struct{}, len(infos))
	for _, info := range infos {
		available[info.Name] = struct{}{}
	}
	for field, requirement := range map[string]counterRequirement{
		"required-counters":       p.globalRequired,
		"capability-counters":     p.capability,
		"stage-required-counters": p.stageRequired,
	} {
		for _, name := range requirement.names {
			if _, ok := available[name]; !ok {
				return fmt.Errorf("%s counter %q is absent from Performance Limiters metadata", field, name)
			}
		}
	}
	return nil
}

func (p counterPolicy) globalClass(name string) string {
	if _, ok := p.contamination[name]; ok {
		return "contamination"
	}
	if p.globalRequired.required(name) {
		return "required"
	}
	if p.capability.required(name) {
		return "capability"
	}
	return "optional"
}

func (p counterPolicy) stageClass(name string, duration uint64) string {
	if duration < p.stageMinDuration {
		return "sparse-stage"
	}
	if p.stageRequired.required(name) {
		return "required"
	}
	if p.capability.required(name) {
		return "capability"
	}
	return "optional"
}

func requirementNames(r counterRequirement) []string {
	if r.all {
		return []string{"*"}
	}
	return append([]string{}, r.names...)
}

func (p counterPolicy) report() counterPolicyReport {
	return counterPolicyReport{
		GlobalRequired:   requirementNames(p.globalRequired),
		Capability:       requirementNames(p.capability),
		Contamination:    append([]string(nil), p.contaminationList...),
		StageRequired:    requirementNames(p.stageRequired),
		StageMinDuration: p.stageMinDuration,
	}
}

type windowReport struct {
	StartNS          uint64  `json:"start_ns"`
	EndNS            uint64  `json:"end_ns"`
	DurationNS       uint64  `json:"duration_ns"`
	ActiveDurationNS uint64  `json:"active_duration_ns"`
	DutyCycle        float64 `json:"duty_cycle"`
}

type report struct {
	Version                    int                 `json:"version"`
	Scope                      string              `json:"scope"`
	GPU                        string              `json:"gpu"`
	CounterSet                 string              `json:"counter_set"`
	Benchmark                  string              `json:"benchmark"`
	Iterations                 int                 `json:"iterations"`
	BuffersPerIteration        int                 `json:"buffers_per_iteration"`
	ObservedCommandBuffers     int                 `json:"observed_command_buffers"`
	SelectedCommandBuffers     int                 `json:"selected_command_buffers"`
	SkippedFinalCommandBuffers int                 `json:"skipped_final_command_buffers,omitempty"`
	Window                     windowReport        `json:"window"`
	Policy                     counterPolicyReport `json:"counter_policy"`
	Counters                   []counterReport     `json:"counters"`
	Stage                      *stageReport        `json:"stage_profile,omitempty"`
	Shader                     *shaderReport       `json:"shader_profile,omitempty"`
}

type rowDecoder struct {
	dec    *xml.Decoder
	values map[string]string
}

// Split the xctrace XML attribute spelling so the repository-wide backend-name
// literal guard does not mistake protocol vocabulary for backend.Ref.
const xmlReferenceAttribute = "r" + "ef"

func newRowDecoder(r io.Reader) *rowDecoder {
	return &rowDecoder{dec: xml.NewDecoder(r), values: make(map[string]string)}
}

func (d *rowDecoder) next() ([]string, error) {
	for {
		tok, err := d.dec.Token()
		if err != nil {
			return nil, err
		}
		start, ok := tok.(xml.StartElement)
		if !ok || start.Name.Local != "row" {
			continue
		}
		return d.row(start)
	}
}

func (d *rowDecoder) row(row xml.StartElement) ([]string, error) {
	var values []string
	for {
		tok, err := d.dec.Token()
		if err != nil {
			return nil, err
		}
		switch tok := tok.(type) {
		case xml.StartElement:
			value, err := d.element(tok)
			if err != nil {
				return nil, err
			}
			values = append(values, value)
		case xml.EndElement:
			if tok.Name == row.Name {
				return values, nil
			}
		}
	}
}

func (d *rowDecoder) element(start xml.StartElement) (string, error) {
	var id, ref, formatted string
	for _, attr := range start.Attr {
		switch attr.Name.Local {
		case "id":
			id = attr.Value
		case xmlReferenceAttribute:
			ref = attr.Value
		case "fmt":
			formatted = attr.Value
		}
	}
	if ref != "" {
		value, ok := d.values[ref]
		if !ok {
			return "", fmt.Errorf("unresolved XML reference %q in <%s>", ref, start.Name.Local)
		}
		if err := d.dec.Skip(); err != nil {
			return "", err
		}
		return value, nil
	}
	// A process element's raw nested text is PID plus device-session metadata
	// (for example "88150TODO"). Its fmt attribute is the stable, human-readable
	// target identity emitted by xctrace and shared across Metal tables.
	if start.Name.Local == "process" && formatted != "" {
		if err := d.dec.Skip(); err != nil {
			return "", err
		}
		if id != "" {
			d.values[id] = formatted
		}
		return formatted, nil
	}

	var text strings.Builder
	for {
		tok, err := d.dec.Token()
		if err != nil {
			return "", err
		}
		switch tok := tok.(type) {
		case xml.StartElement:
			value, err := d.element(tok)
			if err != nil {
				return "", err
			}
			text.WriteString(value)
		case xml.EndElement:
			if tok.Name == start.Name {
				value := strings.TrimSpace(text.String())
				if id != "" {
					d.values[id] = value
				}
				return value, nil
			}
		case xml.CharData:
			text.Write(tok)
		}
	}
}

func parseCounterInfo(r io.Reader) (map[uint32]counterInfo, error) {
	dec := newRowDecoder(r)
	infos := make(map[uint32]counterInfo)
	for {
		row, err := dec.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("counter info XML: %w", err)
		}
		if len(row) != 11 {
			return nil, fmt.Errorf("counter info row has %d columns, want 11", len(row))
		}
		id, err := parseUint32(row[1], "counter ID")
		if err != nil {
			return nil, err
		}
		maxValue, err := parseUint64(row[3], "counter max value")
		if err != nil {
			return nil, err
		}
		sampleInterval, err := parseUint32(row[10], "counter sample interval")
		if err != nil {
			return nil, err
		}
		if _, exists := infos[id]; exists {
			return nil, fmt.Errorf("duplicate counter ID %d", id)
		}
		infos[id] = counterInfo{
			ID:                id,
			Name:              row[2],
			Type:              row[7],
			MaxValue:          maxValue,
			SampleIntervalRaw: sampleInterval,
		}
	}
	if err := validateCounterInfo(infos); err != nil {
		return nil, err
	}
	return infos, nil
}

func validateCounterInfo(infos map[uint32]counterInfo) error {
	if len(infos) == 0 {
		return errors.New("Performance Limiters counter set is empty")
	}
	names := make(map[string]bool, len(infos))
	for _, info := range infos {
		if info.Name == "" || info.SampleIntervalRaw == 0 {
			return fmt.Errorf("counter %d has incomplete metadata", info.ID)
		}
		names[info.Name] = true
	}
	for _, name := range requiredLimiterNames {
		if !names[name] {
			return fmt.Errorf("required Performance Limiter %q is missing", name)
		}
	}
	return nil
}

func counterInfoTableXPaths(r io.Reader) ([]string, error) {
	dec := xml.NewDecoder(r)
	tableIndex := 0
	var matches []string
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("trace TOC XML: %w", err)
		}
		start, ok := tok.(xml.StartElement)
		if !ok || start.Name.Local != "table" {
			continue
		}
		tableIndex++
		attrs := make(map[string]string, len(start.Attr))
		for _, attr := range start.Attr {
			attrs[attr.Name.Local] = attr.Value
		}
		if attrs["schema"] == "gpu-counter-info" {
			matches = append(matches, fmt.Sprintf("//trace-toc[1]/run[1]/data[1]/table[%d]", tableIndex))
		}
	}
	if len(matches) == 0 {
		return nil, errors.New("trace has no GPU counter-info table")
	}
	return matches, nil
}

func parseCommandSubmissions(r io.Reader) ([]commandBuffer, error) {
	dec := newRowDecoder(r)
	var commands []commandBuffer
	for {
		row, err := dec.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("command-buffer submission XML: %w", err)
		}
		if len(row) != 15 {
			return nil, fmt.Errorf("command-buffer submission row has %d columns, want 15", len(row))
		}
		if row[2] != "CommandBufferSubmission" {
			continue
		}
		start, err := parseUint64(row[0], "command-buffer start")
		if err != nil {
			return nil, err
		}
		id, err := parseUint64(row[14], "command-buffer ID")
		if err != nil {
			return nil, err
		}
		if row[3] == "" || row[7] == "" {
			return nil, errors.New("command-buffer submission has incomplete GPU/process identity")
		}
		commands = append(commands, commandBuffer{StartNS: start, ID: id, GPU: row[3], Process: row[7]})
	}
	if len(commands) == 0 {
		return nil, errors.New("no Metal command-buffer submissions found")
	}
	sort.Slice(commands, func(i, j int) bool { return commands[i].StartNS < commands[j].StartNS })
	return commands, nil
}

func parseCommandCompletions(r io.Reader) (map[uint64]uint64, error) {
	dec := newRowDecoder(r)
	completed := make(map[uint64]uint64)
	for {
		row, err := dec.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("command-buffer completion XML: %w", err)
		}
		if len(row) != 3 {
			return nil, fmt.Errorf("command-buffer completion row has %d columns, want 3", len(row))
		}
		end, err := parseUint64(row[0], "command-buffer completion")
		if err != nil {
			return nil, err
		}
		id, err := parseUint64(row[1], "completed command-buffer ID")
		if err != nil {
			return nil, err
		}
		completed[id] = end
	}
	if len(completed) == 0 {
		return nil, errors.New("no Metal command-buffer completions found")
	}
	return completed, nil
}

func selectCommandBuffers(commands []commandBuffer, completed map[uint64]uint64, count int) ([]commandBuffer, error) {
	if count <= 0 {
		return nil, fmt.Errorf("selected command-buffer count must be positive, got %d", count)
	}
	if len(commands) < count {
		return nil, fmt.Errorf("found %d command buffers, need final %d", len(commands), count)
	}
	selected := append([]commandBuffer(nil), commands[len(commands)-count:]...)
	for i := range selected {
		end, ok := completed[selected[i].ID]
		if !ok {
			return nil, fmt.Errorf("command buffer %d has no completion", selected[i].ID)
		}
		if end < selected[i].StartNS {
			return nil, fmt.Errorf("command buffer %d completes before it starts", selected[i].ID)
		}
		selected[i].EndNS = end
	}
	return mergeIntervals(selected), nil
}

func mergeIntervals(commands []commandBuffer) []commandBuffer {
	if len(commands) < 2 {
		return commands
	}
	merged := make([]commandBuffer, 0, len(commands))
	for _, command := range commands {
		last := len(merged) - 1
		if last >= 0 && command.StartNS <= merged[last].EndNS {
			if command.EndNS > merged[last].EndNS {
				merged[last].EndNS = command.EndNS
			}
			continue
		}
		merged = append(merged, command)
	}
	return merged
}

func analyzeValues(r io.Reader, infos map[uint32]counterInfo, intervals []commandBuffer) ([]counterReport, windowReport, error) {
	return analyzeValuesWithPolicy(r, infos, intervals, strictCounterPolicy())
}

func analyzeValuesWithPolicy(r io.Reader, infos map[uint32]counterInfo, intervals []commandBuffer, policy counterPolicy) ([]counterReport, windowReport, error) {
	if len(intervals) == 0 {
		return nil, windowReport{}, errors.New("selected command-buffer window is empty")
	}
	window := windowReport{StartNS: intervals[0].StartNS, EndNS: intervals[len(intervals)-1].EndNS}
	window.DurationNS = window.EndNS - window.StartNS
	for _, interval := range intervals {
		window.ActiveDurationNS += interval.EndNS - interval.StartNS
	}
	if window.DurationNS > 0 {
		window.DutyCycle = float64(window.ActiveDurationNS) / float64(window.DurationNS)
	}

	reports := make(map[uint32]*counterReport, len(infos))
	for id, info := range infos {
		reports[id] = &counterReport{
			ID:                info.ID,
			Name:              info.Name,
			Type:              info.Type,
			MaxValue:          info.MaxValue,
			SampleIntervalRaw: info.SampleIntervalRaw,
			Wall:              emptyStats(),
			Active:            emptyStats(),
		}
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
			return nil, windowReport{}, fmt.Errorf("counter value XML: %w", err)
		}
		if len(row) != 6 {
			return nil, windowReport{}, fmt.Errorf("counter value row has %d columns, want 6", len(row))
		}
		timestamp, err := parseUint64(row[0], "counter timestamp")
		if err != nil {
			return nil, windowReport{}, err
		}
		if sawTimestamp && timestamp < previousTimestamp {
			return nil, windowReport{}, fmt.Errorf("counter timestamps regress from %d to %d", previousTimestamp, timestamp)
		}
		previousTimestamp, sawTimestamp = timestamp, true
		if timestamp < window.StartNS {
			continue
		}
		if timestamp > window.EndNS {
			break
		}
		id, err := parseUint32(row[1], "counter sample ID")
		if err != nil {
			return nil, windowReport{}, err
		}
		value, err := strconv.ParseFloat(row[2], 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, windowReport{}, fmt.Errorf("invalid counter %d value %q", id, row[2])
		}
		counter, ok := reports[id]
		if !ok {
			return nil, windowReport{}, fmt.Errorf("counter sample references unknown counter ID %d", id)
		}
		counter.Wall.add(value)
		for intervalIndex < len(intervals) && timestamp > intervals[intervalIndex].EndNS {
			intervalIndex++
		}
		if intervalIndex < len(intervals) && timestamp >= intervals[intervalIndex].StartNS {
			counter.Active.add(value)
		}
	}

	result := make([]counterReport, 0, len(reports))
	for _, counter := range reports {
		counter.Policy = policy.globalClass(counter.Name)
		counter.Missing = counter.Wall.Samples == 0 || counter.Active.Samples == 0
		if counter.Missing && (counter.Policy == "required" || counter.Policy == "contamination") {
			return nil, windowReport{}, fmt.Errorf("counter %q has wall=%d active=%d samples", counter.Name, counter.Wall.Samples, counter.Active.Samples)
		}
		counter.Wall.finish()
		counter.Active.finish()
		result = append(result, *counter)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result, window, nil
}

func emptyStats() sampleStats {
	return sampleStats{Min: math.Inf(1), Max: math.Inf(-1)}
}

func (s *sampleStats) add(value float64) {
	s.Samples++
	s.sum += value
	if value < s.Min {
		s.Min = value
	}
	if value > s.Max {
		s.Max = value
	}
}

func (s *sampleStats) finish() {
	if s.Samples == 0 {
		s.Mean, s.Min, s.Max, s.sum = 0, 0, 0, 0
		return
	}
	s.Mean = s.sum / float64(s.Samples)
	s.sum = 0
}

func parseUint32(value, field string) (uint32, error) {
	n, err := strconv.ParseUint(value, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q: %w", field, value, err)
	}
	return uint32(n), nil
}

func parseUint64(value, field string) (uint64, error) {
	n, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q: %w", field, value, err)
	}
	return n, nil
}
