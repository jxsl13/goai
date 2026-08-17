package main

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseCounterInfoResolvesReferences(t *testing.T) {
	var rows strings.Builder
	names := append([]string(nil), requiredLimiterNames...)
	names = append(names, "F32 Utilization")
	for i, name := range names {
		if i == 0 {
			rows.WriteString(`<row><event-time id="t">0</event-time><uint32 id="zero">0</uint32><gpu-counter-name>` + name + `</gpu-counter-name><uint64 id="max">100</uint64><uint64>1</uint64><string>desc</string><uint32 ref="zero"/><string id="type">Percentage</string><uint32>2</uint32><boolean>1</boolean><uint32 id="sample">10000</uint32></row>`)
			continue
		}
		rows.WriteString(`<row><event-time ref="t"/><uint32>` + stringInt(i) + `</uint32><gpu-counter-name>` + name + `</gpu-counter-name><uint64 ref="max"/><uint64>1</uint64><string>desc</string><uint32 ref="zero"/><string ref="type"/><uint32>2</uint32><boolean>1</boolean><uint32 ref="sample"/></row>`)
	}
	infos, err := parseCounterInfo(strings.NewReader(`<trace-query-result>` + rows.String() + `</trace-query-result>`))
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != len(names) || infos[0].SampleIntervalRaw != 10000 || infos[1].Type != "Percentage" {
		t.Fatalf("unexpected counter metadata: %+v", infos)
	}
}

func TestParseCounterInfoRejectsUnresolvedReference(t *testing.T) {
	_, err := parseCounterInfo(strings.NewReader(`<trace-query-result><row><event-time ref="missing"/></row></trace-query-result>`))
	if err == nil || !strings.Contains(err.Error(), "unresolved XML reference") {
		t.Fatalf("err=%v, want unresolved reference", err)
	}
}

func TestRowDecoderResolvesIDDefinedInsideCompositeElement(t *testing.T) {
	xml := `<trace-query-result>` +
		`<row><formatted-label id="outer"><process id="p"><pid>42</pid></process></formatted-label><process ref="p"/></row>` +
		`</trace-query-result>`
	row, err := newRowDecoder(strings.NewReader(xml)).next()
	if err != nil {
		t.Fatal(err)
	}
	if len(row) != 2 || row[0] != "42" || row[1] != "42" {
		t.Fatalf("row=%q", row)
	}
}

func TestCounterInfoTableXPathsReturnsEveryCandidate(t *testing.T) {
	toc := `<trace-toc><run><data>` +
		`<table schema="gpu-counter-info" counter-profile="0" shader-profiler="0"/>` +
		`<table schema="process-info"/>` +
		`<table schema="gpu-counter-info" counter-profile="7" shader-profiler="1"/>` +
		`</data></run></trace-toc>`
	xpaths, err := counterInfoTableXPaths(strings.NewReader(toc))
	if err != nil {
		t.Fatal(err)
	}
	if len(xpaths) != 2 || xpaths[0] != "//trace-toc[1]/run[1]/data[1]/table[1]" || xpaths[1] != "//trace-toc[1]/run[1]/data[1]/table[3]" {
		t.Fatalf("xpaths=%q", xpaths)
	}
}

func TestCounterInfoTableXPathsRejectsMissingTable(t *testing.T) {
	_, err := counterInfoTableXPaths(strings.NewReader(`<trace-toc><run><data><table schema="process-info"/></data></run></trace-toc>`))
	if err == nil || !strings.Contains(err.Error(), "no GPU counter-info") {
		t.Fatalf("err=%v", err)
	}
}

func TestSelectAndAnalyzeFinalCommandBuffers(t *testing.T) {
	commands := []commandBuffer{
		{StartNS: 100, ID: 1, GPU: "M2 Pro"},
		{StartNS: 300, ID: 2, GPU: "M2 Pro"},
		{StartNS: 500, ID: 3, GPU: "M2 Pro"},
	}
	intervals, err := selectCommandBuffers(commands, map[uint64]uint64{1: 200, 2: 400, 3: 650}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(intervals) != 2 || intervals[0].StartNS != 300 || intervals[1].EndNS != 650 {
		t.Fatalf("intervals=%+v", intervals)
	}
	infos := make(map[uint32]counterInfo)
	for i, name := range requiredLimiterNames {
		infos[uint32(i)] = counterInfo{ID: uint32(i), Name: name, Type: "Percentage", MaxValue: 100, SampleIntervalRaw: 10}
	}
	var rows strings.Builder
	for _, timestamp := range []int{250, 300, 450, 550, 700} {
		for id := range requiredLimiterNames {
			rows.WriteString(`<row><event-time>` + stringInt(timestamp) + `</event-time><uint32>` + stringInt(id) + `</uint32><fixed-decimal>` + stringInt(timestamp+id) + `</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>`)
		}
	}
	counters, window, err := analyzeValues(strings.NewReader(`<trace-query-result>`+rows.String()+`</trace-query-result>`), infos, intervals)
	if err != nil {
		t.Fatal(err)
	}
	if window.DurationNS != 350 || window.ActiveDurationNS != 250 || len(counters) != len(requiredLimiterNames) {
		t.Fatalf("window=%+v counters=%d", window, len(counters))
	}
	if got := counters[0].Wall; got.Samples != 3 || got.Mean != (300+450+550)/3.0 {
		t.Fatalf("wall=%+v", got)
	}
	if got := counters[0].Active; got.Samples != 2 || got.Mean != (300+550)/2.0 {
		t.Fatalf("active=%+v", got)
	}
}

func TestSelectCommandBuffersFailsClosed(t *testing.T) {
	commands := []commandBuffer{{StartNS: 10, ID: 1}}
	if _, err := selectCommandBuffers(commands, map[uint64]uint64{1: 20}, 2); err == nil {
		t.Fatal("expected too-few-command-buffers error")
	}
	if _, err := selectCommandBuffers(commands, map[uint64]uint64{}, 1); err == nil {
		t.Fatal("expected missing-completion error")
	}
}

func TestCompletedFixedBenchmark(t *testing.T) {
	output := "goos: darwin\nBenchmarkMetalQ4KDecodeLeaf/K2048N5632/cooperative-10\t10000\t250000 ns/op\nPASS\n"
	if !completedFixedBenchmark(output, 10000) {
		t.Fatal("completed benchmark not recognized")
	}
	if completedFixedBenchmark(output, 9999) {
		t.Fatal("wrong iteration count accepted")
	}
}

func TestCompileBenchmarkArgs(t *testing.T) {
	got := compileBenchmarkArgs(options{
		packagePath: "./internal/benchcompare",
		buildTags:   " vulkan,profile ",
	}, "/tmp/full-model.test")
	want := []string{"test", "-c", "-o", "/tmp/full-model.test", "-tags", "vulkan,profile", "./internal/benchcompare"}
	if len(got) != len(want) {
		t.Fatalf("args=%q want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("args[%d]=%q want %q; all=%q", i, got[i], want[i], got)
		}
	}

	got = compileBenchmarkArgs(options{packagePath: "./backend/metal"}, "metal.test")
	want = []string{"test", "-c", "-o", "metal.test", "./backend/metal"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("default args[%d]=%q want %q; all=%q", i, got[i], want[i], got)
		}
	}
}

func TestForwardedEnvArgs(t *testing.T) {
	t.Setenv("GOAI_MODEL", "/tmp/model.gguf")
	t.Setenv("GOAI_MODE", "profile")
	got, err := forwardedEnvArgs(" GOAI_MODEL,GOAI_MODE ")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--env", "GOAI_MODEL=/tmp/model.gguf", "--env", "GOAI_MODE=profile"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("args[%d]=%q want %q; all=%q", i, got[i], want[i], got)
		}
	}
	for _, value := range []string{"MISSING", "BAD-NAME", "GOAI_MODEL,GOAI_MODEL"} {
		if _, err := forwardedEnvArgs(value); err == nil {
			t.Fatalf("forwardedEnvArgs(%q) unexpectedly succeeded", value)
		}
	}
}

func TestRunRequiresStageProfileForExclusiveGPUWindow(t *testing.T) {
	err := run(context.Background(), options{
		packagePath:         "./backend/metal",
		iterations:          1,
		buffersPerIteration: 1,
		requiredCounters:    "*",
		stageRequired:       "*",
		requireExclusiveGPU: true,
	}, io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "requires stage-profile") {
		t.Fatalf("err=%v, want stage-profile requirement", err)
	}
}

func TestResolveAnalyzePathsDiscoversPerformanceLimiterMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "counter-info.xml"), []byte(`<trace-query-result></trace-query-result>`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "counter-info-0.xml"), []byte(`<trace-query-result></trace-query-result>`), 0o600); err != nil {
		t.Fatal(err)
	}
	var rows strings.Builder
	for i, name := range requiredLimiterNames {
		rows.WriteString(`<row><event-time>0</event-time><uint32>` + stringInt(i) + `</uint32><gpu-counter-name>` + name + `</gpu-counter-name><uint64>100</uint64><uint64>1</uint64><string>desc</string><uint32>0</uint32><string>Percentage</string><uint32>2</uint32><boolean>1</boolean><uint32>10000</uint32></row>`)
	}
	valid := `<trace-query-result>` + rows.String() + `</trace-query-result>`
	validPath := filepath.Join(dir, "counter-info-1.xml")
	if err := os.WriteFile(validPath, []byte(valid), 0o600); err != nil {
		t.Fatal(err)
	}
	paths, err := resolveAnalyzePaths(dir)
	if err != nil {
		t.Fatal(err)
	}
	if paths["counter-info"] != validPath {
		t.Fatalf("counter-info=%q want %q", paths["counter-info"], validPath)
	}
}

func TestParseCounterCeilings(t *testing.T) {
	got, err := parseCounterCeilings(" Fragment Occupancy=0.1,Texture Sample Limiter=0 ")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].name != "Fragment Occupancy" || got[0].max != 0.1 || got[1].name != "Texture Sample Limiter" || got[1].max != 0 {
		t.Fatalf("ceilings=%+v", got)
	}
	for _, value := range []string{
		"missing-separator",
		"=0.1",
		"Fragment Occupancy=",
		"Fragment Occupancy=bad",
		"Fragment Occupancy=-0.1",
		"Fragment Occupancy=NaN",
		"Fragment Occupancy=+Inf",
		"Fragment Occupancy=0.1,Fragment Occupancy=0.2",
	} {
		if _, err := parseCounterCeilings(value); err == nil {
			t.Fatalf("parseCounterCeilings(%q) unexpectedly succeeded", value)
		}
	}
}

func TestValidateCounterCeilings(t *testing.T) {
	report := report{Counters: []counterReport{
		{Name: "Fragment Occupancy", Active: sampleStats{Samples: 4, Mean: 0.1}},
		{Name: "Texture Sample Limiter", Active: sampleStats{Samples: 4, Mean: 0}},
	}}
	ceilings := []counterCeiling{{name: "Fragment Occupancy", max: 0.1}, {name: "Texture Sample Limiter", max: 0}}
	if err := validateCounterCeilings(report, ceilings); err != nil {
		t.Fatal(err)
	}

	report.Counters[0].Active.Mean = 0.100001
	if err := validateCounterCeilings(report, ceilings); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("err=%v, want ceiling error", err)
	}
	report.Counters[0].Active.Mean = 0.1
	if err := validateCounterCeilings(report, []counterCeiling{{name: "Missing", max: 0}}); err == nil || !strings.Contains(err.Error(), "absent") {
		t.Fatalf("err=%v, want absent-counter error", err)
	}
	report.Counters[0].Active.Samples = 0
	if err := validateCounterCeilings(report, ceilings); err == nil || !strings.Contains(err.Error(), "no active samples") {
		t.Fatalf("err=%v, want no-samples error", err)
	}
}

func TestParseCounterPolicy(t *testing.T) {
	policy, err := parseCounterPolicy(
		"Buffer Read Limiter,ALU Limiter",
		"GPU Read Bandwidth",
		"ALU Limiter",
		250_000,
		[]counterCeiling{{name: "Fragment Occupancy", max: 0.1}},
	)
	if err != nil {
		t.Fatal(err)
	}
	infos := map[uint32]counterInfo{
		0: {ID: 0, Name: "ALU Limiter"},
		1: {ID: 1, Name: "Buffer Read Limiter"},
		2: {ID: 2, Name: "GPU Read Bandwidth"},
	}
	if err := policy.validate(infos); err != nil {
		t.Fatal(err)
	}
	if got := policy.globalClass("ALU Limiter"); got != "required" {
		t.Fatalf("global class=%q, want required", got)
	}
	if got := policy.globalClass("GPU Read Bandwidth"); got != "capability" {
		t.Fatalf("global class=%q, want capability", got)
	}
	if got := policy.globalClass("Fragment Occupancy"); got != "contamination" {
		t.Fatalf("global class=%q, want contamination", got)
	}
	if got := policy.stageClass("ALU Limiter", 249_999); got != "sparse-stage" {
		t.Fatalf("short stage class=%q, want sparse-stage", got)
	}
	if got := policy.stageClass("ALU Limiter", 250_000); got != "required" {
		t.Fatalf("resolved stage class=%q, want required", got)
	}
	report := policy.report()
	if len(report.GlobalRequired) != 2 || report.GlobalRequired[0] != "ALU Limiter" ||
		len(report.Capability) != 1 || report.Capability[0] != "GPU Read Bandwidth" ||
		len(report.Contamination) != 1 || report.Contamination[0] != "Fragment Occupancy" ||
		report.StageMinDuration != 250_000 {
		t.Fatalf("policy report=%+v", report)
	}

	for _, test := range []struct {
		global, capability, stage string
	}{
		{global: "*", capability: "GPU Read Bandwidth", stage: "*"},
		{global: "ALU Limiter", capability: "ALU Limiter", stage: "ALU Limiter"},
		{global: "ALU Limiter,ALU Limiter", stage: "ALU Limiter"},
		{global: "ALU Limiter", capability: "*", stage: "ALU Limiter"},
	} {
		if _, err := parseCounterPolicy(test.global, test.capability, test.stage, 0, nil); err == nil {
			t.Fatalf("parseCounterPolicy(%q, %q, %q) unexpectedly succeeded", test.global, test.capability, test.stage)
		}
	}
	unknown, err := parseCounterPolicy("Unknown", "", "ALU Limiter", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := unknown.validate(infos); err == nil || !strings.Contains(err.Error(), "absent") {
		t.Fatalf("err=%v, want absent metadata rejection", err)
	}
}

func TestAnalyzeValuesPolicyRetainsOptionalMissingCounters(t *testing.T) {
	infos := map[uint32]counterInfo{
		0: {ID: 0, Name: "ALU Limiter", Type: "Percentage", MaxValue: 100, SampleIntervalRaw: 10},
		1: {ID: 1, Name: "GPU Read Bandwidth", Type: "Rate", SampleIntervalRaw: 10},
	}
	intervals := []commandBuffer{{StartNS: 100, EndNS: 200}}
	xml := `<trace-query-result>` +
		`<row><event-time>150</event-time><uint32>0</uint32><fixed-decimal>12.5</fixed-decimal><uint64>1</uint64><uint32>0</uint32><uint32>0</uint32></row>` +
		`</trace-query-result>`
	policy, err := parseCounterPolicy("ALU Limiter", "GPU Read Bandwidth", "ALU Limiter", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	counters, _, err := analyzeValuesWithPolicy(strings.NewReader(xml), infos, intervals, policy)
	if err != nil {
		t.Fatal(err)
	}
	if len(counters) != 2 || counters[0].Policy != "required" || counters[0].Missing ||
		counters[1].Policy != "capability" || !counters[1].Missing || counters[1].Active.Samples != 0 ||
		counters[1].Active.Mean != 0 || counters[1].Active.Min != 0 || counters[1].Active.Max != 0 {
		t.Fatalf("counters=%+v", counters)
	}
	encoded, err := json.Marshal(report{Version: reportVersion, Policy: policy.report(), Counters: counters})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"missing":true`) || !strings.Contains(string(encoded), `"samples":0`) {
		t.Fatalf("encoded report does not retain finite missingness: %s", encoded)
	}
	if _, _, err := analyzeValues(strings.NewReader(xml), infos, intervals); err == nil {
		t.Fatal("strict compatibility mode unexpectedly accepted a missing counter")
	}
	required, err := parseCounterPolicy("ALU Limiter,GPU Read Bandwidth", "", "ALU Limiter", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := analyzeValuesWithPolicy(strings.NewReader(xml), infos, intervals, required); err == nil || !strings.Contains(err.Error(), "GPU Read Bandwidth") {
		t.Fatalf("err=%v, want required-counter rejection", err)
	}
}

func stringInt(value int) string {
	const digits = "0123456789"
	if value == 0 {
		return "0"
	}
	var buf [32]byte
	i := len(buf)
	for value > 0 {
		i--
		buf[i] = digits[value%10]
		value /= 10
	}
	return string(buf[i:])
}
