package main

import (
	"strings"
	"testing"
)

func TestAnalyzeShaderIntervalsFiltersAndRanksTargetCommand(t *testing.T) {
	row := func(start, duration, name, process, gpu string) string {
		values := []string{start, duration, name, "", name, "Shader Timeline", "Compute", "50", "", "1001", "1", process, gpu, "Compute", "0", "7"}
		var out strings.Builder
		out.WriteString("<row>")
		for _, value := range values {
			out.WriteString("<v>")
			out.WriteString(value)
			out.WriteString("</v>")
		}
		out.WriteString("</row>")
		return out.String()
	}
	xml := `<trace-query-result>` +
		row("70", "20", "warmup (1)", "llama-bench (42)", "M2 Pro") +
		row("110", "20", "kernel_mul_mv_q4_K_f32 (10)", "llama-bench (42)", "M2 Pro") +
		row("140", "30", "kernel_mul_mv_q4_K_f32 (10)", "llama-bench (42)", "M2 Pro") +
		row("170", "30", "kernel_rms_norm (9)", "llama-bench (42)", "M2 Pro") +
		row("130", "50", "foreign (4)", "WindowServer (7)", "M2 Pro") +
		row("130", "50", "wrong_gpu (4)", "llama-bench (42)", "Other") +
		`</trace-query-result>`
	command := commandBuffer{ID: 9, Process: "llama-bench (42)", GPU: "M2 Pro"}
	gpu := []gpuInterval{{StartNS: 100, EndNS: 200}}
	report, err := analyzeShaderIntervals(strings.NewReader(xml), command, gpu, gpuOverlapReport{})
	if err != nil {
		t.Fatal(err)
	}
	if report.SampleCount != 3 || report.SampleDurationNS != 80 || report.GPUSpanNS != 100 || report.SampledSpanRatio != 0.8 {
		t.Fatalf("report=%+v", report)
	}
	if len(report.Kernels) != 2 || report.Kernels[0].Name != "kernel_mul_mv_q4_K_f32" ||
		report.Kernels[0].Samples != 2 || report.Kernels[0].SampleDurationNS != 50 ||
		report.Kernels[0].MedianNS != 20 || report.Kernels[0].P90NS != 30 ||
		report.Kernels[1].Name != "kernel_rms_norm" || report.Kernels[1].SampleDurationNS != 30 {
		t.Fatalf("kernels=%+v", report.Kernels)
	}
}

func TestAnalyzeShaderIntervalsFailsClosed(t *testing.T) {
	command := commandBuffer{ID: 9, Process: "llama-bench (42)", GPU: "M2 Pro"}
	gpu := []gpuInterval{{StartNS: 100, EndNS: 200}}
	for _, xml := range []string{
		`<trace-query-result></trace-query-result>`,
		`<trace-query-result><row><v>1</v></row></trace-query-result>`,
	} {
		if _, err := analyzeShaderIntervals(strings.NewReader(xml), command, gpu, gpuOverlapReport{}); err == nil {
			t.Fatalf("XML unexpectedly accepted: %s", xml)
		}
	}
}

func TestCanonicalShaderName(t *testing.T) {
	for input, want := range map[string]string{
		"kernel_mul_mv_q4_K_f32 (10)": "kernel_mul_mv_q4_K_f32",
		"kernel (special)":            "kernel (special)",
		"kernel ()":                   "kernel ()",
		"plain":                       "plain",
	} {
		if got := canonicalShaderName(input); got != want {
			t.Fatalf("canonicalShaderName(%q)=%q want %q", input, got, want)
		}
	}
}

func TestDecodeLaunchCommand(t *testing.T) {
	got, err := decodeLaunchCommand(`["/opt/homebrew/bin/llama-bench","-p","1"]`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != "/opt/homebrew/bin/llama-bench" || got[2] != "1" {
		t.Fatalf("command=%q", got)
	}
	for _, value := range []string{`[]`, `["llama-bench"]`, `["/bin/tool"] {}`, `{}`} {
		if _, err := decodeLaunchCommand(value); err == nil {
			t.Fatalf("decodeLaunchCommand(%q) unexpectedly succeeded", value)
		}
	}
}
