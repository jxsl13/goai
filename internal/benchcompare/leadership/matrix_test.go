package main

import (
	"strings"
	"testing"
)

func validPublishedCell() cell {
	return cell{
		ID: "m2-test", Status: "published", Hardware: "Apple M2 Pro 32 GiB",
		OS: "macOS 26.5.1 arm64", Accelerator: "CPU", Workload: "SiLU",
		Shape: "[256,1408]", DtypeQuantization: "F64", BatchContext: "batch 256",
		State: "warm", WorkspaceTransfers: "caller-owned input and output; no transfers",
		Quality: "parity test", Metric: "ns/op lower is better",
		GoAI:       implementation{Name: "GoAI", Revision: "abcdef1", Runtime: "go1.26.6", Threads: 12},
		Incumbent:  implementation{Name: "PyTorch", Revision: "v2.12.1", Runtime: "python3.14", Threads: 8},
		GoAIResult: "100 ns/op", IncumbentResult: "150 ns/op", Verdict: "GoAI 1.5x",
		Samples: 10, Interleaved: true, Prebuilt: true, Statistics: "benchstat 95% CI",
		Reproduce: []string{"go run ./internal/benchcompare/leadership collect-silu"},
		Evidence:  []string{"evidence/benchstat.txt"},
	}
}

func TestValidatePublishedCell(t *testing.T) {
	m := matrix{Schema: 1, Protocol: protocol{MinimumSamples: 10, Interleaved: true, Prebuilt: true, Statistics: "benchstat"}, Cells: []cell{validPublishedCell()}}
	report := validateMatrix(m)
	if len(report.Errors) != 0 || len(report.Warnings) != 0 {
		t.Fatalf("unexpected report: errors=%v warnings=%v", report.Errors, report.Warnings)
	}
}

func TestValidatePublishedRejectsWeakEvidence(t *testing.T) {
	c := validPublishedCell()
	c.Samples = 3
	c.GoAI.Revision = "workspace-uncommitted"
	m := matrix{Schema: 1, Protocol: protocol{MinimumSamples: 10, Interleaved: true, Prebuilt: true, Statistics: "benchstat"}, Cells: []cell{c}}
	report := validateMatrix(m)
	joined := strings.Join(report.Errors, "\n")
	if !strings.Contains(joined, "samples=3") || !strings.Contains(joined, "GoAI revision") {
		t.Fatalf("missing publication failures: %v", report.Errors)
	}
}

func TestValidateProvisionalMakesWeakEvidenceVisible(t *testing.T) {
	c := validPublishedCell()
	c.Status = "provisional"
	c.Samples = 3
	c.GoAI.Revision = "workspace-uncommitted"
	c.Blockers = []string{"Git commit unavailable", "need 10 samples"}
	m := matrix{Schema: 1, Protocol: protocol{MinimumSamples: 10, Interleaved: true, Prebuilt: true, Statistics: "benchstat"}, Cells: []cell{c}}
	report := validateMatrix(m)
	if len(report.Errors) != 0 || len(report.Warnings) != 2 {
		t.Fatalf("unexpected report: errors=%v warnings=%v", report.Errors, report.Warnings)
	}
}

func TestRenderMatrixLabelsBlocker(t *testing.T) {
	c := validPublishedCell()
	c.Status = "provisional"
	c.Blockers = []string{"source pin pending"}
	m := matrix{Cells: []cell{c}}
	got := renderMatrix(m)
	if !strings.Contains(got, "provisional") || !strings.Contains(got, "source pin pending") || !strings.Contains(got, "Apple M2 Pro") {
		t.Fatalf("render omitted evidence boundary:\n%s", got)
	}
}

func TestRenderOpenMatrixDoesNotLeadWithSeparator(t *testing.T) {
	c := validPublishedCell()
	c.Status = "open"
	c.Verdict = ""
	c.Blockers = []string{"measurement pending"}
	m := matrix{Cells: []cell{c}}
	got := renderMatrix(m)
	if strings.Contains(got, "; blocker: measurement pending") ||
		!strings.Contains(got, "blocker: measurement pending") {
		t.Fatalf("open blocker formatting is ambiguous:\n%s", got)
	}
}

func TestMedian(t *testing.T) {
	if got := median([]float64{9, 1, 3, 7}); got != 5 {
		t.Fatalf("median=%v want 5", got)
	}
}
