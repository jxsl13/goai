package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const defaultManifest = "internal/benchcompare/leadership/m2.json"

var (
	commitRevision  = regexp.MustCompile(`^[0-9a-f]{7,40}$`)
	releaseRevision = regexp.MustCompile(`^v?[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][0-9A-Za-z.-]+)?$`)
	buildRevision   = regexp.MustCompile(`^b[0-9]+$`)
)

type matrix struct {
	Schema   int      `json:"schema"`
	Protocol protocol `json:"protocol"`
	Cells    []cell   `json:"cells"`
}

type protocol struct {
	MinimumSamples int    `json:"minimum_samples"`
	Interleaved    bool   `json:"interleaved"`
	Prebuilt       bool   `json:"prebuilt"`
	Statistics     string `json:"statistics"`
}

type cell struct {
	ID                 string         `json:"id"`
	Status             string         `json:"status"`
	Hardware           string         `json:"hardware"`
	OS                 string         `json:"os"`
	Accelerator        string         `json:"accelerator"`
	Workload           string         `json:"workload"`
	Shape              string         `json:"shape"`
	DtypeQuantization  string         `json:"dtype_quantization"`
	BatchContext       string         `json:"batch_context"`
	State              string         `json:"state"`
	WorkspaceTransfers string         `json:"workspace_transfers"`
	Quality            string         `json:"quality"`
	Metric             string         `json:"metric"`
	GoAI               implementation `json:"goai"`
	Incumbent          implementation `json:"incumbent"`
	GoAIResult         string         `json:"goai_result"`
	IncumbentResult    string         `json:"incumbent_result"`
	Verdict            string         `json:"verdict"`
	Samples            int            `json:"samples"`
	Interleaved        bool           `json:"interleaved"`
	Prebuilt           bool           `json:"prebuilt"`
	Statistics         string         `json:"statistics"`
	Reproduce          []string       `json:"reproduce"`
	Evidence           []string       `json:"evidence"`
	Blockers           []string       `json:"blockers"`
}

type implementation struct {
	Name       string `json:"name"`
	Revision   string `json:"revision"`
	Runtime    string `json:"runtime"`
	Threads    int    `json:"threads"`
	SourceHash string `json:"source_hash,omitempty"`
}

type validationReport struct {
	Errors   []string
	Warnings []string
}

func loadMatrix(path string) (matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return matrix{}, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var m matrix
	if err := dec.Decode(&m); err != nil {
		return matrix{}, fmt.Errorf("decode %s: %w", path, err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return matrix{}, fmt.Errorf("decode %s: multiple JSON values", path)
		}
		return matrix{}, fmt.Errorf("decode %s: %w", path, err)
	}
	return m, nil
}

func validateMatrix(m matrix) validationReport {
	var out validationReport
	if m.Schema != 1 {
		out.Errors = append(out.Errors, fmt.Sprintf("schema: got %d, want 1", m.Schema))
	}
	if m.Protocol.MinimumSamples < 10 {
		out.Errors = append(out.Errors, "protocol.minimum_samples: must be at least 10")
	}
	if !m.Protocol.Interleaved {
		out.Errors = append(out.Errors, "protocol.interleaved: must be true")
	}
	if !m.Protocol.Prebuilt {
		out.Errors = append(out.Errors, "protocol.prebuilt: must be true")
	}
	if strings.TrimSpace(m.Protocol.Statistics) == "" {
		out.Errors = append(out.Errors, "protocol.statistics: required")
	}
	seen := make(map[string]bool, len(m.Cells))
	for i := range m.Cells {
		c := &m.Cells[i]
		prefix := "cells[" + strconv.Itoa(i) + "]"
		if c.ID == "" {
			out.Errors = append(out.Errors, prefix+".id: required")
			continue
		}
		prefix = c.ID
		if seen[c.ID] {
			out.Errors = append(out.Errors, prefix+": duplicate id")
		}
		seen[c.ID] = true
		if c.Status != "published" && c.Status != "provisional" && c.Status != "open" {
			out.Errors = append(out.Errors, prefix+".status: must be published, provisional, or open")
			continue
		}
		for name, value := range map[string]string{
			"hardware": c.Hardware, "os": c.OS, "workload": c.Workload,
			"dtype_quantization": c.DtypeQuantization, "batch_context": c.BatchContext,
			"state": c.State, "workspace_transfers": c.WorkspaceTransfers,
		} {
			if strings.TrimSpace(value) == "" {
				out.Errors = append(out.Errors, prefix+"."+name+": required")
			}
		}
		if c.Status == "open" {
			if len(c.Blockers) == 0 {
				out.Errors = append(out.Errors, prefix+".blockers: open cell requires a blocker")
			}
			continue
		}
		for name, value := range map[string]string{
			"shape": c.Shape, "quality": c.Quality, "metric": c.Metric,
			"goai.name": c.GoAI.Name, "goai.revision": c.GoAI.Revision,
			"goai.runtime": c.GoAI.Runtime, "incumbent.name": c.Incumbent.Name,
			"incumbent.revision": c.Incumbent.Revision, "incumbent.runtime": c.Incumbent.Runtime,
			"goai_result": c.GoAIResult, "incumbent_result": c.IncumbentResult,
			"verdict": c.Verdict, "statistics": c.Statistics,
		} {
			if strings.TrimSpace(value) == "" {
				out.Errors = append(out.Errors, prefix+"."+name+": required")
			}
		}
		if len(c.Reproduce) == 0 {
			out.Errors = append(out.Errors, prefix+".reproduce: required")
		}
		if len(c.Evidence) == 0 {
			out.Errors = append(out.Errors, prefix+".evidence: required")
		}
		problems := publicationProblems(m.Protocol, c)
		if c.Status == "published" {
			for _, problem := range problems {
				out.Errors = append(out.Errors, prefix+": "+problem)
			}
			if len(c.Blockers) != 0 {
				out.Errors = append(out.Errors, prefix+".blockers: published cell cannot have blockers")
			}
		} else {
			if len(c.Blockers) == 0 {
				out.Errors = append(out.Errors, prefix+".blockers: provisional cell requires a blocker")
			}
			for _, problem := range problems {
				out.Warnings = append(out.Warnings, prefix+": "+problem)
			}
		}
	}
	slices.Sort(out.Errors)
	slices.Sort(out.Warnings)
	return out
}

func publicationProblems(p protocol, c *cell) []string {
	var out []string
	if c.Samples < p.MinimumSamples {
		out = append(out, fmt.Sprintf("samples=%d, need >=%d", c.Samples, p.MinimumSamples))
	}
	if !c.Interleaved {
		out = append(out, "samples are not interleaved")
	}
	if !c.Prebuilt {
		out = append(out, "timed implementation is not prebuilt")
	}
	if !immutableRevision(c.GoAI.Revision) {
		out = append(out, "GoAI revision is not an immutable commit or release")
	}
	if !immutableRevision(c.Incumbent.Revision) {
		out = append(out, "incumbent revision is not an immutable commit or release")
	}
	return out
}

func immutableRevision(s string) bool {
	return commitRevision.MatchString(s) || releaseRevision.MatchString(s) || buildRevision.MatchString(s)
}

func renderMatrix(m matrix) string {
	var b strings.Builder
	b.WriteString("| Status | Cell | Hardware / OS | Workload dimensions | GoAI | Incumbent | Evidence / verdict |\n")
	b.WriteString("| --- | --- | --- | --- | --- | --- | --- |\n")
	for i := range m.Cells {
		c := &m.Cells[i]
		goai := implementationCell(c.GoAI, c.GoAIResult)
		incumbent := implementationCell(c.Incumbent, c.IncumbentResult)
		if c.Status == "open" {
			goai, incumbent = "—", "—"
		}
		dims := strings.Join(nonEmpty(
			c.Workload, c.Shape, c.DtypeQuantization, c.BatchContext, c.State,
			c.WorkspaceTransfers,
		), "; ")
		blockers := strings.Join(c.Blockers, "; ")
		var evidence strings.Builder
		evidence.Grow(len(c.Verdict) + len(blockers) + len("; blocker: "))
		evidence.WriteString(c.Verdict)
		if blockers != "" {
			if evidence.Len() != 0 {
				evidence.WriteString("; ")
			}
			evidence.WriteString("blocker: ")
			evidence.WriteString(blockers)
		}
		fmt.Fprintf(&b, "| %s | `%s` | %s | %s | %s | %s | %s |\n",
			md(c.Status), md(c.ID), md(c.Hardware+"; "+c.OS+"; "+c.Accelerator),
			md(dims), md(goai), md(incumbent), md(evidence.String()))
	}
	return b.String()
}

func nonEmpty(values ...string) []string {
	out := values[:0]
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			out = append(out, value)
		}
	}
	return out
}

func implementationCell(impl implementation, result string) string {
	if impl.Name == "" {
		return "—"
	}
	threads := ""
	if impl.Threads > 0 {
		threads = fmt.Sprintf(", threads=%d", impl.Threads)
	}
	return fmt.Sprintf("%s %s (%s%s): %s", impl.Name, impl.Revision, impl.Runtime, threads, result)
}

func md(s string) string {
	if strings.TrimSpace(s) == "" {
		return "—"
	}
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}
