package cpu_test

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

const (
	allocationDiagnosticEnv        = "GOAI_ALLOC_DIAGNOSTIC"
	allocationDiagnosticIterations = 1024
)

type allocationDiagnosticRecord struct {
	Benchmark       string `json:"benchmark"`
	Procs           int    `json:"procs"`
	N               int    `json:"n"`
	TotalNS         int64  `json:"total_ns"`
	TotalBytes      uint64 `json:"total_bytes"`
	TotalAllocs     uint64 `json:"total_allocs"`
	BytesPerOp      uint64 `json:"bytes_per_op"`
	BytesRemainder  uint64 `json:"bytes_remainder"`
	AllocsPerOp     uint64 `json:"allocs_per_op"`
	AllocsRemainder uint64 `json:"allocs_remainder"`
}

func allocationDiagnosticConfig(optIn, benchTime string, procs int) (bool, error) {
	if optIn != "1" {
		return false, nil
	}
	if benchTime != "1024x" {
		return false, fmt.Errorf("%s=1 requires -test.benchtime=1024x (got %q)", allocationDiagnosticEnv, benchTime)
	}
	if procs != 1 && procs != 12 {
		return false, fmt.Errorf("%s=1 requires GOMAXPROCS 1 or 12 (got %d)", allocationDiagnosticEnv, procs)
	}
	return true, nil
}

func allocationDiagnosticResult(name string, procs int, result testing.BenchmarkResult) (string, error) {
	if result.N != allocationDiagnosticIterations {
		return "", fmt.Errorf("%s: benchmark ran %d iterations, want %d", name, result.N, allocationDiagnosticIterations)
	}
	if result.T <= 0 {
		return "", fmt.Errorf("%s: benchmark duration must be positive (got %s)", name, result.T)
	}
	if _, ok := result.Extra["B/op"]; ok {
		return "", fmt.Errorf("%s: B/op metric override is not permitted", name)
	}
	if _, ok := result.Extra["allocs/op"]; ok {
		return "", fmt.Errorf("%s: allocs/op metric override is not permitted", name)
	}

	n := uint64(result.N)
	record := allocationDiagnosticRecord{
		Benchmark:       name,
		Procs:           procs,
		N:               result.N,
		TotalNS:         result.T.Nanoseconds(),
		TotalBytes:      result.MemBytes,
		TotalAllocs:     result.MemAllocs,
		BytesPerOp:      result.MemBytes / n,
		BytesRemainder:  result.MemBytes % n,
		AllocsPerOp:     result.MemAllocs / n,
		AllocsRemainder: result.MemAllocs % n,
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		return "", fmt.Errorf("encode %s allocation diagnostic: %w", name, err)
	}
	return "allocdiag: " + string(encoded), nil
}

func benchmarkWithFailure(f func(*testing.B)) (testing.BenchmarkResult, bool) {
	var failed atomic.Bool
	result := testing.Benchmark(func(b *testing.B) {
		defer func() {
			if b.Failed() {
				failed.Store(true)
			}
		}()
		f(b)
	})
	return result, failed.Load()
}

func runAllocationDiagnosticControls(
	enabled bool,
	procs int,
	run func(func(*testing.B)) (testing.BenchmarkResult, bool),
	emit func(string),
) error {
	if !enabled {
		return nil
	}
	controls := []struct {
		name string
		fn   func(*testing.B)
	}{
		{"BenchmarkSiLUBackwardF64_256K_cpu", BenchmarkSiLUBackwardF64_256K_cpu},
		{"BenchmarkSigmoidF64_64K_cpu", BenchmarkSigmoidF64_64K_cpu},
		{"BenchmarkSoftplusF64_256K_cpu", BenchmarkSoftplusF64_256K_cpu},
	}
	for _, control := range controls {
		result, failed := run(control.fn)
		if failed {
			return fmt.Errorf("%s failed", control.name)
		}
		line, err := allocationDiagnosticResult(control.name, procs, result)
		if err != nil {
			return err
		}
		emit(line)
	}
	return nil
}

func TestCPUControlAllocationDiagnostics(t *testing.T) {
	benchTimeFlag := flag.Lookup("test.benchtime")
	if benchTimeFlag == nil {
		t.Fatal("test.benchtime flag is not registered")
	}
	procs := runtime.GOMAXPROCS(0)
	enabled, err := allocationDiagnosticConfig(
		os.Getenv(allocationDiagnosticEnv),
		benchTimeFlag.Value.String(),
		procs,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !enabled {
		t.Skipf("set %s=1 with -test.benchtime=1024x and GOMAXPROCS=1 or 12 to run allocation diagnostics", allocationDiagnosticEnv)
	}

	if err := runAllocationDiagnosticControls(enabled, procs, benchmarkWithFailure, func(line string) {
		fmt.Println(line)
	}); err != nil {
		t.Fatal(err)
	}
	// testing.Benchmark performs an unreported one-iteration calibration run;
	// the records above contain only the subsequent fixed 1024-iteration run.
}

func TestAllocationDiagnosticConfig(t *testing.T) {
	tests := []struct {
		name      string
		optIn     string
		benchTime string
		procs     int
		enabled   bool
		wantErr   bool
	}{
		{"disabled", "", "1s", 8, false, false},
		{"valid-serial", "1", "1024x", 1, true, false},
		{"valid-parallel", "1", "1024x", 12, true, false},
		{"wrong-count", "1", "1023x", 1, false, true},
		{"wrong-procs", "1", "1024x", 2, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			enabled, err := allocationDiagnosticConfig(tt.optIn, tt.benchTime, tt.procs)
			if enabled != tt.enabled || (err != nil) != tt.wantErr {
				t.Fatalf("allocationDiagnosticConfig(%q, %q, %d) = (%v, %v), want enabled=%v error=%v", tt.optIn, tt.benchTime, tt.procs, enabled, err, tt.enabled, tt.wantErr)
			}
		})
	}
}

func TestAllocationDiagnosticResultExactArithmetic(t *testing.T) {
	const totalBytes = uint64(1<<53) + 1025
	const totalAllocs = uint64(1<<54) + 1027
	line, err := allocationDiagnosticResult("synthetic", 12, testing.BenchmarkResult{
		N:         allocationDiagnosticIterations,
		T:         123456789 * time.Nanosecond,
		MemBytes:  totalBytes,
		MemAllocs: totalAllocs,
	})
	if err != nil {
		t.Fatal(err)
	}
	const want = `allocdiag: {"benchmark":"synthetic","procs":12,"n":1024,"total_ns":123456789,"total_bytes":9007199254742017,"total_allocs":18014398509483011,"bytes_per_op":8796093022209,"bytes_remainder":1,"allocs_per_op":17592186044417,"allocs_remainder":3}`
	if line != want {
		t.Fatalf("diagnostic line:\n got %s\nwant %s", line, want)
	}
	var got allocationDiagnosticRecord
	if err := json.Unmarshal([]byte(line[len("allocdiag: "):]), &got); err != nil {
		t.Fatal(err)
	}
	if got.TotalBytes != totalBytes || got.BytesPerOp != totalBytes/1024 || got.BytesRemainder != totalBytes%1024 {
		t.Fatalf("byte arithmetic lost precision: %+v", got)
	}
	if got.TotalAllocs != totalAllocs || got.AllocsPerOp != totalAllocs/1024 || got.AllocsRemainder != totalAllocs%1024 {
		t.Fatalf("allocation arithmetic lost precision: %+v", got)
	}
}

func TestAllocationDiagnosticResultRejectsInvalidResults(t *testing.T) {
	valid := testing.BenchmarkResult{N: 1024, T: time.Nanosecond}
	tests := []struct {
		name   string
		mutate func(*testing.BenchmarkResult)
	}{
		{"invalid-N", func(r *testing.BenchmarkResult) { r.N = 1 }},
		{"invalid-time", func(r *testing.BenchmarkResult) { r.T = 0 }},
		{"bytes-override", func(r *testing.BenchmarkResult) { r.Extra = map[string]float64{"B/op": 1} }},
		{"allocs-override", func(r *testing.BenchmarkResult) { r.Extra = map[string]float64{"allocs/op": 1} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := valid
			tt.mutate(&result)
			if _, err := allocationDiagnosticResult("synthetic", 1, result); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestAllocationDiagnosticDisabledRunsNoBenchmarks(t *testing.T) {
	called := false
	err := runAllocationDiagnosticControls(false, 1, func(func(*testing.B)) (testing.BenchmarkResult, bool) {
		called = true
		return testing.BenchmarkResult{}, false
	}, func(string) {
		called = true
	})
	if err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("disabled diagnostic invoked a benchmark or emitted output")
	}
}

func TestBenchmarkWithFailurePropagatesNestedFatal(t *testing.T) {
	_, failed := benchmarkWithFailure(func(b *testing.B) {
		b.Fatal(errors.New("synthetic nested benchmark failure"))
	})
	if !failed {
		t.Fatal("nested b.Fatal was not propagated")
	}
}
