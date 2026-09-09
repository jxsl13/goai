package cpu_test

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/pprof"
	"slices"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

const (
	allocationSiteSelector    = "softplus"
	allocationSiteSchema      = "goai-control-alloc-sites-v1"
	allocationSiteTailSchema  = "goai-control-alloc-sites-tail-v1"
	allocationSiteControl     = "SoftplusF64_256K"
	allocationSiteCalls       = 1024
	allocationSiteRawCapacity = 65536
	allocationSiteWorker      = "github.com/jxsl13/goai/backend/cpu.poolWorker"
)

type allocationSiteConfigInput struct {
	selector            string
	output              string
	godebug             string
	gogc                string
	gomemlimit          string
	gomaxprocs          int
	profileRate         int
	testRun             string
	testCount           string
	short               bool
	memprofileSupplied  bool
	profileRateSupplied bool
}

type allocationSiteConfig struct {
	enabled    bool
	output     string
	godebug    string
	gogc       string
	gomemlimit string
	gomaxprocs int
}

func validateAllocationSiteConfig(in allocationSiteConfigInput) (allocationSiteConfig, error) {
	if in.selector == "" {
		return allocationSiteConfig{}, nil
	}
	if in.selector != allocationSiteSelector {
		return allocationSiteConfig{}, fmt.Errorf("GOAI_ALLOC_SITE_DIAGNOSTIC must be %q, got %q", allocationSiteSelector, in.selector)
	}
	if in.output == "" || !filepath.IsAbs(in.output) {
		return allocationSiteConfig{}, fmt.Errorf("GOAI_ALLOC_SITE_OUT must be a nonempty absolute path, got %q", in.output)
	}
	if in.godebug != "memprofilerate=1" {
		return allocationSiteConfig{}, fmt.Errorf("GODEBUG must be exactly %q, got %q", "memprofilerate=1", in.godebug)
	}
	if in.gogc != "100" {
		return allocationSiteConfig{}, fmt.Errorf("GOGC must be exactly %q, got %q", "100", in.gogc)
	}
	if in.gomemlimit != "off" {
		return allocationSiteConfig{}, fmt.Errorf("GOMEMLIMIT must be exactly %q, got %q", "off", in.gomemlimit)
	}
	if in.gomaxprocs != 1 && in.gomaxprocs != 12 {
		return allocationSiteConfig{}, fmt.Errorf("GOMAXPROCS must be 1 or 12, got %d", in.gomaxprocs)
	}
	if in.profileRate != 1 {
		return allocationSiteConfig{}, fmt.Errorf("runtime.MemProfileRate must be 1, got %d", in.profileRate)
	}
	if in.testRun != "^TestCPUControlAllocationSites$" {
		return allocationSiteConfig{}, fmt.Errorf("-test.run must be exactly %q, got %q", "^TestCPUControlAllocationSites$", in.testRun)
	}
	if in.testCount != "1" {
		return allocationSiteConfig{}, fmt.Errorf("-test.count must be exactly %q, got %q", "1", in.testCount)
	}
	if in.short {
		return allocationSiteConfig{}, errors.New("enabled allocation-site capture rejects -test.short")
	}
	if in.memprofileSupplied {
		return allocationSiteConfig{}, errors.New("enabled allocation-site capture rejects an explicitly supplied -test.memprofile")
	}
	if in.profileRateSupplied {
		return allocationSiteConfig{}, errors.New("enabled allocation-site capture rejects an explicitly supplied -test.memprofilerate")
	}
	return allocationSiteConfig{
		enabled:    true,
		output:     in.output,
		godebug:    in.godebug,
		gogc:       in.gogc,
		gomemlimit: in.gomemlimit,
		gomaxprocs: in.gomaxprocs,
	}, nil
}

func currentAllocationSiteConfigInput(selector string, profileRate int) allocationSiteConfigInput {
	visited := make(map[string]bool)
	flag.Visit(func(f *flag.Flag) {
		visited[f.Name] = true
	})
	return allocationSiteConfigInput{
		selector:            selector,
		output:              os.Getenv("GOAI_ALLOC_SITE_OUT"),
		godebug:             os.Getenv("GODEBUG"),
		gogc:                os.Getenv("GOGC"),
		gomemlimit:          os.Getenv("GOMEMLIMIT"),
		gomaxprocs:          runtime.GOMAXPROCS(0),
		profileRate:         profileRate,
		testRun:             testFlagValue("test.run"),
		testCount:           testFlagValue("test.count"),
		short:               testing.Short(),
		memprofileSupplied:  visited["test.memprofile"],
		profileRateSupplied: visited["test.memprofilerate"],
	}
}

func testFlagValue(name string) string {
	f := flag.Lookup(name)
	if f == nil {
		return ""
	}
	return f.Value.String()
}

type allocationSiteArtifacts struct {
	capture *os.File
	tail    *os.File
	profile *os.File
}

func openAllocationSiteArtifacts(dir string) (allocationSiteArtifacts, error) {
	if dir == "" || !filepath.IsAbs(dir) {
		return allocationSiteArtifacts{}, fmt.Errorf("artifact directory must be a nonempty absolute path, got %q", dir)
	}
	if err := os.Mkdir(dir, 0o700); err != nil {
		return allocationSiteArtifacts{}, fmt.Errorf("create artifact directory %q: %w", dir, err)
	}

	var artifacts allocationSiteArtifacts
	opened := make([]io.Closer, 0, 3)
	open := func(name string) (*os.File, error) {
		f, err := os.OpenFile(filepath.Join(dir, name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err == nil {
			opened = append(opened, f)
		}
		return f, err
	}
	var err error
	if artifacts.capture, err = open("capture.json"); err == nil {
		artifacts.tail, err = open("tail.json")
	}
	if err == nil {
		artifacts.profile, err = open("allocs.pprof")
	}
	if err != nil {
		closeErr := closeAllocationSiteClosers(opened...)
		return allocationSiteArtifacts{}, errors.Join(fmt.Errorf("open allocation-site artifacts: %w", err), closeErr)
	}
	return artifacts, nil
}

func writeAllocationSiteJSON(w io.Writer, value any) error {
	b, err := json.Marshal(value)
	if err != nil {
		return err
	}
	n, err := w.Write(b)
	if err != nil {
		return err
	}
	if n != len(b) {
		return io.ErrShortWrite
	}
	return nil
}

func closeAllocationSiteClosers(closers ...io.Closer) error {
	var errs []error
	for _, closer := range closers {
		if closer == nil {
			continue
		}
		if err := closer.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

type allocationSiteCounters struct {
	TotalAlloc  uint64 `json:"total_alloc"`
	Mallocs     uint64 `json:"mallocs"`
	Frees       uint64 `json:"frees"`
	HeapAlloc   uint64 `json:"heap_alloc"`
	HeapObjects uint64 `json:"heap_objects"`
	NumGC       uint64 `json:"num_gc"`
}

func allocationSiteCountersFromMemStats(stats *runtime.MemStats) allocationSiteCounters {
	return allocationSiteCounters{
		TotalAlloc:  stats.TotalAlloc,
		Mallocs:     stats.Mallocs,
		Frees:       stats.Frees,
		HeapAlloc:   stats.HeapAlloc,
		HeapObjects: stats.HeapObjects,
		NumGC:       uint64(stats.NumGC),
	}
}

type allocationSiteFrame struct {
	PC       string `json:"pc"`
	Function string `json:"function"`
	File     string `json:"file"`
	Line     int    `json:"line"`
}

type allocationSiteRawRow struct {
	Ordinal      int                   `json:"ordinal"`
	AllocBytes   int64                 `json:"alloc_bytes"`
	FreeBytes    int64                 `json:"free_bytes"`
	AllocObjects int64                 `json:"alloc_objects"`
	FreeObjects  int64                 `json:"free_objects"`
	PCs          []string              `json:"pcs"`
	Frames       []allocationSiteFrame `json:"frames"`
}

type allocationSiteSnapshot struct {
	ReportedCount int                    `json:"reported_count"`
	OK            bool                   `json:"ok"`
	Rows          []allocationSiteRawRow `json:"rows"`
}

type allocationSiteFrameResolver func([]uintptr) []allocationSiteFrame

func runtimeAllocationSiteFrames(pcs []uintptr) []allocationSiteFrame {
	framesOut := make([]allocationSiteFrame, 0)
	if len(pcs) == 0 {
		return framesOut
	}
	frames := runtime.CallersFrames(pcs)
	for {
		frame, more := frames.Next()
		framesOut = append(framesOut, allocationSiteFrameFromRuntime(frame))
		if !more {
			break
		}
	}
	return framesOut
}

func allocationSiteFrameFromRuntime(frame runtime.Frame) allocationSiteFrame {
	return allocationSiteFrame{
		PC:       fmt.Sprintf("0x%x", frame.PC),
		Function: frame.Function,
		File:     frame.File,
		Line:     frame.Line,
	}
}

func formatAllocationSiteRawRow(ordinal int, record runtime.MemProfileRecord, resolve allocationSiteFrameResolver) allocationSiteRawRow {
	pcs := make([]string, len(record.Stack0))
	nonzero := make([]uintptr, 0, len(record.Stack0))
	seenZero := false
	for i, pc := range record.Stack0 {
		pcs[i] = fmt.Sprintf("0x%x", pc)
		if !seenZero && pc != 0 {
			nonzero = append(nonzero, pc)
		} else if pc == 0 {
			seenZero = true
		}
	}
	frames := resolve(nonzero)
	if frames == nil {
		frames = make([]allocationSiteFrame, 0)
	}
	return allocationSiteRawRow{
		Ordinal:      ordinal,
		AllocBytes:   record.AllocBytes,
		FreeBytes:    record.FreeBytes,
		AllocObjects: record.AllocObjects,
		FreeObjects:  record.FreeObjects,
		PCs:          pcs,
		Frames:       frames,
	}
}

func formatAllocationSiteSnapshot(n int, ok bool, records []runtime.MemProfileRecord, resolve allocationSiteFrameResolver) allocationSiteSnapshot {
	snapshot := allocationSiteSnapshot{
		ReportedCount: n,
		OK:            ok,
		Rows:          make([]allocationSiteRawRow, 0),
	}
	if !ok || n < 0 || n > len(records) {
		return snapshot
	}
	snapshot.Rows = make([]allocationSiteRawRow, 0, n)
	for i := range n {
		snapshot.Rows = append(snapshot.Rows, formatAllocationSiteRawRow(i, records[i], resolve))
	}
	return snapshot
}

type allocationSiteBoundaries struct {
	PreRawBefore allocationSiteCounters `json:"pre_raw_before"`
	Start        allocationSiteCounters `json:"start"`
	End          allocationSiteCounters `json:"end"`
	PostGC       allocationSiteCounters `json:"post_gc"`
	PostRaw      allocationSiteCounters `json:"post_raw"`
}

type allocationSiteCapture struct {
	Schema            string                   `json:"schema"`
	Control           string                   `json:"control"`
	N                 int                      `json:"n"`
	WarmupCalls       int                      `json:"warmup_calls"`
	CompletedCalls    int                      `json:"completed_calls"`
	GoVersion         string                   `json:"go_version"`
	GOOS              string                   `json:"goos"`
	GOARCH            string                   `json:"goarch"`
	GOMAXPROCS        int                      `json:"gomaxprocs"`
	GODEBUG           string                   `json:"godebug"`
	GOGC              string                   `json:"gogc"`
	GOMEMLIMIT        string                   `json:"gomemlimit"`
	ProfileRateBefore int                      `json:"profile_rate_before"`
	ProfileRateAfter  int                      `json:"profile_rate_after"`
	RawCapacity       int                      `json:"raw_capacity"`
	CallerFunction    string                   `json:"caller_function"`
	WorkerFunction    string                   `json:"worker_function"`
	Boundaries        allocationSiteBoundaries `json:"boundaries"`
	Pre               allocationSiteSnapshot   `json:"pre"`
	Post              allocationSiteSnapshot   `json:"post"`
	RegionError       string                   `json:"region_error"`
	Errors            []string                 `json:"errors"`
}

type allocationSiteTail struct {
	Schema              string                 `json:"schema"`
	Tail                allocationSiteCounters `json:"tail"`
	ReportWriteExcluded bool                   `json:"report_write_excluded"`
}

//go:noinline
func allocationSiteExecuteRegion(ctx *backend.Context, ins []*tensor.Tensor) (int, error) {
	completed := 0
	for range allocationSiteCalls {
		if _, err := backend.Execute(ctx, backend.OpSoftplus, ins, nil); err != nil {
			return completed, err
		}
		completed++
	}
	return completed, nil
}

func TestCPUControlAllocationSites(t *testing.T) {
	selector := os.Getenv("GOAI_ALLOC_SITE_DIAGNOSTIC")
	if selector == "" {
		t.Skip("GOAI_ALLOC_SITE_DIAGNOSTIC is empty")
	}

	profileRateBefore := runtime.MemProfileRate
	config, err := validateAllocationSiteConfig(currentAllocationSiteConfigInput(selector, profileRateBefore))
	if err != nil {
		t.Fatal(err)
	}
	artifacts, err := openAllocationSiteArtifacts(config.output)
	if err != nil {
		t.Fatal(err)
	}

	caller := runtime.FuncForPC(reflect.ValueOf(allocationSiteExecuteRegion).Pointer())
	if caller == nil {
		if closeErr := closeAllocationSiteClosers(artifacts.capture, artifacts.tail, artifacts.profile); closeErr != nil {
			t.Errorf("close artifacts after caller lookup failure: %v", closeErr)
		}
		t.Fatal("resolve allocationSiteExecuteRegion function name")
	}
	callerFunction := caller.Name()
	be, ok := backend.Get(backend.CPU)
	if !ok {
		if closeErr := closeAllocationSiteClosers(artifacts.capture, artifacts.tail, artifacts.profile); closeErr != nil {
			t.Errorf("close artifacts after backend lookup failure: %v", closeErr)
		}
		t.Fatal("CPU backend is not registered")
	}
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{bench.RandF64(tensor.Shape{1 << 18}, 3)}
	preRecords := make([]runtime.MemProfileRecord, allocationSiteRawCapacity, allocationSiteRawCapacity)
	postRecords := make([]runtime.MemProfileRecord, allocationSiteRawCapacity, allocationSiteRawCapacity)
	var preRawBefore, start, end, postGC, postRaw, tail runtime.MemStats

	if _, err := backend.Execute(ctx, backend.OpSoftplus, ins, nil); err != nil {
		if closeErr := closeAllocationSiteClosers(artifacts.capture, artifacts.tail, artifacts.profile); closeErr != nil {
			t.Errorf("close artifacts after warmup failure: %v", closeErr)
		}
		t.Fatalf("warmup Softplus Execute: %v", err)
	}

	runtime.GC()
	runtime.ReadMemStats(&preRawBefore)
	preN, preOK := runtime.MemProfile(preRecords, true)
	runtime.ReadMemStats(&start)
	completed, regionErr := allocationSiteExecuteRegion(ctx, ins)
	runtime.ReadMemStats(&end)
	runtime.GC()
	runtime.ReadMemStats(&postGC)
	postN, postOK := runtime.MemProfile(postRecords, true)
	runtime.ReadMemStats(&postRaw)

	profileRateAfter := runtime.MemProfileRate
	captureErrors := make([]string, 0)
	if preRawBefore.Mallocs != start.Mallocs || preRawBefore.TotalAlloc != start.TotalAlloc {
		captureErrors = append(captureErrors, fmt.Sprintf("pre snapshot moved process counters: mallocs %d -> %d, total_alloc %d -> %d", preRawBefore.Mallocs, start.Mallocs, preRawBefore.TotalAlloc, start.TotalAlloc))
	}
	if !preOK || preN < 0 || preN > len(preRecords) {
		captureErrors = append(captureErrors, fmt.Sprintf("pre snapshot invalid: reported_count=%d ok=%t capacity=%d", preN, preOK, len(preRecords)))
	}
	if !postOK || postN < 0 || postN > len(postRecords) {
		captureErrors = append(captureErrors, fmt.Sprintf("post snapshot invalid: reported_count=%d ok=%t capacity=%d", postN, postOK, len(postRecords)))
	}
	if completed != allocationSiteCalls {
		captureErrors = append(captureErrors, fmt.Sprintf("region completed %d calls, want %d", completed, allocationSiteCalls))
	}
	regionError := ""
	if regionErr != nil {
		regionError = regionErr.Error()
		captureErrors = append(captureErrors, "region error: "+regionError)
	}
	if profileRateAfter != profileRateBefore || profileRateAfter != 1 {
		captureErrors = append(captureErrors, fmt.Sprintf("runtime.MemProfileRate changed: before=%d after=%d", profileRateBefore, profileRateAfter))
	}

	capture := allocationSiteCapture{
		Schema:            allocationSiteSchema,
		Control:           allocationSiteControl,
		N:                 allocationSiteCalls,
		WarmupCalls:       1,
		CompletedCalls:    completed,
		GoVersion:         runtime.Version(),
		GOOS:              runtime.GOOS,
		GOARCH:            runtime.GOARCH,
		GOMAXPROCS:        config.gomaxprocs,
		GODEBUG:           config.godebug,
		GOGC:              config.gogc,
		GOMEMLIMIT:        config.gomemlimit,
		ProfileRateBefore: profileRateBefore,
		ProfileRateAfter:  profileRateAfter,
		RawCapacity:       allocationSiteRawCapacity,
		CallerFunction:    callerFunction,
		WorkerFunction:    allocationSiteWorker,
		Boundaries: allocationSiteBoundaries{
			PreRawBefore: allocationSiteCountersFromMemStats(&preRawBefore),
			Start:        allocationSiteCountersFromMemStats(&start),
			End:          allocationSiteCountersFromMemStats(&end),
			PostGC:       allocationSiteCountersFromMemStats(&postGC),
			PostRaw:      allocationSiteCountersFromMemStats(&postRaw),
		},
		Pre:         formatAllocationSiteSnapshot(preN, preOK, preRecords, runtimeAllocationSiteFrames),
		Post:        formatAllocationSiteSnapshot(postN, postOK, postRecords, runtimeAllocationSiteFrames),
		RegionError: regionError,
		Errors:      captureErrors,
	}

	var ioErrors []error
	if err := writeAllocationSiteJSON(artifacts.capture, capture); err != nil {
		ioErrors = append(ioErrors, fmt.Errorf("write capture.json: %w", err))
	}
	if profile := pprof.Lookup("allocs"); profile == nil {
		ioErrors = append(ioErrors, errors.New("lookup allocs profile: not found"))
	} else if err := profile.WriteTo(artifacts.profile, 0); err != nil {
		ioErrors = append(ioErrors, fmt.Errorf("write allocs.pprof: %w", err))
	}
	runtime.ReadMemStats(&tail)
	tailDocument := allocationSiteTail{
		Schema:              allocationSiteTailSchema,
		Tail:                allocationSiteCountersFromMemStats(&tail),
		ReportWriteExcluded: true,
	}
	if err := writeAllocationSiteJSON(artifacts.tail, tailDocument); err != nil {
		ioErrors = append(ioErrors, fmt.Errorf("write tail.json: %w", err))
	}
	if err := closeAllocationSiteClosers(artifacts.capture, artifacts.tail, artifacts.profile); err != nil {
		ioErrors = append(ioErrors, fmt.Errorf("close artifacts: %w", err))
	}
	// Keep the fixed raw buffers live through all serialization and artifact closes.
	runtime.KeepAlive(preRecords)
	runtime.KeepAlive(postRecords)
	for _, err := range ioErrors {
		t.Error(err)
	}
	for _, message := range captureErrors {
		t.Error(message)
	}
}

func validAllocationSiteConfigInput(output string) allocationSiteConfigInput {
	return allocationSiteConfigInput{
		selector:    allocationSiteSelector,
		output:      output,
		godebug:     "memprofilerate=1",
		gogc:        "100",
		gomemlimit:  "off",
		gomaxprocs:  1,
		profileRate: 1,
		testRun:     "^TestCPUControlAllocationSites$",
		testCount:   "1",
	}
}

func TestAllocationSiteDisabledSelectionDoesNoWork(t *testing.T) {
	input := allocationSiteConfigInput{selector: "", output: "relative", short: true, profileRate: 999}
	config, err := validateAllocationSiteConfig(input)
	if err != nil {
		t.Fatalf("disabled selector: %v", err)
	}
	if config.enabled {
		t.Fatal("disabled selector unexpectedly enabled capture")
	}
}

func TestAllocationSiteConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		change func(*allocationSiteConfigInput)
	}{
		{"selector", func(in *allocationSiteConfigInput) { in.selector = "gelu" }},
		{"relative output", func(in *allocationSiteConfigInput) { in.output = "relative" }},
		{"godebug", func(in *allocationSiteConfigInput) { in.godebug = "memprofilerate=2" }},
		{"gogc", func(in *allocationSiteConfigInput) { in.gogc = "off" }},
		{"gomemlimit", func(in *allocationSiteConfigInput) { in.gomemlimit = "1GiB" }},
		{"procs", func(in *allocationSiteConfigInput) { in.gomaxprocs = 2 }},
		{"rate", func(in *allocationSiteConfigInput) { in.profileRate = 2 }},
		{"run", func(in *allocationSiteConfigInput) { in.testRun = "TestCPUControlAllocationSites" }},
		{"count", func(in *allocationSiteConfigInput) { in.testCount = "2" }},
		{"short", func(in *allocationSiteConfigInput) { in.short = true }},
		{"memprofile", func(in *allocationSiteConfigInput) { in.memprofileSupplied = true }},
		{"memprofilerate flag", func(in *allocationSiteConfigInput) { in.profileRateSupplied = true }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := validAllocationSiteConfigInput(filepath.Join(t.TempDir(), "output"))
			test.change(&input)
			if _, err := validateAllocationSiteConfig(input); err == nil {
				t.Fatal("invalid configuration was accepted")
			}
		})
	}
	for _, procs := range []int{1, 12} {
		input := validAllocationSiteConfigInput(filepath.Join(t.TempDir(), "output"))
		input.gomaxprocs = procs
		config, err := validateAllocationSiteConfig(input)
		if err != nil {
			t.Fatalf("valid GOMAXPROCS=%d: %v", procs, err)
		}
		if !config.enabled {
			t.Fatalf("valid GOMAXPROCS=%d did not enable capture", procs)
		}
	}
}

func TestAllocationSiteCounterJSONRoundTrip(t *testing.T) {
	want := allocationSiteCounters{
		TotalAlloc:  1<<53 + 17,
		Mallocs:     math.MaxUint64 - 1,
		Frees:       1<<63 + 9,
		HeapAlloc:   math.MaxUint64,
		HeapObjects: 1<<54 + 3,
		NumGC:       math.MaxUint32,
	}
	b, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got allocationSiteCounters
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("counter round trip = %+v, want %+v", got, want)
	}
	var tokens map[string]json.RawMessage
	if err := json.Unmarshal(b, &tokens); err != nil {
		t.Fatal(err)
	}
	for name, token := range tokens {
		if strings.ContainsAny(string(token), ".eE") {
			t.Fatalf("counter %s used floating notation: %s", name, token)
		}
	}
}

func TestAllocationSiteFrameConversionRetainsLiteralLine(t *testing.T) {
	want := runtime.Frame{PC: 0x123, Function: "function", File: "file.go", Line: -7}
	got := allocationSiteFrameFromRuntime(want)
	if got.PC != "0x123" || got.Function != want.Function || got.File != want.File || got.Line != want.Line {
		t.Fatalf("converted frame = %+v, want literal %+v", got, want)
	}
}

func TestAllocationSiteRawRowsRetainSignedCountersPCsAndFrames(t *testing.T) {
	record := runtime.MemProfileRecord{AllocBytes: -1, FreeBytes: -2, AllocObjects: -3, FreeObjects: -4}
	for i := range record.Stack0 {
		record.Stack0[i] = uintptr(i + 1)
	}
	wantFrames := []allocationSiteFrame{
		{PC: "0xa", Function: "inline.second", File: "b.go", Line: 20},
		{PC: "0x9", Function: "inline.first", File: "a.go", Line: 10},
		{PC: "0xa", Function: "inline.second", File: "b.go", Line: 20},
	}
	row := formatAllocationSiteRawRow(7, record, func(got []uintptr) []allocationSiteFrame {
		if len(got) != 32 {
			t.Fatalf("resolver received %d PCs, want 32", len(got))
		}
		return append([]allocationSiteFrame(nil), wantFrames...)
	})
	if row.Ordinal != 7 || row.AllocBytes != -1 || row.FreeBytes != -2 || row.AllocObjects != -3 || row.FreeObjects != -4 {
		t.Fatalf("raw counters or ordinal changed: %+v", row)
	}
	if len(row.PCs) != 32 || row.PCs[0] != "0x1" || row.PCs[31] != "0x20" {
		t.Fatalf("PC array = %#v", row.PCs)
	}
	if !slices.Equal(row.Frames, wantFrames) {
		t.Fatalf("frames = %#v, want %#v", row.Frames, wantFrames)
	}
}

func TestAllocationSiteEmptyStackUsesEmptyArrays(t *testing.T) {
	row := formatAllocationSiteRawRow(0, runtime.MemProfileRecord{}, func(got []uintptr) []allocationSiteFrame {
		if len(got) != 0 {
			t.Fatalf("empty stack resolver received %d PCs", len(got))
		}
		return nil
	})
	if len(row.PCs) != 32 {
		t.Fatalf("PC count = %d, want 32", len(row.PCs))
	}
	if row.Frames == nil || len(row.Frames) != 0 {
		t.Fatalf("frames = %#v, want non-nil empty", row.Frames)
	}
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), `"frames":null`) || strings.Contains(string(b), `"pcs":null`) {
		t.Fatalf("empty arrays encoded as null: %s", b)
	}
}

func TestAllocationSiteSnapshotPrefixAndInsufficientBuffer(t *testing.T) {
	records := make([]runtime.MemProfileRecord, 4)
	for i := range records {
		records[i].AllocObjects = int64(i + 1)
	}
	resolve := func([]uintptr) []allocationSiteFrame { return make([]allocationSiteFrame, 0) }
	got := formatAllocationSiteSnapshot(2, true, records, resolve)
	if len(got.Rows) != 2 || got.Rows[0].Ordinal != 0 || got.Rows[1].Ordinal != 1 || got.Rows[1].AllocObjects != 2 {
		t.Fatalf("valid prefix = %+v", got)
	}
	overflow := formatAllocationSiteSnapshot(5, false, records, resolve)
	if overflow.ReportedCount != 5 || overflow.OK || overflow.Rows == nil || len(overflow.Rows) != 0 {
		t.Fatalf("overflow snapshot = %+v", overflow)
	}
	inconsistent := formatAllocationSiteSnapshot(5, true, records, resolve)
	if inconsistent.ReportedCount != 5 || !inconsistent.OK || inconsistent.Rows == nil || len(inconsistent.Rows) != 0 {
		t.Fatalf("inconsistent snapshot = %+v", inconsistent)
	}
}

func TestAllocationSiteOutputPathReuseRejectedWithoutOverwrite(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "existing")
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(dir, "capture.json")
	if err := os.WriteFile(sentinel, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := openAllocationSiteArtifacts(dir); err == nil {
		t.Fatal("reused path was accepted")
	}
	b, err := os.ReadFile(sentinel)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "keep" {
		t.Fatalf("existing artifact changed to %q", b)
	}
}

type allocationSiteShortWriter struct{}

func (allocationSiteShortWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	return len(p) - 1, nil
}

type allocationSiteErrorWriter struct{ err error }

func (w allocationSiteErrorWriter) Write([]byte) (int, error) { return 0, w.err }

type allocationSiteErrorCloser struct{ err error }

func (c allocationSiteErrorCloser) Close() error { return c.err }

func TestAllocationSiteWriteAndCloseFailuresPropagate(t *testing.T) {
	if err := writeAllocationSiteJSON(allocationSiteShortWriter{}, allocationSiteTail{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write error = %v, want io.ErrShortWrite", err)
	}
	writeErr := errors.New("write failed")
	if err := writeAllocationSiteJSON(allocationSiteErrorWriter{err: writeErr}, allocationSiteTail{}); !errors.Is(err, writeErr) {
		t.Fatalf("write error = %v, want %v", err, writeErr)
	}
	closeErrA := errors.New("close A")
	closeErrB := errors.New("close B")
	err := closeAllocationSiteClosers(allocationSiteErrorCloser{err: closeErrA}, allocationSiteErrorCloser{err: closeErrB})
	if !errors.Is(err, closeErrA) || !errors.Is(err, closeErrB) {
		t.Fatalf("close error = %v, want both failures", err)
	}
}

func allocationSiteTestContext(t *testing.T) *backend.Context {
	t.Helper()
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend is not registered")
	}
	return backend.NewContext().WithBackend(be)
}

func TestAllocationSiteRegionReturnsActualCountOnExecuteError(t *testing.T) {
	completed, err := allocationSiteExecuteRegion(allocationSiteTestContext(t), nil)
	if err == nil {
		t.Fatal("invalid inputs unexpectedly succeeded")
	}
	if completed != 0 {
		t.Fatalf("completed = %d, want 0", completed)
	}
}

func TestAllocationSiteRegionCompletes1024Calls(t *testing.T) {
	ins := []*tensor.Tensor{bench.RandF64(tensor.Shape{1}, 3)}
	completed, err := allocationSiteExecuteRegion(allocationSiteTestContext(t), ins)
	if err != nil {
		t.Fatal(err)
	}
	if completed != allocationSiteCalls {
		t.Fatalf("completed = %d, want %d", completed, allocationSiteCalls)
	}
}
