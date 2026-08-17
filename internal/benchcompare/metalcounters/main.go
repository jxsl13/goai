package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	counterValueXPath = `/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-value"]`
	submissionXPath   = `/trace-toc/run[@number="1"]/data/table[@schema="metal-application-command-buffer-submissions"]`
	completionXPath   = `/trace-toc/run[@number="1"]/data/table[@schema="metal-command-buffer-completed"]`
	gpuIntervalsXPath = `/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]`
	stageProfileEnv   = "GOAI_METAL_STAGE_PROFILE"
)

type options struct {
	repo                string
	analyzeDir          string
	packagePath         string
	buildTags           string
	forwardEnv          string
	activeCounterMax    string
	requiredCounters    string
	capabilityCounters  string
	stageRequired       string
	stageMinDuration    time.Duration
	stageProfile        bool
	requireExclusiveGPU bool
	benchmark           string
	iterations          int
	buffersPerIteration int
	timeLimit           time.Duration
	output              string
	keepTemp            bool
}

func main() {
	var opts options
	flag.StringVar(&opts.repo, "repo", ".", "goai repository root")
	flag.StringVar(&opts.analyzeDir, "analyze-dir", "", "analyze an existing temporary export directory instead of capturing")
	flag.StringVar(&opts.packagePath, "package", "./backend/metal", "Go package containing the benchmark")
	flag.StringVar(&opts.buildTags, "tags", "", "comma-separated Go build tags for the benchmark binary")
	flag.StringVar(&opts.forwardEnv, "forward-env", "", "comma-separated environment variable names to pass explicitly through xctrace --launch")
	flag.StringVar(&opts.activeCounterMax, "active-counter-max", "", "comma-separated NAME=MAX ceilings for active command-buffer counter means")
	flag.StringVar(&opts.requiredCounters, "required-counters", "*", "comma-separated globally required counter names; * preserves strict-all completeness")
	flag.StringVar(&opts.capabilityCounters, "capability-counters", "", "comma-separated capability counters retained when unsampled but not claim-required")
	flag.StringVar(&opts.stageRequired, "stage-required-counters", "*", "comma-separated counters required per resolved stage; * preserves strict-all completeness")
	flag.DurationVar(&opts.stageMinDuration, "stage-min-duration", 0, "stages shorter than this resolution floor retain missing samples without rejecting the capture")
	flag.BoolVar(&opts.stageProfile, "stage-profile", false, "correlate a one-buffer Recorder stage sidecar with target GPU intervals (report JSON v4)")
	flag.BoolVar(&opts.requireExclusiveGPU, "require-exclusive-gpu-window", false, "reject stage reports with any foreign same-GPU interval overlapping the target command-buffer span")
	flag.StringVar(&opts.benchmark, "benchmark", `^BenchmarkMetalQ4KDecodeLeaf/K2048N5632/cooperative$`, "exact Go benchmark regular expression")
	flag.IntVar(&opts.iterations, "iterations", 10000, "fixed benchmark iteration count")
	flag.IntVar(&opts.buffersPerIteration, "buffers-per-iteration", 1, "expected Metal command buffers per timed benchmark iteration")
	flag.DurationVar(&opts.timeLimit, "time-limit", 30*time.Second, "xctrace recording time limit")
	flag.StringVar(&opts.output, "output", "", "compact JSON output path (stdout when empty)")
	flag.BoolVar(&opts.keepTemp, "keep-temp", false, "retain raw trace and XML in the printed system temporary directory")
	flag.Parse()

	if err := run(context.Background(), opts, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "metalcounters: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, opts options, stdout, stderr io.Writer) error {
	if opts.iterations <= 0 || opts.buffersPerIteration <= 0 {
		return errors.New("iterations and buffers-per-iteration must be positive")
	}
	if strings.TrimSpace(opts.packagePath) == "" {
		return errors.New("package must not be empty")
	}
	if opts.stageProfile && (opts.iterations != 1 || opts.buffersPerIteration != 1) {
		return errors.New("stage-profile requires exactly one iteration and one buffer per iteration")
	}
	if opts.stageProfile && forwardedEnvContains(opts.forwardEnv, stageProfileEnv) {
		return fmt.Errorf("stage-profile reserves forward-env %s", stageProfileEnv)
	}
	if opts.requireExclusiveGPU && !opts.stageProfile {
		return errors.New("require-exclusive-gpu-window requires stage-profile")
	}
	ceilings, err := parseCounterCeilings(opts.activeCounterMax)
	if err != nil {
		return err
	}
	if opts.stageMinDuration < 0 {
		return errors.New("stage-min-duration must be nonnegative")
	}
	policy, err := parseCounterPolicy(opts.requiredCounters, opts.capabilityCounters, opts.stageRequired, uint64(opts.stageMinDuration), ceilings)
	if err != nil {
		return err
	}
	if opts.analyzeDir != "" {
		paths, err := resolveAnalyzePaths(opts.analyzeDir)
		if err != nil {
			return err
		}
		result, err := analyzeFiles(paths, opts, policy)
		if err != nil {
			return err
		}
		if err := validateCounterCeilings(result, ceilings); err != nil {
			return err
		}
		return emitReport(stdout, stderr, opts.output, result)
	}
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("Metal counter capture requires darwin, got %s", runtime.GOOS)
	}
	if opts.timeLimit <= 0 {
		return errors.New("time-limit must be positive")
	}
	repo, err := filepath.Abs(opts.repo)
	if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(repo, "go.mod")); err != nil {
		return fmt.Errorf("repository root %q: %w", repo, err)
	}
	if _, err := exec.LookPath("xcrun"); err != nil {
		return errors.New("xcrun is required; install Xcode command-line tools")
	}

	tmp, err := os.MkdirTemp("", "goai-metal-counters-")
	if err != nil {
		return err
	}
	inside, err := pathWithin(repo, tmp)
	if err != nil {
		return err
	}
	if inside {
		return fmt.Errorf("refusing raw trace directory inside repository: %s", tmp)
	}
	if opts.keepTemp {
		fmt.Fprintf(stderr, "METAL_COUNTER_TEMP=%s\n", tmp)
	} else {
		defer os.RemoveAll(tmp)
	}

	testBinary := filepath.Join(tmp, "metal.test")
	if err := command(ctx, repo, stderr, "go", compileBenchmarkArgs(opts, testBinary)...); err != nil {
		return err
	}
	tracePath := filepath.Join(tmp, "capture.trace")
	targetOutput := filepath.Join(tmp, "benchmark.txt")
	timeLimit := formatXctraceDuration(opts.timeLimit)
	testArgs := []string{
		"xctrace", "record",
		"--instrument", "Metal GPU Counters",
		"--instrument", "Metal Application",
	}
	if opts.stageProfile {
		testArgs = append(testArgs, "--instrument", "GPU")
	}
	testArgs = append(testArgs,
		"--time-limit", timeLimit,
		"--output", tracePath,
		"--target-stdout", targetOutput,
	)
	forwarded, err := forwardedEnvArgs(opts.forwardEnv)
	if err != nil {
		return err
	}
	testArgs = append(testArgs, forwarded...)
	if opts.stageProfile {
		testArgs = append(testArgs, "--env", stageProfileEnv+"="+filepath.Join(tmp, "stage-profile.json"))
	}
	testArgs = append(testArgs,
		"--launch", "--", testBinary,
		"-test.run", "^$",
		"-test.bench", opts.benchmark,
		"-test.benchtime="+strconv.Itoa(opts.iterations)+"x",
		"-test.count=1",
		"-test.benchmem",
	)
	deadline, cancel := context.WithTimeout(ctx, opts.timeLimit+2*time.Minute)
	defer cancel()
	if err := command(deadline, repo, stderr, "xcrun", testArgs...); err != nil {
		return err
	}
	benchmarkOutput, err := os.ReadFile(targetOutput)
	if err != nil {
		return fmt.Errorf("read benchmark output: %w", err)
	}
	if !completedFixedBenchmark(string(benchmarkOutput), opts.iterations) {
		return fmt.Errorf("target output has no completed %d-iteration benchmark: %s", opts.iterations, strings.TrimSpace(string(benchmarkOutput)))
	}

	paths := exportPaths(tmp)
	tocPath := filepath.Join(tmp, "toc.xml")
	if err := command(deadline, repo, stderr, "xcrun", "xctrace", "export", "--input", tracePath, "--toc", "--output", tocPath); err != nil {
		return err
	}
	tocFile, err := os.Open(tocPath)
	if err != nil {
		return err
	}
	infoXPaths, err := counterInfoTableXPaths(tocFile)
	tocFile.Close()
	if err != nil {
		return err
	}
	var infoErrors []string
	for i, xpath := range infoXPaths {
		candidate := filepath.Join(tmp, fmt.Sprintf("counter-info-%d.xml", i))
		if err := command(deadline, repo, stderr, "xcrun", "xctrace", "export", "--input", tracePath, "--xpath", xpath, "--output", candidate); err != nil {
			return err
		}
		file, err := os.Open(candidate)
		if err != nil {
			return err
		}
		_, parseErr := parseCounterInfo(file)
		file.Close()
		if parseErr != nil {
			infoErrors = append(infoErrors, fmt.Sprintf("%s: %v", xpath, parseErr))
			continue
		}
		if paths["counter-info"] != filepath.Join(tmp, "counter-info.xml") {
			return errors.New("multiple GPU counter-info tables contain Performance Limiters")
		}
		paths["counter-info"] = candidate
	}
	if paths["counter-info"] == filepath.Join(tmp, "counter-info.xml") {
		return fmt.Errorf("no GPU counter-info table contains Performance Limiters: %s", strings.Join(infoErrors, "; "))
	}
	queries := map[string]string{
		"counter-value": counterValueXPath,
		"submissions":   submissionXPath,
		"completions":   completionXPath,
		"gpu-intervals": gpuIntervalsXPath,
	}
	exports := []string{"submissions", "completions", "counter-value"}
	if opts.stageProfile {
		exports = append(exports, "gpu-intervals")
	}
	for _, name := range exports {
		if err := command(deadline, repo, stderr, "xcrun", "xctrace", "export", "--input", tracePath, "--xpath", queries[name], "--output", paths[name]); err != nil {
			return err
		}
	}

	result, err := analyzeFiles(paths, opts, policy)
	if err != nil {
		return err
	}
	if err := validateCounterCeilings(result, ceilings); err != nil {
		return err
	}
	return emitReport(stdout, stderr, opts.output, result)
}

func compileBenchmarkArgs(opts options, output string) []string {
	args := []string{"test", "-c", "-o", output}
	if tags := strings.TrimSpace(opts.buildTags); tags != "" {
		args = append(args, "-tags", tags)
	}
	return append(args, opts.packagePath)
}

func forwardedEnvArgs(names string) ([]string, error) {
	if strings.TrimSpace(names) == "" {
		return nil, nil
	}
	seen := make(map[string]struct{})
	var args []string
	for _, raw := range strings.Split(names, ",") {
		name := strings.TrimSpace(raw)
		if !validEnvName(name) {
			return nil, fmt.Errorf("invalid forward-env name %q", name)
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("duplicate forward-env name %q", name)
		}
		seen[name] = struct{}{}
		value, ok := os.LookupEnv(name)
		if !ok {
			return nil, fmt.Errorf("forward-env %s is not set", name)
		}
		args = append(args, "--env", name+"="+value)
	}
	return args, nil
}

func validEnvName(name string) bool {
	if name == "" || !(name[0] == '_' || name[0] >= 'A' && name[0] <= 'Z' || name[0] >= 'a' && name[0] <= 'z') {
		return false
	}
	for i := 1; i < len(name); i++ {
		c := name[i]
		if c != '_' && (c < 'A' || c > 'Z') && (c < 'a' || c > 'z') && (c < '0' || c > '9') {
			return false
		}
	}
	return true
}

func forwardedEnvContains(names, target string) bool {
	for _, raw := range strings.Split(names, ",") {
		if strings.TrimSpace(raw) == target {
			return true
		}
	}
	return false
}

type counterCeiling struct {
	name string
	max  float64
}

func parseCounterCeilings(value string) ([]counterCeiling, error) {
	if strings.TrimSpace(value) == "" {
		return nil, nil
	}
	seen := make(map[string]struct{})
	var ceilings []counterCeiling
	for _, raw := range strings.Split(value, ",") {
		clause := strings.TrimSpace(raw)
		separator := strings.LastIndexByte(clause, '=')
		if separator <= 0 || separator == len(clause)-1 {
			return nil, fmt.Errorf("invalid active-counter-max clause %q; want NAME=MAX", clause)
		}
		name := strings.TrimSpace(clause[:separator])
		if name == "" {
			return nil, fmt.Errorf("invalid active-counter-max clause %q; counter name is empty", clause)
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("duplicate active-counter-max counter %q", name)
		}
		maximum, err := strconv.ParseFloat(strings.TrimSpace(clause[separator+1:]), 64)
		if err != nil || math.IsNaN(maximum) || math.IsInf(maximum, 0) || maximum < 0 {
			return nil, fmt.Errorf("invalid active-counter-max value for %q: %q", name, strings.TrimSpace(clause[separator+1:]))
		}
		seen[name] = struct{}{}
		ceilings = append(ceilings, counterCeiling{name: name, max: maximum})
	}
	return ceilings, nil
}

func validateCounterCeilings(result report, ceilings []counterCeiling) error {
	if len(ceilings) == 0 {
		return nil
	}
	counters := make(map[string]counterReport, len(result.Counters))
	for _, counter := range result.Counters {
		counters[counter.Name] = counter
	}
	for _, ceiling := range ceilings {
		counter, ok := counters[ceiling.name]
		if !ok {
			return fmt.Errorf("active-counter-max counter %q is absent from report", ceiling.name)
		}
		if counter.Active.Samples == 0 {
			return fmt.Errorf("active-counter-max counter %q has no active samples", ceiling.name)
		}
		if counter.Active.Mean > ceiling.max {
			return fmt.Errorf("active-counter-max counter %q mean %.6f exceeds %.6f", ceiling.name, counter.Active.Mean, ceiling.max)
		}
	}
	return nil
}

func exportPaths(dir string) map[string]string {
	return map[string]string{
		"counter-info":  filepath.Join(dir, "counter-info.xml"),
		"counter-value": filepath.Join(dir, "counter-value.xml"),
		"submissions":   filepath.Join(dir, "submissions.xml"),
		"completions":   filepath.Join(dir, "completions.xml"),
		"gpu-intervals": filepath.Join(dir, "gpu-intervals.xml"),
		"stage-profile": filepath.Join(dir, "stage-profile.json"),
	}
}

func resolveAnalyzePaths(dir string) (map[string]string, error) {
	paths := exportPaths(dir)
	var candidates []string
	if _, err := os.Stat(paths["counter-info"]); err == nil {
		candidates = append(candidates, paths["counter-info"])
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	numbered, err := filepath.Glob(filepath.Join(dir, "counter-info-*.xml"))
	if err != nil {
		return nil, err
	}
	candidates = append(candidates, numbered...)
	var match string
	var parseErrors []string
	for _, candidate := range candidates {
		file, err := os.Open(candidate)
		if err != nil {
			return nil, err
		}
		_, parseErr := parseCounterInfo(file)
		file.Close()
		if parseErr != nil {
			parseErrors = append(parseErrors, fmt.Sprintf("%s: %v", filepath.Base(candidate), parseErr))
			continue
		}
		if match != "" {
			return nil, errors.New("multiple exported counter-info files contain Performance Limiters")
		}
		match = candidate
	}
	if match == "" {
		return nil, fmt.Errorf("no exported counter-info file contains Performance Limiters: %s", strings.Join(parseErrors, "; "))
	}
	paths["counter-info"] = match
	return paths, nil
}

func emitReport(stdout, stderr io.Writer, output string, result report) error {
	printHuman(stderr, result)
	var out io.Writer = stdout
	var file *os.File
	if output != "" {
		var err error
		file, err = os.Create(output)
		if err != nil {
			return err
		}
		defer file.Close()
		out = file
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		return err
	}
	return nil
}

func analyzeFiles(paths map[string]string, opts options, policy counterPolicy) (report, error) {
	infoFile, err := os.Open(paths["counter-info"])
	if err != nil {
		return report{}, err
	}
	infos, err := parseCounterInfo(infoFile)
	infoFile.Close()
	if err != nil {
		return report{}, err
	}
	if err := policy.validate(infos); err != nil {
		return report{}, err
	}

	submissionFile, err := os.Open(paths["submissions"])
	if err != nil {
		return report{}, err
	}
	commands, err := parseCommandSubmissions(submissionFile)
	submissionFile.Close()
	if err != nil {
		return report{}, err
	}
	completionFile, err := os.Open(paths["completions"])
	if err != nil {
		return report{}, err
	}
	completed, err := parseCommandCompletions(completionFile)
	completionFile.Close()
	if err != nil {
		return report{}, err
	}
	selectedCount := opts.iterations * opts.buffersPerIteration
	intervals, err := selectCommandBuffers(commands, completed, selectedCount)
	if err != nil {
		return report{}, err
	}

	valueFile, err := os.Open(paths["counter-value"])
	if err != nil {
		return report{}, err
	}
	counters, window, err := analyzeValuesWithPolicy(valueFile, infos, intervals, policy)
	valueFile.Close()
	if err != nil {
		return report{}, err
	}
	result := report{
		Version:                reportVersion,
		Scope:                  "device-wide counters sampled within target command-buffer timing windows",
		GPU:                    intervals[0].GPU,
		CounterSet:             "Performance Limiters",
		Benchmark:              opts.benchmark,
		Iterations:             opts.iterations,
		BuffersPerIteration:    opts.buffersPerIteration,
		ObservedCommandBuffers: len(commands),
		SelectedCommandBuffers: selectedCount,
		Window:                 window,
		Policy:                 policy.report(),
		Counters:               counters,
	}
	if opts.stageProfile {
		stage, err := analyzeStageProfile(paths, infos, intervals, policy, opts.requireExclusiveGPU)
		if err != nil {
			return report{}, err
		}
		result.Version = stageReportVersion
		result.Stage = &stage
	}
	return result, nil
}

func command(ctx context.Context, dir string, stderr io.Writer, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Stdout = stderr
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s: %w", name, strings.Join(args, " "), err)
	}
	return nil
}

func completedFixedBenchmark(output string, iterations int) bool {
	want := strconv.Itoa(iterations)
	for _, line := range strings.Split(output, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 3 && strings.HasPrefix(fields[0], "Benchmark") && fields[1] == want {
			return true
		}
	}
	return false
}

func formatXctraceDuration(duration time.Duration) string {
	seconds := int64(duration.Round(time.Second) / time.Second)
	if seconds < 1 {
		seconds = 1
	}
	return strconv.FormatInt(seconds, 10) + "s"
}

func pathWithin(parent, child string) (bool, error) {
	rel, err := filepath.Rel(parent, child)
	if err != nil {
		return false, err
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))), nil
}

func printHuman(w io.Writer, result report) {
	fmt.Fprintf(w, "METAL_COUNTER_PROFILE gpu=%q benchmark=%q iterations=%d selected_buffers=%d observed_buffers=%d wall=%dns active=%dns duty=%.4f\n",
		result.GPU, result.Benchmark, result.Iterations, result.SelectedCommandBuffers,
		result.ObservedCommandBuffers, result.Window.DurationNS, result.Window.ActiveDurationNS,
		result.Window.DutyCycle)
	for _, counter := range result.Counters {
		if !isRequiredLimiter(counter.Name) {
			continue
		}
		fmt.Fprintf(w, "METAL_COUNTER name=%q wall_mean=%.6f wall_samples=%d active_mean=%.6f active_samples=%d\n",
			counter.Name, counter.Wall.Mean, counter.Wall.Samples, counter.Active.Mean, counter.Active.Samples)
	}
}

func isRequiredLimiter(name string) bool {
	for _, required := range requiredLimiterNames {
		if name == required {
			return true
		}
	}
	return false
}
