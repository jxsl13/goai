package main

import (
	"bufio"
	"crypto/sha256"
	"debug/buildinfo"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"
)

const pinnedBenchstatVersion = "v0.0.0-20260709024250-82a0b07e230d"

var siluBenchLine = regexp.MustCompile(`^BenchmarkSiLUF64Into-[0-9]+\s+[0-9]+\s+([0-9.]+) ns/op`)

type evidenceMetadata struct {
	Schema     int               `json:"schema"`
	Cell       string            `json:"cell"`
	CapturedAt string            `json:"captured_at"`
	Host       map[string]any    `json:"host"`
	Protocol   map[string]any    `json:"protocol"`
	GoAI       map[string]any    `json:"goai"`
	Incumbent  map[string]any    `json:"incumbent"`
	Commands   map[string]string `json:"commands"`
}

type evidenceSummary struct {
	Schema             int       `json:"schema"`
	Cell               string    `json:"cell"`
	GoAISamplesNS      []float64 `json:"goai_samples_ns"`
	IncumbentSamplesNS []float64 `json:"incumbent_samples_ns"`
	GoAIMedianNS       float64   `json:"goai_median_ns"`
	IncumbentMedianNS  float64   `json:"incumbent_median_ns"`
	GoAISpeedup        float64   `json:"goai_speedup"`
}

func collectSiLU(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("collect-silu", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "GoAI workspace root")
	outDir := fs.String("out", "", "new evidence directory (required)")
	samples := fs.Int("samples", 10, "alternating samples per implementation")
	seconds := fs.Float64("seconds", 1, "minimum seconds per sample")
	goProcs := fs.Int("go-procs", 12, "GOMAXPROCS for GoAI")
	torchThreads := fs.Int("torch-threads", 8, "PyTorch intra-op threads")
	python := fs.String("python", ".venv/bin/python", "Python with pinned torch")
	benchstat := fs.String("benchstat", "benchstat", "pinned benchstat binary")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *outDir == "" {
		return fmt.Errorf("-out is required")
	}
	if *samples < 10 {
		return fmt.Errorf("-samples=%d: leadership protocol requires at least 10", *samples)
	}
	if *seconds <= 0 {
		return fmt.Errorf("-seconds must be positive")
	}
	absRoot, err := filepath.Abs(*root)
	if err != nil {
		return err
	}
	absOut, err := filepath.Abs(*outDir)
	if err != nil {
		return err
	}
	if err := os.Mkdir(absOut, 0o755); err != nil {
		return fmt.Errorf("create evidence directory: %w", err)
	}
	pythonPath := *python
	if !filepath.IsAbs(pythonPath) {
		pythonPath = filepath.Join(absRoot, pythonPath)
	}
	benchstatPath, err := exec.LookPath(*benchstat)
	if err != nil {
		return fmt.Errorf("find benchstat: %w", err)
	}
	if err := requireBenchstatVersion(benchstatPath); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp("", "goai-leadership-silu-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	testBin := filepath.Join(tmp, "cpu.test")
	compileEnv := withEnv(os.Environ(), map[string]string{"CGO_ENABLED": "0", "GOEXPERIMENT": "simd"})
	compile := []string{"test", "-c", "-o", testBin, "./backend/cpu"}
	if _, err := runCommand(absRoot, compileEnv, "go", compile...); err != nil {
		return fmt.Errorf("prebuild Go benchmark: %w", err)
	}
	qualityArgs := []string{
		"test", "./backend/cpu", "-run",
		"TestSiLUIntoMatchesExecute|TestVsiluF64Arm64Accuracy|TestVsiluF64Arm64VectorTailBitIdentity|TestVsiluF64Arm64Edges",
		"-count=1", "-v",
	}
	quality, err := runCommand(absRoot, compileEnv, "go", qualityArgs...)
	if err != nil {
		return fmt.Errorf("quality gate: %w", err)
	}
	if err := os.WriteFile(filepath.Join(absOut, "quality.txt"), quality, 0o644); err != nil {
		return err
	}
	goFile, err := exclusiveFile(filepath.Join(absOut, "goai.txt"))
	if err != nil {
		return err
	}
	defer goFile.Close()
	pyFile, err := exclusiveFile(filepath.Join(absOut, "pytorch.txt"))
	if err != nil {
		return err
	}
	defer pyFile.Close()
	cpu := hostCPU()
	goEnv := withEnv(os.Environ(), map[string]string{
		"CGO_ENABLED": "0", "GOEXPERIMENT": "simd", "GOMAXPROCS": strconv.Itoa(*goProcs),
	})
	pyEnv := withEnv(os.Environ(), map[string]string{"GOAI_CPU": cpu})
	goArgs := []string{"-test.run=^$", "-test.bench=^BenchmarkSiLUF64Into$", "-test.benchtime=" + strconv.FormatFloat(*seconds, 'f', 3, 64) + "s", "-test.count=1", "-test.benchmem"}
	pyScript := filepath.Join(absRoot, "internal/benchcompare/leadership/silu_out.py")
	pyArgs := []string{pyScript, "--seconds", strconv.FormatFloat(*seconds, 'f', 3, 64), "--threads", strconv.Itoa(*torchThreads), "--name", fmt.Sprintf("BenchmarkSiLUF64Into-%d", *goProcs)}
	for i := 0; i < *samples; i++ {
		if i%2 == 0 {
			if err := appendRun(goFile, absRoot, goEnv, testBin, goArgs...); err != nil {
				return fmt.Errorf("GoAI sample %d: %w", i+1, err)
			}
			if err := appendRun(pyFile, absRoot, pyEnv, pythonPath, pyArgs...); err != nil {
				return fmt.Errorf("PyTorch sample %d: %w", i+1, err)
			}
		} else {
			if err := appendRun(pyFile, absRoot, pyEnv, pythonPath, pyArgs...); err != nil {
				return fmt.Errorf("PyTorch sample %d: %w", i+1, err)
			}
			if err := appendRun(goFile, absRoot, goEnv, testBin, goArgs...); err != nil {
				return fmt.Errorf("GoAI sample %d: %w", i+1, err)
			}
		}
		fmt.Fprintf(stdout, "sample %d/%d complete\n", i+1, *samples)
	}
	if err := goFile.Close(); err != nil {
		return err
	}
	if err := pyFile.Close(); err != nil {
		return err
	}
	goPath := filepath.Join(absOut, "goai.txt")
	pyPath := filepath.Join(absOut, "pytorch.txt")
	stats, err := runCommand(absRoot, os.Environ(), benchstatPath, "GoAI="+goPath, "PyTorch="+pyPath)
	if err != nil {
		return fmt.Errorf("benchstat: %w", err)
	}
	if err := os.WriteFile(filepath.Join(absOut, "benchstat.txt"), stats, 0o644); err != nil {
		return err
	}
	goSamples, err := benchmarkSamples(goPath)
	if err != nil {
		return err
	}
	pySamples, err := benchmarkSamples(pyPath)
	if err != nil {
		return err
	}
	goMedian, pyMedian := median(goSamples), median(pySamples)
	summary := evidenceSummary{
		Schema: 1, Cell: "m2-cpu-silu-f64-into", GoAISamplesNS: goSamples,
		IncumbentSamplesNS: pySamples, GoAIMedianNS: goMedian,
		IncumbentMedianNS: pyMedian, GoAISpeedup: pyMedian / goMedian,
	}
	if err := writeJSONExclusive(filepath.Join(absOut, "summary.json"), summary); err != nil {
		return err
	}
	metadata, err := buildMetadata(absRoot, testBin, pythonPath, benchstatPath, cpu, *samples, *seconds, *goProcs, *torchThreads, compile, goArgs, pyArgs)
	if err != nil {
		return err
	}
	if err := writeJSONExclusive(filepath.Join(absOut, "metadata.json"), metadata); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "GoAI median %.0f ns/op; PyTorch median %.0f ns/op; GoAI %.3fx faster\n", goMedian, pyMedian, pyMedian/goMedian)
	fmt.Fprintf(stdout, "evidence: %s\n", absOut)
	return nil
}

func buildMetadata(root, testBin, python, benchstat, cpu string, samples int, seconds float64, goProcs, torchThreads int, compile, goArgs, pyArgs []string) (evidenceMetadata, error) {
	revision := gitIdentity(root)
	sourceHash, err := sourceDigest(root, []string{
		"backend/backend.go", "backend/execute.go", "tensor/storage.go",
		"backend/cpu/cpu.go", "backend/cpu/elementwise.go", "backend/cpu/vexp_arm64.go",
		"backend/cpu/vexp_arm64.s", "backend/cpu/vsilu_f64_bench_test.go",
	})
	if err != nil {
		return evidenceMetadata{}, err
	}
	binaryHash, err := fileDigest(testBin)
	if err != nil {
		return evidenceMetadata{}, err
	}
	goVersion, _ := runCommand(root, os.Environ(), "go", "version")
	pyVersion, _ := runCommand(root, os.Environ(), python, "-c", "import platform,torch; print(platform.python_version()); print(torch.__version__)")
	osVersion, _ := runCommand(root, os.Environ(), "sw_vers", "-productVersion")
	mem, _ := runCommand(root, os.Environ(), "sysctl", "-n", "hw.memsize")
	return evidenceMetadata{
		Schema: 1, Cell: "m2-cpu-silu-f64-into", CapturedAt: time.Now().UTC().Format(time.RFC3339),
		Host: map[string]any{"os": runtime.GOOS, "arch": runtime.GOARCH, "os_version": strings.TrimSpace(string(osVersion)), "cpu": cpu, "memory_bytes": strings.TrimSpace(string(mem))},
		Protocol: map[string]any{
			"samples": samples, "interleaved": true, "prebuilt_go": true,
			"prebuilt_incumbent": true, "seconds_per_sample": seconds,
			"statistics": "benchstat median and 95% confidence interval",
		},
		GoAI:      map[string]any{"revision": revision, "source_sha256": sourceHash, "binary_sha256": binaryHash, "runtime": strings.TrimSpace(string(goVersion)), "gomaxprocs": goProcs},
		Incumbent: map[string]any{"name": "PyTorch", "revision": "v2.12.1", "runtime": strings.TrimSpace(string(pyVersion)), "threads": torchThreads},
		Commands: map[string]string{
			"compile":    "GOEXPERIMENT=simd CGO_ENABLED=0 go " + strings.Join(compile, " "),
			"goai":       testBin + " " + strings.Join(goArgs, " "),
			"incumbent":  python + " " + strings.Join(pyArgs, " "),
			"statistics": benchstat + " GoAI=goai.txt PyTorch=pytorch.txt",
		},
	}, nil
}

func requireBenchstatVersion(path string) error {
	info, err := buildinfo.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read benchstat build info: %w", err)
	}
	if info.Path != "golang.org/x/perf/cmd/benchstat" || info.Main.Version != pinnedBenchstatVersion {
		return fmt.Errorf("benchstat must be golang.org/x/perf@%s, got %s@%s", pinnedBenchstatVersion, info.Path, info.Main.Version)
	}
	return nil
}

func exclusiveFile(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return nil, fmt.Errorf("create %s: %w", path, err)
	}
	return f, nil
}

func appendRun(dst io.Writer, dir string, env []string, name string, args ...string) error {
	out, err := runCommand(dir, env, name, args...)
	if err != nil {
		return err
	}
	if _, err := dst.Write(out); err != nil {
		return err
	}
	_, err = io.WriteString(dst, "\n")
	return err
}

func runCommand(dir string, env []string, name string, args ...string) ([]byte, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w\n%s", name, strings.Join(args, " "), err, out)
	}
	return out, nil
}

func withEnv(base []string, values map[string]string) []string {
	out := make([]string, 0, len(base)+len(values))
	for _, entry := range base {
		key := entry
		if i := strings.IndexByte(entry, '='); i >= 0 {
			key = entry[:i]
		}
		if _, replace := values[key]; !replace {
			out = append(out, entry)
		}
	}
	for key, value := range values {
		out = append(out, key+"="+value)
	}
	return out
}

func benchmarkSamples(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []float64
	s := bufio.NewScanner(f)
	for s.Scan() {
		m := siluBenchLine.FindStringSubmatch(s.Text())
		if m == nil {
			continue
		}
		v, err := strconv.ParseFloat(m[1], 64)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(out) < 10 {
		return nil, fmt.Errorf("%s: found %d benchmark samples, need at least 10", path, len(out))
	}
	return out, nil
}

func median(in []float64) float64 {
	v := slices.Clone(in)
	slices.Sort(v)
	n := len(v)
	if n%2 == 1 {
		return v[n/2]
	}
	return (v[n/2-1] + v[n/2]) / 2
}

func gitIdentity(root string) string {
	bare, err := runCommand(root, os.Environ(), "git", "rev-parse", "--is-bare-repository")
	if err == nil && strings.TrimSpace(string(bare)) == "true" {
		return "bare-workspace-uncommitted"
	}
	sha, err := runCommand(root, os.Environ(), "git", "rev-parse", "--verify", "HEAD")
	if err != nil {
		return "workspace-uncommitted"
	}
	status, err := runCommand(root, os.Environ(), "git", "status", "--porcelain")
	if err != nil || len(strings.TrimSpace(string(status))) != 0 {
		return strings.TrimSpace(string(sha)) + "-dirty"
	}
	return strings.TrimSpace(string(sha))
}

func hostCPU() string {
	out, err := runCommand(".", os.Environ(), "sysctl", "-n", "machdep.cpu.brand_string")
	if err == nil && strings.TrimSpace(string(out)) != "" {
		return strings.TrimSpace(string(out))
	}
	return runtime.GOARCH
}

func sourceDigest(root string, paths []string) (string, error) {
	h := sha256.New()
	slices.Sort(paths)
	for _, path := range paths {
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil {
			return "", err
		}
		io.WriteString(h, path)
		h.Write([]byte{0})
		h.Write(data)
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func fileDigest(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func writeJSONExclusive(path string, value any) error {
	f, err := exclusiveFile(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	err = enc.Encode(value)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
