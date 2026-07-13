// Command rendertables converts the combined output of `make bench-compare`
// and `make bench-python` into the markdown comparison matrix for
// docs/benchmarking.md (SPEC T606): every GoAI variant (ref/cpu/metal/vulkan)
// side by side with every industry reference measured on the same machine
// (numpy, torch-cpu, torch-mps).
//
// In plain terms: it turns raw benchmark logs into the readable table the
// documentation shows, so the numbers in the docs are always regenerable from
// a single command instead of hand-edited.
//
// Usage: go run ./internal/benchcompare/rendertables <bench-output-file>
package main

import (
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/jxsl13/goai/backend"
)

var (
	goLine = regexp.MustCompile(`^Benchmark(\w+)(?:/([\w]+))?/(ref|cpu|metal|vulkan|batched-\w+|per-op-\w+|stepn-\w+|sequential-\w+)-\d+\s+\d+\s+(\d+) ns/op(?:\s+([\d.]+ (?:GFLOP/s|tok/s)))?`)
	pyLine = regexp.MustCompile(`^(\w+)/([\w/x]+)/(numpy-cpu|torch-cpu|torch-mps)\s+([\d.]+ (?:GFLOP/s|ms/op))`)
)

type goCell struct {
	ns    int64
	extra string
}

func fmtNs(ns int64) string {
	switch {
	case ns >= 1e9:
		return fmt.Sprintf("%.2f s", float64(ns)/1e9)
	case ns >= 1e6:
		return fmt.Sprintf("%.2f ms", float64(ns)/1e6)
	case ns >= 1e3:
		return fmt.Sprintf("%.1f µs", float64(ns)/1e3)
	}
	return fmt.Sprintf("%d ns", ns)
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: rendertables <bench-output-file>")
		os.Exit(2)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	goRows := map[string]map[string]goCell{}
	pyRows := map[string]map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		if m := goLine.FindStringSubmatch(line); m != nil {
			name := m[1]
			if m[2] != "" {
				name += "/" + m[2]
			}
			ns, _ := strconv.ParseInt(m[4], 10, 64)
			if goRows[name] == nil {
				goRows[name] = map[string]goCell{}
			}
			goRows[name][m[3]] = goCell{ns: ns, extra: m[5]}
			continue
		}
		if m := pyLine.FindStringSubmatch(line); m != nil {
			key := m[1] + "/" + m[2]
			if pyRows[key] == nil {
				pyRows[key] = map[string]string{}
			}
			pyRows[key][m[3]] = m[4]
		}
	}

	variants := []string{string(backend.Ref), string(backend.CPU), string(backend.Metal), string(backend.Vulkan)}
	names := sortedKeys(goRows)
	fmt.Println("| Workload | goai-ref | goai-cpu | goai-metal | goai-vulkan | industry (same box) |")
	fmt.Println("|---|---|---|---|---|---|")
	var decode []string
	for _, name := range names {
		row := goRows[name]
		hasVariant := false
		cells := make([]string, 0, len(variants))
		for _, v := range variants {
			if c, ok := row[v]; ok {
				hasVariant = true
				cell := fmtNs(c.ns)
				if c.extra != "" {
					cell += " (" + c.extra + ")"
				}
				cells = append(cells, cell)
			} else {
				cells = append(cells, "—")
			}
		}
		if !hasVariant {
			decode = append(decode, name)
			continue
		}
		industry := ""
		base := strings.ToLower(strings.SplitN(name, "/", 2)[0])
		if len(base) > 6 {
			base = base[:6]
		}
		for _, pk := range sortedKeys(pyRows) {
			if strings.HasPrefix(strings.ToLower(pk), base) {
				parts := make([]string, 0, 3)
				for _, st := range sortedKeys(pyRows[pk]) {
					parts = append(parts, st+": "+pyRows[pk][st])
				}
				industry = strings.Join(parts, "; ")
				break
			}
		}
		fmt.Printf("| %s | %s | %s |\n", name, strings.Join(cells, " | "), industry)
	}
	if len(decode) > 0 {
		fmt.Println()
		fmt.Println("| Decode workload | variant | rate |")
		fmt.Println("|---|---|---|")
		for _, name := range decode {
			for _, v := range sortedKeys(goRows[name]) {
				c := goRows[name][v]
				rate := c.extra
				if rate == "" {
					rate = fmtNs(c.ns)
				}
				fmt.Printf("| %s | %s | %s |\n", name, v, rate)
			}
		}
	}
	fmt.Println()
	fmt.Println("| Python stack workload | numpy-cpu | torch-cpu | torch-mps |")
	fmt.Println("|---|---|---|---|")
	for _, name := range sortedKeys(pyRows) {
		st := pyRows[name]
		fmt.Printf("| %s | %s | %s | %s |\n", name, cell(st, "numpy-cpu"), cell(st, "torch-cpu"), cell(st, "torch-mps"))
	}
}

func cell(m map[string]string, k string) string {
	if v, ok := m[k]; ok {
		return v
	}
	return "—"
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
