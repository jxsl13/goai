// Command capprobe prints this machine's acceleration surface in go-test-shaped
// lines (§T601): which GPU APIs and SIMD classes are actually present. CI runs
// it FIRST in every accelerated lane, so the log shows what the runner offers
// BEFORE any test executes — and a skipped suite is explainable at a glance.
//
// In plain terms: before running GPU tests, print whether this machine even
// has the GPU (or instruction set) those tests need.
//
// Output format (tab columns like `go test`):
//
//	cap 	goos/goarch	darwin/arm64
//	cap 	metal    	available
//	cap 	vulkan   	ICD present (lavapipe)
//	cap 	cuda     	nvidia-smi not found
package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func main() {
	fmt.Printf("cap \tgoos/goarch\t%s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("cap \tsimd\t%s\n", simdClass())
	fmt.Printf("cap \tmetal\t%s\n", metalStatus())
	fmt.Printf("cap \tvulkan\t%s\n", vulkanStatus())
	fmt.Printf("cap \tcuda\t%s\n", cudaStatus())
}

// simdClass names the vector ISA the pure-Go SIMD paths can target here.
func simdClass() string {
	switch runtime.GOARCH {
	case "arm64":
		return "NEON (arm64 baseline)"
	case "amd64":
		if flags, err := os.ReadFile("/proc/cpuinfo"); err == nil {
			s := string(flags)
			switch {
			case strings.Contains(s, "avx512"):
				return "AVX-512"
			case strings.Contains(s, "avx2"):
				return "AVX2"
			case strings.Contains(s, "sse4"):
				return "SSE4"
			}
		}
		return "amd64 (feature flags not readable on this OS)"
	}
	return runtime.GOARCH + " (no SIMD path)"
}

// vulkanStatus reports whether a loadable ICD answers vulkaninfo.
func vulkanStatus() string {
	out, err := exec.Command("vulkaninfo", "--summary").Output()
	if err != nil {
		return "no loadable ICD (vulkaninfo failed or not installed)"
	}
	if i := strings.Index(string(out), "deviceName"); i >= 0 {
		line := string(out)[i:]
		if j := strings.IndexByte(line, '\n'); j >= 0 {
			line = line[:j]
		}
		return "ICD present (" + strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "deviceName")) + ")"
	}
	return "ICD present"
}

// cudaStatus reports whether the NVIDIA driver answers nvidia-smi.
func cudaStatus() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return "nvidia-smi not found (no CUDA driver)"
	}
	return "driver present (" + strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0]) + ")"
}
