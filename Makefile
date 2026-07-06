# GoAI — build & verification targets. Pure-Go-first (§C), CGO_ENABLED=0 default.
GO       ?= go
PKGS     ?= ./...
CGO_OFF   = CGO_ENABLED=0

.PHONY: all build vet test race bench fmt tidy ci golden simd-build clean \
	metal-test metal-bench cuda-test vulkan-spv vulkan-test vulkan-bench

all: build vet test

## build: Pure-Go build, no C toolchain (§V7).
build:
	$(CGO_OFF) $(GO) build $(PKGS)

## vet: static checks.
vet:
	$(CGO_OFF) $(GO) vet $(PKGS)

## test: unit + golden + property tests (§V1,§V2,§V6).
test:
	$(CGO_OFF) $(GO) test $(PKGS)

## race: test under the race detector (needs cgo; local dev only).
race:
	$(GO) test -race $(PKGS)

## bench: benchmarks (§V5). Records numbers for the cgo gate (§C3).
bench:
	$(CGO_OFF) $(GO) test -run=^$$ -bench=. -benchmem $(PKGS)

## simd-build: build with the experimental SIMD backend (AMD64, §T11).
simd-build:
	GOEXPERIMENT=simd $(CGO_OFF) $(GO) build -tags=simd $(PKGS)

## metal-test: build+test the optional Metal/MPS GPU backend (darwin, cgo, §T20).
metal-test:
	$(GO) test -tags metal ./backend/metal/

## metal-bench: the §C3 cgo-gate benchmarks (metal vs optimized Pure-Go cpu).
metal-bench:
	$(GO) test -tags metal ./backend/metal/ -run '^$$' -bench MatMulF32 -benchtime 10x

## cuda-test: build+test the optional CUDA/cuBLAS GPU backend (linux/windows,
## cgo, §T42). Needs the CUDA toolkit; skips at runtime if no GPU (§V4).
cuda-test:
	$(GO) test -tags cuda ./backend/cuda/

## vulkan-spv: compile the compute shader to SPIR-V (needs glslc from the Vulkan
## SDK / shaderc). The committed .spv is a build artifact embedded by vulkan.go;
## regenerate it here after editing matmul.comp (§T43).
vulkan-spv:
	glslc backend/vulkan/shaders/matmul.comp -o backend/vulkan/shaders/matmul.spv

## vulkan-test: build+test the optional portable Vulkan compute backend (cgo,
## §T43). On macOS the backend runs on the GPU via MoltenVK — point the loader at
## its ICD (brew path); native Linux/Windows auto-discover the driver. Skips at
## runtime if no compute device (§V4).
VK_MOLTENVK_ICD ?= /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
vulkan-test: vulkan-spv
	VK_ICD_FILENAMES=$(VK_MOLTENVK_ICD) $(GO) test -tags vulkan ./backend/vulkan/

## vulkan-bench: the §C3 cgo-gate benchmarks (vulkan compute vs optimized Pure-Go).
vulkan-bench: vulkan-spv
	VK_ICD_FILENAMES=$(VK_MOLTENVK_ICD) $(GO) test -tags vulkan ./backend/vulkan/ -run '^$$' -bench MatMulF32 -benchtime 10x

## golden: regenerate reference values from NumPy/torch (§V1,§V13).
## Uses a local venv (PEP 668). Bootstrap: python3 -m venv .venv && .venv/bin/pip install numpy
golden:
	.venv/bin/python testdata/gen.py

fmt:
	$(GO) fmt $(PKGS)

tidy:
	$(GO) mod tidy

## ci: the gate that must stay green on macOS/Windows/Linux (§V4).
ci: build vet test

clean:
	$(GO) clean $(PKGS)
