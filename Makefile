# GoAI — build & verification targets. Pure-Go-first (§C), CGO_ENABLED=0 default.
GO       ?= go
PKGS     ?= ./...
CGO_OFF   = CGO_ENABLED=0

.PHONY: all build vet test race bench fmt tidy ci golden simd-build clean apicheck \
	metal-test metal-bench cuda-test vulkan-spv vulkan-test vulkan-bench

all: build vet apicheck test

## build: Pure-Go build, no C toolchain (§V7).
build:
	$(CGO_OFF) $(GO) build $(PKGS)

## vet: static checks.
vet:
	$(CGO_OFF) $(GO) vet $(PKGS)

## apicheck: public-API doc+example gate (§C13, §V19) — every exported symbol
## documented, every user-facing package has a runnable Example. Run before
## commit/push; also part of `all` and `ci`.
apicheck:
	$(CGO_OFF) $(GO) test ./internal/apicheck/

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

## metal-test: build+test the Metal/MPS GPU backend (darwin, cgo — NO build tag
## needed, §T47). Skips at runtime if no GPU (§V4).
metal-test:
	CGO_ENABLED=1 $(GO) test ./backend/metal/

## metal-bench: the §C3 cgo-gate benchmarks (metal vs optimized Pure-Go cpu).
metal-bench:
	CGO_ENABLED=1 $(GO) test ./backend/metal/ -run '^$$' -bench MatMulF32 -benchtime 10x

## cuda-test: build+test the optional CUDA/cuBLAS GPU backend (linux/windows,
## cgo, §T42). Needs the CUDA toolkit; skips at runtime if no GPU (§V4).
cuda-test:
	$(GO) test -tags cuda ./backend/cuda/

## vulkan-spv: compile the compute shader to SPIR-V (needs glslc from the Vulkan
## SDK / shaderc). The committed .spv is a build artifact embedded by vulkan.go;
## regenerate it here after editing matmul.comp (§T43).
vulkan-spv:
	glslc backend/vulkan/shaders/matmul.comp -o backend/vulkan/shaders/matmul.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm.comp -o backend/vulkan/shaders/coopmat_gemm.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm_tiled.comp -o backend/vulkan/shaders/coopmat_gemm_tiled.spv
	glslc backend/vulkan/shaders/matmul_strided.comp -o backend/vulkan/shaders/matmul_strided.spv
	glslc backend/vulkan/shaders/softmax_causal.comp -o backend/vulkan/shaders/softmax_causal.spv
	glslc backend/vulkan/shaders/crossentropy.comp -o backend/vulkan/shaders/crossentropy.spv
	glslc backend/vulkan/shaders/embed.comp -o backend/vulkan/shaders/embed.spv
	glslc --target-env=vulkan1.1 backend/vulkan/shaders/mha_decode.comp -o backend/vulkan/shaders/mha_decode.spv
	glslc backend/vulkan/shaders/qmatmul_q8.comp -o backend/vulkan/shaders/qmatmul_q8.spv
	glslc backend/vulkan/shaders/qmatmul_q4_0.comp -o backend/vulkan/shaders/qmatmul_q4_0.spv
	glslc backend/vulkan/shaders/qmatmul_q4k.comp -o backend/vulkan/shaders/qmatmul_q4k.spv
	glslc backend/vulkan/shaders/qmatmul_q6k.comp -o backend/vulkan/shaders/qmatmul_q6k.spv
	glslc backend/vulkan/shaders/qmatmul_q5k.comp -o backend/vulkan/shaders/qmatmul_q5k.spv
	glslc backend/vulkan/shaders/qmatmul_q2k.comp -o backend/vulkan/shaders/qmatmul_q2k.spv
	glslc backend/vulkan/shaders/qmatmul_q3k.comp -o backend/vulkan/shaders/qmatmul_q3k.spv
	glslc backend/vulkan/shaders/mha.comp -o backend/vulkan/shaders/mha.spv
	glslc backend/vulkan/shaders/flashattn.comp -o backend/vulkan/shaders/flashattn.spv
	glslc backend/vulkan/shaders/retention.comp -o backend/vulkan/shaders/retention.spv
	glslc backend/vulkan/shaders/retention_bwd.comp -o backend/vulkan/shaders/retention_bwd.spv
	glslc backend/vulkan/shaders/mha_bwd.comp -o backend/vulkan/shaders/mha_bwd.spv
	glslc backend/vulkan/shaders/im2col.comp -o backend/vulkan/shaders/im2col.spv
	glslc backend/vulkan/shaders/colout.comp -o backend/vulkan/shaders/colout.spv
	glslc backend/vulkan/shaders/conv2d_bwd.comp -o backend/vulkan/shaders/conv2d_bwd.spv
	glslc backend/vulkan/shaders/conv_igemm.comp -o backend/vulkan/shaders/conv_igemm.spv
	glslc backend/vulkan/shaders/rmsnorm.comp -o backend/vulkan/shaders/rmsnorm.spv
	glslc backend/vulkan/shaders/rmsnorm_bwd.comp -o backend/vulkan/shaders/rmsnorm_bwd.spv
	glslc backend/vulkan/shaders/crossentropy_bwd.comp -o backend/vulkan/shaders/crossentropy_bwd.spv
	glslc backend/vulkan/shaders/embed_bwd.comp -o backend/vulkan/shaders/embed_bwd.spv
	glslc backend/vulkan/shaders/rope.comp -o backend/vulkan/shaders/rope.spv
	glslc backend/vulkan/shaders/rope2.comp -o backend/vulkan/shaders/rope2.spv
	glslc backend/vulkan/shaders/rope_bwd.comp -o backend/vulkan/shaders/rope_bwd.spv
	glslc backend/vulkan/shaders/softmax.comp -o backend/vulkan/shaders/softmax.spv
	glslc backend/vulkan/shaders/layernorm.comp -o backend/vulkan/shaders/layernorm.spv
	glslc backend/vulkan/shaders/layernorm_bwd.comp -o backend/vulkan/shaders/layernorm_bwd.spv
	glslc backend/vulkan/shaders/unary.comp -o backend/vulkan/shaders/unary.spv
	glslc backend/vulkan/shaders/binary.comp -o backend/vulkan/shaders/binary.spv
	glslc backend/vulkan/shaders/addbias.comp -o backend/vulkan/shaders/addbias.spv
	glslc backend/vulkan/shaders/gelu_bwd.comp -o backend/vulkan/shaders/gelu_bwd.spv
	glslc backend/vulkan/shaders/silu_bwd.comp -o backend/vulkan/shaders/silu_bwd.spv
	glslc backend/vulkan/shaders/addbias_bwd.comp -o backend/vulkan/shaders/addbias_bwd.spv

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

## bench-compare: cross-backend comparison micro-benchmarks — every op timed on
## the reference, cpu, Metal, and Vulkan side by side (needs darwin+cgo+MoltenVK).
bench-compare: vulkan-spv
	VK_ICD_FILENAMES=$(VK_MOLTENVK_ICD) $(GO) test -tags vulkan ./internal/benchcompare/ -run '^$$' -bench . -benchtime 30x

## bench-python: the SAME ops timed against the common Python stacks (NumPy +
## PyTorch CPU/MPS) so GoAI can be compared to torch-cpu / torch-mps. Needs the
## project .venv (numpy + torch).
bench-python:
	.venv/bin/python internal/benchcompare/python_compare.py

## lint-md: dependency-free markdown lint — discovers every *.md recursively (SPEC T612).
## Also runs as a test in ./internal/mdlint (CI-enforced on code pushes).
lint-md:
	$(GO) run ./internal/mdlint ./...

## install-hooks: wire lint-md as a git pre-commit hook.
install-hooks:
	printf '#!/bin/sh\nmake lint-md || exit 1\n' > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

## golden: regenerate reference values from NumPy/torch (§V1,§V13).
## Uses a local venv (PEP 668). Bootstrap: python3 -m venv .venv && .venv/bin/pip install numpy
golden:
	.venv/bin/python testdata/gen.py

fmt:
	$(GO) fmt $(PKGS)

tidy:
	$(GO) mod tidy

## ci: the gate that must stay green on macOS/Windows/Linux (§V4). Includes the
## public-API doc+example gate (§V19) — blocks commit/push on undocumented API.
ci: build vet apicheck test

clean:
	$(GO) clean $(PKGS)
