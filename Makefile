# GoAI — build & verification targets. Pure-Go-first (§C), CGO_ENABLED=0 default.
GO       ?= go
PKGS     ?= ./...
CGO_OFF   = CGO_ENABLED=0

.PHONY: all build vet test race bench fmt tidy ci golden simd-build clean apicheck \
	metal-test metal-bench cuda-test vulkan-spv vulkan-test vulkan-bench \
	perfscan perfscan-check preflight preflight-full install-hooks

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

## cuda-cubin: nvcc-compile the .cu tensor-core kernels (mma.h/WMMA — nvrtc cannot) to
## committed fatbins, loaded at runtime via cuModuleLoadDataEx. Needs nvcc + a supported host
## compiler: source scripts/cuda-nvcc-env.sh (pip nvidia-cuda-nvcc-cu13 + brew gcc@14). sm_86
## SASS + PTX fallback. Regenerate after editing a kernel; the fatbin is a build artifact like
## the vulkan .spv. (Tw-FLASHATTN.)
cuda-cubin:
	@. scripts/cuda-nvcc-env.sh && nvcc_86 --fatbin -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 backend/cuda/kernels/wmma_gemm.cu -o backend/cuda/kernels/wmma_gemm.fatbin
	@. scripts/cuda-nvcc-env.sh && nvcc_86 --fatbin -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 backend/cuda/kernels/wmma_attn.cu -o backend/cuda/kernels/wmma_attn.fatbin
	@. scripts/cuda-nvcc-env.sh && nvcc_86 --fatbin -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 backend/cuda/kernels/wmma_attn_gqa.cu -o backend/cuda/kernels/wmma_attn_gqa.fatbin
	@. scripts/cuda-nvcc-env.sh && nvcc_86 --fatbin -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 backend/cuda/kernels/wmma_paged_decode.cu -o backend/cuda/kernels/wmma_paged_decode.fatbin

## vulkan-spv: compile the compute shader to SPIR-V (needs glslc from the Vulkan
## SDK / shaderc). The committed .spv is a build artifact embedded by vulkan.go;
## regenerate it here after editing matmul.comp (§T43).
vulkan-spv:
	glslc backend/vulkan/shaders/matmul.comp -o backend/vulkan/shaders/matmul.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm.comp -o backend/vulkan/shaders/coopmat_gemm.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm_tiled.comp -o backend/vulkan/shaders/coopmat_gemm_tiled.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm_v2.comp -o backend/vulkan/shaders/coopmat_gemm_v2.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm_v3.comp -o backend/vulkan/shaders/coopmat_gemm_v3.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm_cm2.comp -o backend/vulkan/shaders/coopmat_gemm_cm2.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm_f16acc.comp -o backend/vulkan/shaders/coopmat_gemm_f16acc.spv
	glslc --target-env=vulkan1.3 backend/vulkan/shaders/coopmat_gemm_v4.comp -o backend/vulkan/shaders/coopmat_gemm_v4.spv
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

## bench-classic-python: time scikit-learn's fit for the six classical scorecard
## methods on the IDENTICAL data classic/perfcompare_test.go writes, so GoAI's
## classical learners can be compared head-to-head (BENCHMARKS.md §5). Needs the
## project .venv (scikit-learn + numpy). Prints GoAI then scikit-learn. T881.
bench-classic-python:
	@dir=$$(mktemp -d); \
	PERF_CSV_DIR=$$dir $(GO) test ./classic/ -run TestPerfCompareVsSklearn -count=1 -v | grep GOAI_FIT; \
	PERF_CSV_DIR=$$dir .venv/bin/python testdata/bench_sklearn.py; \
	rm -rf $$dir

## bench-gpt-train-python: time a GPT training step (fwd+CE+bwd) in PyTorch at the
## SAME geometry as internal/benchcompare BenchmarkGPTTrainingStep, for the GoAI-vs-
## torch end-to-end training comparison (BENCHMARKS.md §6). Needs .venv (torch). T883.
bench-gpt-train-python:
	.venv/bin/python testdata/bench_gpt_train_torch.py

## bench-safetensors-load: time safetensors model-file loading (full + one-tensor)
## in GoAI vs the Rust-cored safetensors-python on a shared fixture, for the pure-Go
## reader comparison (BENCHMARKS.md losses table). Needs .venv (safetensors). T885.
bench-safetensors-load:
	@f=$$(mktemp -u).safetensors; \
	ST_BENCH_FILE=$$f .venv/bin/python testdata/bench_safetensors_load.py; \
	ST_BENCH_FILE=$$f $(GO) test ./format/safetensors -run TestSafetensorsLoadCompare -count=1 -v | grep GOAI_LOAD; \
	rm -f $$f

## bench-gguf-load: time GGUF model-file loading in GoAI (gguf.ReadFile) vs gguf-py's
## GGUFReader on a shared F32 fixture, for the pure-Go reader comparison (BENCHMARKS.md
## losses table, T885). Needs .venv (gguf). T907 tracks the F32/F16 fast-path lever.
bench-gguf-load:
	@f=$$(mktemp -u).gguf; \
	GGUF_BENCH_FILE=$$f .venv/bin/python testdata/bench_gguf_load.py; \
	GGUF_BENCH_FILE=$$f $(GO) test ./internal/benchcompare -run TestGGUFLoadCompare -count=1 -v | grep GOAI_GGUF_LOAD; \
	rm -f $$f

## bench-vision-python: time a ViT and a CNN (forward + training step) in GoAI
## (cpu-simd/Metal/Vulkan) and PyTorch (cpu/mps) at identical geometry, for the vision
## comparison (BENCHMARKS.md §7). Needs .venv (torch) + MoltenVK. T884.
bench-vision-python:
	VK_ICD_FILENAMES=$(VK_MOLTENVK_ICD) GOEXPERIMENT=simd $(GO) test -tags vulkan ./internal/benchcompare -run '^$$' -bench 'BenchmarkViT|BenchmarkCNN' -benchtime 8x | grep -E 'Benchmark(ViT|CNN)'
	.venv/bin/python testdata/bench_vision_torch.py

## bench-mlx: time Apple's MLX framework decoding a native-4-bit TinyLlama on the M2
## Metal GPU, for the T887 three-way Apple head-to-head (§2). Needs .venv (mlx-lm) and a
## one-time convert: .venv/bin/python -m mlx_lm convert --hf-path
## TinyLlama/TinyLlama-1.1B-Chat-v1.0 -q --q-bits 4 --mlx-path models/tinyllama-mlx-4bit
bench-mlx:
	.venv/bin/python testdata/bench_mlx.py

## lint-md: dependency-free markdown lint — discovers every *.md recursively (SPEC T612).
## Also runs as a test in ./internal/mdlint (CI-enforced on code pushes).
lint-md:
	$(GO) run ./internal/mdlint ./...

## spec-check: §V36 SPEC-integrity guard — id-uniqueness (across SPEC.md + worker
## specs), §T-membership, §T-last. A Go test in internal/speccheck; also runs on
## every non-empty CI selection via the cichange always-run list (T886/T893).
spec-check:
	$(CGO_OFF) $(GO) test ./internal/speccheck/

## spec-graph: query the SPEC.md/docs citation graph (T915) — e.g.
## make spec-graph ARGS='why V22' | ARGS='bugs-for format/gguf' | ARGS='patterns'.
## The graph rebuilds from the corpus + git log; .docgraph/ only caches.
spec-graph:
	$(CGO_OFF) $(GO) run ./internal/docgraph $(ARGS)

## spec-graph-check: the docgraph test suite (fixture extraction, real-corpus
## golden edges, determinism, cache roundtrip, speccheck-drift pin).
spec-graph-check:
	$(CGO_OFF) $(GO) test ./internal/docgraph/

## spec-render: regenerate the SPEC.md + SPEC-worker-*.md views from spec/ (§V41).
spec-render:
	$(CGO_OFF) $(GO) run ./internal/docgraph spec render

## spec-verify: render-sync + §V36 speccheck + table shapes + §V39 dangling
## strong refs, plus the full markdown lint. The local pre-push spec gate.
spec-verify:
	$(CGO_OFF) $(GO) run ./internal/docgraph spec verify
	$(GO) run ./internal/mdlint ./...

## perfscan: static finder for the per-element hot-loop anti-patterns (T919) —
## per-element AtF64/SetF64 with no flatF64/flatF32 fast path, allocation inside a
## per-element loop, batch API fed a wrapped single row. Advisory (candidates need
## an A/B measurement, §C3/§V22); -strict exits non-zero. e.g. make perfscan
## ARGS='-strict ./nn/...'. The detectors are unit-tested in internal/perfscan.
perfscan:
	$(CGO_OFF) $(GO) run ./internal/perfscan $(ARGS)

## perfscan-check: the perfscan detector test suite (positive + negative fixtures).
perfscan-check:
	$(CGO_OFF) $(GO) test ./internal/perfscan/

## preflight: the local PRE-PUSH gate — every HARD CI check runnable without a
## C/CUDA/Vulkan toolchain, fail-fast (§V23). Mirrors ci.yml: gofmt (the cgo+race
## lane), go vet ./... (all lanes' §V23 soundness backstop, compiles every _test.go),
## the -short test suite INCLUDING the always-run meta-tests speccheck / perfscan /
## docgraph render-sync (pure-go lane + §V41), and the go-mod-tidy drift gate. The
## -short suite self-skips the trained-model e2e tests, so this is ~seconds. It runs
## every package EXCEPT internal/mdlint — the one gate CI still holds out of always-run
## as known-red debt (mdlint reddens on unrelated worker markdown). apicheck is now
## GREEN and IN the gate (its doc/example debt was cleared), matching CI. The
## cgo/metal/cuda/vulkan/simd COMPILE lanes need toolchains and run in CI; add the
## locally-available ones with `make preflight-full`.
preflight:
	@echo "→ gofmt (tracked *.go)"
	@bad=$$(gofmt -l $$(git ls-files '*.go')); if [ -n "$$bad" ]; then echo "unformatted — run gofmt -w:"; echo "$$bad"; exit 1; fi
	@echo "→ CGO_ENABLED=0 go build ./..."
	@$(CGO_OFF) $(GO) build ./...
	@echo "→ CGO_ENABLED=0 go vet ./...  (compiles every test, §V23)"
	@$(CGO_OFF) $(GO) vet ./...
	@echo "→ go mod tidy drift gate"
	@$(CGO_OFF) $(GO) mod tidy && git diff --exit-code -- go.mod go.sum || { echo "go.mod/go.sum drift — commit the tidy result"; exit 1; }
	@echo "→ CGO_ENABLED=0 go test -short  (buildable pure-go packages incl. apicheck; only mdlint held out)"
	@$(CGO_OFF) $(GO) test -short -count=1 $$($(CGO_OFF) $(GO) list -e -f '{{if or .GoFiles .TestGoFiles}}{{.ImportPath}}{{end}}' ./... | grep -vE 'internal/(mdlint)$$')
	@echo "✓ preflight OK — cgo/metal/cuda/vulkan/simd lanes run in CI (or: make preflight-full)"

## preflight-full: preflight + the remaining CI lanes runnable on THIS machine — the
## §V5 benchmark smoke (build+run each once) and the simd build (soft lane). cgo/metal
## (needs CGO), vulkan (needs an ICD) and cuda (needs the CUDA toolkit) stay explicit
## opt-ins via their own targets (make metal-test / vulkan-test / cuda-test).
preflight-full: preflight
	@echo "→ benchmark smoke (§V5, -benchtime=1x)"
	@$(CGO_OFF) $(GO) test ./... -run='^$$' -bench=. -benchtime=1x -timeout 15m >/dev/null
	@echo "→ simd build (GOEXPERIMENT=simd, soft lane)"
	@GOEXPERIMENT=simd $(CGO_OFF) $(GO) build -tags=simd ./...
	@echo "✓ preflight-full OK"

## install-hooks: wire `make preflight` as the git PRE-PUSH hook — CI runs on push, so
## the comprehensive gate belongs there (commits stay fast). Retires the old lint-md
## pre-commit hook (CI does not gate markdown; it reddened on unrelated worker files).
## Re-run after editing the preflight target.
install-hooks:
	rm -f .git/hooks/pre-commit
	printf '#!/bin/sh\nexec make preflight\n' > .git/hooks/pre-push
	chmod +x .git/hooks/pre-push
	@echo "wired .git/hooks/pre-push → make preflight (retired the lint-md pre-commit)"

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
