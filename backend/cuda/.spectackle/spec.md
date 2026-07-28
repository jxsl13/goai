---
schema: v1
prefix: CUDA
---

## CUDA-005
The CUDA backend SHALL sit behind -tags cuda with cgo on linux or windows, and be validated on a physical GPU before push, since CI has no GPU runner.

Rationale: A green CI board is not CUDA coverage. Migrated from the worker spec Iw3.

## CUDA-006
The caller-owned device-buffer allocator in cuda_bridge.c SHALL route its returned pointer through track() into the gLiveBufs counter, which cu_free_f32 decrements.

Rationale: The counter measures buffers rather than bytes because the cudaMallocAsync mempool caches freed memory, so cudaMemGetInfo cannot observe a release. Non-owning view handles must not decrement on Free. Migrated from the worker spec Iw7.

## CUDA-007
WHEN a new cgo-touching public operation or allocator is added, the pull request SHALL add a noLeak case in cuda_leak_test.go that warms up once, snapshots LiveBufs(), runs N iterations and asserts a delta of zero.

Rationale: The warm-up absorbs process-lifetime lazy caches, so a per-call leak grows with N while a one-time cache stays at zero. A deliberate-leak sanity test guards the detector itself against false negatives. Migrated from the worker spec Iw7.

## CUDA-008
WHEN GPU work is scheduled on the worker, the loop SHALL serialize it against other GPU consumers and check nvidia-smi --query-compute-apps first, since a killed llama-cli holds VRAM.

Rationale: A leaked process was observed holding 9.9 GB. The CUDA environment script must be sourced before any -tags cuda build or test. Migrated from the worker spec RUN6.
