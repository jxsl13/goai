# ADR-0020: PagedAttention is out of scope (revisit trigger defined)

Date: 2026-07-13 · Status: accepted · §T560

## Context

PagedAttention (Kwon et al. 2023, vLLM, arXiv:2309.06180) manages the KV cache
as fixed-size blocks addressed through a block table, eliminating fragmentation
and enabling prefix sharing. It is the de-facto standard in **multi-tenant
serving engines** — its wins come from many concurrent sequences competing for
one memory pool.

## Decision

Not implemented. GoAI's inference surface is single-sequence: the batched GPU
decoders (`llamagpu`) hold one device-resident KV cache per decoder, the
analysis-path caches are per-model, and there is no request scheduler or
continuous-batching engine that would consume a block table. Building the
block-table machinery without that consumer would create exactly the orphan
infrastructure the integration audits (§T444, §T519) exist to prevent.

The single-sequence memory concerns PagedAttention incidentally touches are
already served: bounded sliding-window caches with attention sinks
(StreamGenerate), eviction policies (kvevict, SnapKV, PyramidKV), and an 8-bit
quantized KV cache.

## Revisit trigger

If a multi-sequence serving engine (continuous batching, concurrent decoders
over one GPU memory pool) becomes a project goal, PagedAttention becomes its
FIRST building block — reopen via a §T task citing this ADR.
