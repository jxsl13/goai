## §GAP — vendor-BLAS gap on this Zen3 (torch-cpu 2.13, numpy 2.4.4/OpenBLAS)

GAP-1 (1024³ GFLOP/s): F64 goai 84 | torch 177 | numpy 227 → ≈2.7×. F32 goai(f32-native nr16) 153 | torch 580 | numpy 485 → ≈3.8× (was 13× vs scalar 43).
GAP-2: F64 gap partly = bit-exact `Mul`+`Add` (≈2× of FMA peak). CACHE BLOCKING re-measured on this Zen3 (ADR-0017 resume condition): packed-B REGRESSED (512 −16%, 1024 −6%) → DISCARDED, x86 resume condition CLOSED with data (kernel ⊥ cache-capacity/B-read-bound; B fits L3). remaining ≈3× vendor gap = microkernel saturation + §V10 f64-accum policy, ⊥ blocking.
GAP-3 (thread finding): torch FASTER at 8 threads than 16 on 8c/16t (SMT contention, compute-bound GEMM). BUT goai GEMM is SLOWER at 8 than 16 (69 vs 81 GFLOP/s) — its less-saturated kernel benefits from SMT hiding stalls → ⊥ cap parallelWork at physical cores (measured negative).
