# GoAI — Landschafts- & Machbarkeits-Report (Bootstrap Phase 0)

> Erstellt: 2026-07-05 · Ziel-Go-Version: 1.26 · Policy: cgo-last (Pure Go zuerst)
> Status: Iteration 1 des autonomen Loops. Quellen unten. `?` = noch unbestätigt,
> in Phase 1 (`/research`) zu härten.
>
> Hinweis zur Methodik: Der `deep-research`-Workflow scheiterte an einem
> Harness-Schemafehler (StructuredOutput-Retry-Cap). Gemäß Autonomie-Regel wurde
> umgeleitet auf direkte, gezielte WebSearch-Verifikation der versionssensiblen
> Fakten + Fachwissen. Nicht adversarial gegengeprüfte Behauptungen sind mit `?`
> markiert.

---

## 1. Stand der Technik in Go-ML — und die Lücke

| Projekt | Reifegrad / Wartung (2025–26) | Beschleunigung | Bewertung |
|---|---|---|---|
| **GoMLX** | Aktivster Go-ML-Stack, v0.26.0 (Dez 2025) | OpenXLA-JIT (CPU/GPU/TPU) **via `gopjrt` = cgo**; optionaler Pure-Go-Backend (unoptimiert), WASM | Stark, aber Peak-Performance hängt an C++-XLA; Pure-Go-Pfad ist nur Fallback |
| **Gorgonia** | Weitgehend **dormant/legacy**, funktional | Graph-basiert, teils cgo (CUDA) | Kein tragfähiges Fundament mehr |
| **gonum** | Aktiv, solide Numerik | Pure-Go **BLAS unvollständig**, nur float32/float64; optional cgo→OpenBLAS | Gute Basis für Numerik-Bausteine, aber kein DL/Autograd/GPU |
| **tract (Rust)** | Referenz für ONNX-Inferenz | — | Vergleichsmaßstab, kein Go |

**Konsequenz (die Lücke, die Neubau rechtfertigt):** Es gibt keinen
Go-nativen, **Pure-Go-first / cgo-last** Full-Spectrum-Stack, der (a) das neue
Go-1.26-`simd`-Paket als primären Beschleuniger ausreizt, (b) eine referenz-valide
Pure-Go-Wahrheit mit numerischer Parität garantiert und (c) cgo/GPU nur als
optionales, benchmarkgetriggertes Add-on führt. GoMLX kehrt die Priorität um
(XLA/cgo zuerst); Gorgonia ist tot; gonum deckt nur Numerik. → GoAI besetzt genau
diese Nische.

## 2. CPU-SIMD in reinem Go (der Kern der Pure-Go-Decke)

- **`simd/archsimd`** (Go 1.26, `GOEXPERIMENT=simd`, Release Feb 2026): erstmals
  explizite SIMD-Intrinsics **ohne cgo und ohne handgeschriebene asm-Stubs**.
  Vektortypen als Structs (`Int8x16`, `Float64x8`, …), 128/256/512-bit, **AVX2 +
  AVX-512**. **Aktuell AMD64-only.** API generiert vorerst immer AVX-Form;
  `X86`-Variable für Feature-Detection (AVX2/AVX512). Quelle: golang/go #73787,
  go1.26 Release Notes, pkg.go.dev/simd/archsimd. `?` API-Stabilität (experimentell,
  kann sich ändern).
- **Performance-Ranking (Okt 2025, Callista-Benchmark):** `simd`-Paket **inlined
  ~4× schneller** als die nächstbeste Lösung und ~16× vs. plain Go-Loop; `avo`
  ~3× vs. plain Loop (kann durch `.s`-Stub nicht inlinen); `simd` (nicht-inlined)
  ~30 % über avo. → **Reihenfolge der Wahl:** `simd`-Paket > `avo` > Plan9-Asm >
  Auto-Vektorisierung.
- **ARM64 (Apple Silicon, das Entwickler-Primärziel hier):** `simd`-Paket deckt
  ARM64/NEON **noch nicht** ab → dort Pure-Go via **Plan9-Asm (NEON)** oder
  `avo`-Äquivalent bzw. Fallback. `?` Zeitplan für ARM64 im `simd`-Paket.
- **Realistische Pure-Go-Decke:** Für BLAS-1/2 (elementwise, dot, axpy) und
  gut kachelbare GEMM ist mit `simd`+Blocking+Goroutinen ein **substanzieller
  Anteil an OpenBLAS/oneDNN** erreichbar; die exakte %-Zahl ist Op- und
  Hardware-abhängig und wird pro Kernel gemessen (setzt die §C-cgo-Schwelle).
- **Weitere Bausteine ohne cgo:** `math/bits`, `segmentio/asm`, `go-highway`
  (portable SIMD-Abstraktion mit Pure-Go-Fallback). `?` Reife von go-highway.

## 3. GPU-Wege aus Go — alle brauchen faktisch cgo

| Weg | cgo-Bedarf | Plattform | Anmerkung |
|---|---|---|---|
| CUDA / cuBLAS / cuDNN | **Ja** (C-Libs) | Linux, Windows | Höchste Peak-Perf für NVIDIA |
| Metal | **Ja** (Obj-C/cgo) | macOS | Pflicht für Apple-GPU/ANE-Nähe |
| Vulkan-Compute | **Ja** (Loader ist C) | portabel | Ein Backend für viele GPUs |
| WebGPU / wgpu | **Ja** (wgpu = Rust, via C-ABI) | portabel/WASM | Zukunftsträchtig, jung |
| ROCm / HIP | **Ja** | Linux | AMD |

**Befund:** Es gibt **keinen praktikablen Pure-Go-Weg zu diskreter GPU-Compute.**
Das ist kein Widerspruch zur Policy, sondern ihr Kern: GPU ist genau die Klasse,
in der cgo seinen Platz **nach** Ausschöpfung der Pure-Go-CPU-Decke verdient —
als optionales Build-Tag-Backend mit Pure-Go-Fallback. `?` Reifegrad einzelner
Go-Bindings (Metal-cgo, vulkan-go) in Phase 1 prüfen.

## 4. NPU / Accelerator — überwiegend ehrliches Nicht-Ziel (vorerst)

- **Apple Neural Engine (ANE):** nur indirekt über **CoreML** (cgo/Obj-C)
  ansprechbar; kein direkter Zugriff. Realistisch als spätes optionales Backend.
- **Windows DirectML:** über cgo/COM. Machbar, aber Aufwand.
- **Intel oneDNN (inkl. NPU-Pfade):** cgo.
- **Empfehlung:** NPU als **explizites Nicht-Ziel der ersten Ausbaustufe**
  markieren (kein stilles Versprechen); nach GPU-Reife re-evaluieren.

## 5. Referenz-Baselines für numerische Parität

- **BLAS/GEMM:** OpenBLAS / Eigen als Perf-Baseline; NumPy (`@`) als Korrektheits-
  Golden. Toleranz f64: rtol≈1e-12, f32: rtol≈1e-5 (`?` pro Op fixieren).
- **DL-Ops (Conv/Norm/Attention/Optimizer):** PyTorch/ATen als Golden-Quelle;
  Werte reproduzierbar per kleinem Python-Skript (torch, fixierter Seed) nach
  `testdata/golden/*.npy` exportieren → in Go via npy-Reader laden.
- **LLM-Inferenz:** llama.cpp/ggml als Perf- und Bit-Referenz (Quantisierung).
- **Golden-Erzeugung:** deterministisch (Seed, dtype, Shape dokumentiert),
  eingecheckt; Python 3.14 + NumPy/torch lokal vorhanden.

## 6. Modell-Interop-Formate

| Format | Pure-Go-Aufwand | Priorität |
|---|---|---|
| **safetensors** | Niedrig (JSON-Header + rohe Tensoren, zero-copy) | **Zuerst** |
| **GGUF** | Mittel (Header + Quant-Blöcke; Pure-Go-Reader existieren) | Für LLM-Inferenz |
| **ONNX** | Hoch (Protobuf-Schema + großes Opset) | Später, inkrementell nach Opset |
| HuggingFace | = safetensors + Tokenizer/Config | Mit NLP-Schicht |

## 7. Verifikations-Methodik

- **V-PARITY:** Golden-Tests gegen NumPy/torch in fixierten Toleranzen.
- **V-GRAD:** numerische Gradientenprüfung (central finite differences,
  Schwelle ~1e-4 rel.) für jede differenzierbare Op.
- **V-PROP:** Property-Based-Tests (Shape-Algebra, Linearität, Assoziativität wo
  mathematisch garantiert) via `testing/quick` bzw. rapid.
- **V-CROSS:** Backend-Ergebnis == Pure-Go-Referenz (Differential-Testing).
- **Fuzzing:** Go-native `go test -fuzz` für Shape-/Numerik-Randfälle.
- **CI-Matrix:** {macOS, Windows, Linux} × {Pure-Go-Fallback (immer) + verfügbarer
  Accel}; fehlender Accel ⇒ Skip mit Log, nie stiller Pass; `CGO_ENABLED=0`-Build
  muss überall grün bleiben (V-CGO).

---

## (a) Tragende Architektur-Wetten

1. **Pure-Go-`simd`-Paket als primärer Beschleuniger.** Es schlägt `avo` messbar
   und braucht keine C-Toolchain → es trägt die Pure-Go-Decke auf AMD64. ARM64
   über Plan9-NEON, bis das `simd`-Paket ARM64 unterstützt.
2. **Ein backend-agnostisches `Backend`/`Kernel`-Interface** mit Pure-Go-Referenz
   als Wahrheit; cgo/GPU/NPU nur als austauschbare, benchmarkgetriggerte Add-ons
   hinter Build-Tags. `CGO_ENABLED=0` bleibt vollwertig lauffähig.
3. **Golden-Parität als Abnahme.** Jede Op wird gegen NumPy/torch-Golden in
   fixierten Toleranzen abgenommen, bevor optimiert wird.
4. **GPU = cgo, bewusst und spät.** Da kein Pure-Go-GPU-Weg existiert, ist GPU
   der kanonische Ort, an dem der cgo-Gate greift — nach ausgereizter CPU-Decke.

## (b) Größte Risiken + Gegenmaßnahmen

| Risiko | Gegenmaßnahme |
|---|---|
| `simd`-Paket ist experimentell, API kann brechen | Hinter dünnem internem SIMD-Wrapper kapseln; Pure-Scalar-Fallback immer vorhanden; auf Go-1.26-Version pinnen |
| ARM64 (Apple Silicon = Dev-Host) fehlt im `simd`-Paket | Plan9-NEON-Kernels + Scalar-Fallback; Perf-Ziel dort separat messen |
| GPU/NPU zwingen cgo → Portabilitätsbruch | Strikte Build-Tag-Trennung, `CGO_ENABLED=0`-CI-Job als Pflicht-Gate |
| Golden aus torch nicht bit-reproduzierbar | Seeds/dtype/Shape fixieren, Toleranzen mit §R-Begründung, keine Aufweichung |
| Scope-Explosion (ganzes KI-Spektrum) | Strikte §T-Reihenfolge, ein auslieferbares Inkrement pro Task |

## (c) Empfohlene Bau-Reihenfolge

1. **L0 Core:** Tensor, Dtype (f32/f64 zuerst), Device, Allocator, Strides/Views.
2. **L1 Referenz-Compute (Pure Go, skalar):** elementwise, reduce, dot, GEMM +
   Golden-Tests + Bench-Harness. **Wahrheit, unoptimiert.**
3. **L1-Opt (separat):** GEMM/elementwise via `simd`-Paket (AMD64) / NEON (ARM64),
   Blocking, Goroutinen — gegen die Referenz aus (2), mit Benchmark-Delta.
4. **L2 Autograd:** Tape + VJP-Regeln der L1-Ops, V-GRAD.
5. **L3 NN:** Linear, Aktivierungen, Loss, SGD/Adam — end-to-end auf CPU.
6. **L5 IO:** safetensors zuerst.
7. **L1b GPU (cgo-Gate):** erstes GPU-Backend (Metal auf macOS / CUDA) als
   optionales Build-Tag-Backend, nur bei gerissener §C-Schwelle.
8. **L4 Domänen:** Transformer/LLM-Inferenz (GGUF), dann CV, klassisches ML, RL.

---

## Quellen

- [Go 1.26 Release Notes](https://go.dev/doc/go1.26)
- [golang/go #73787 — simd/archsimd intrinsics under GOEXPERIMENT](https://github.com/golang/go/issues/73787)
- [golang/go #76175 — simd CPU feature vet check](https://github.com/golang/go/issues/76175)
- [pkg.go.dev/simd/archsimd](https://pkg.go.dev/simd/archsimd)
- [Go 1.26 features overview — saraikin.com](https://saraikin.com/posts/go-1-26-features/)
- [Go 1.26 interactive tour — antonz.org](https://antonz.org/go-1-26/)
- [Go SIMD part 1 (Benchmark simd vs avo), Callista, Okt 2025](https://callistaenterprise.se/blogg/teknik/2025/10/20/trying-out-go-simd-support/)
- [go-highway — portable SIMD mit Pure-Go-Fallback](https://github.com/ajroetker/go-highway)
- [segmentio/asm](https://github.com/segmentio/asm)
- [gonum/blas](https://pkg.go.dev/gonum.org/v1/gonum/blas)
- [GoMLX (GitHub)](https://github.com/gomlx/gomlx) · [gopjrt](https://pkg.go.dev/github.com/gomlx/gopjrt)
- [Gorgonia (GitHub)](https://github.com/gorgonia/gorgonia)
