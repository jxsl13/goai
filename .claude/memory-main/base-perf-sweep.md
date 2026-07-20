---
name: base-perf-sweep
description: A base-perf audit must sweep the L0 tensor layer AND the per-token inference path, not just the compute kernels — the per-element Unravel-alloc / AtF64-SetF64-dispatch anti-pattern
metadata:
  node_type: memory
  type: feedback
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
---

„Optimierungen an der Basis multiplizieren sich über alle aufbauenden Layer"
(Nutzer-Prinzip 2026-07-14). Beim Umsetzen habe ich DREIMAL voreilig „Basis
komplett" geschlossen, nachdem ich nur die backend/cpu-Compute-Kernel devirt hatte
— die Nutzer-Beharrlichkeit deckte jedes Mal einen tieferen Layer auf, wo die
GRÖSSTEN Einzel-Wins lagen.

**Die Anti-Patterns** (überall dieselbe Ursache):
- `tensor.Unravel(pos, shape)` in einem per-Element-Loop → eine **Heap-Allocation
  pro Element** (make([]int, ndim)).
- `AtF64(idx...)` / `SetF64(v, idx...)` in einem per-Element-Loop → Dtype-Dispatch +
  flatOffset-Berechnung pro Element.
- Closures `f64at`/`f64set`/`get func(int)float64` pro Element (Compute-Kernel).

**Der Fix** (immer gleich): Dtype-Switch EINMAL → direkter typed `[]T`-Zugriff; für
strided Traversal einen laufenden Multi-Index mit INKREMENTELLEM Offset (kein
per-Element-Alloc). Parität bleibt (gleiche Reihenfolge).

**Wo die großen Wins lagen (die ich fast übersah):**
- L0 `tensor`: **Cast() 17×** (F64→F32-GPU-Prep, den jeder Tensor zahlt),
  Contiguous() 3× (jede Transpose-Materialisierung), broadcastContig 3,8×.
- Per-Token-Inferenz: **concatRows() 19×** (KV-Cache-Append pro Decode-Step),
  rowLogits 4,9× (Sampling), embedAt, QuantKVCache.stack.
- Compute-Kernel (der Teil, den ich zuerst fand): Aktivierungen/Norms/Softmax/
  Retention/Conv/Pool, bis 4,25×.

**Auch der TOKENIZER ist eine Basis** (§T625, 2026-07-14): `bpeMerge` (nlp/bpe.go)
läuft auf JEDEM Prompt, upstream aller Inferenz. War der naive O(L²)-BPE-Scan mit
`string(a)+string(b)` als Merge-Key pro Kandidatenpaar (3 String-Allocs/Paar/Iter).
Fix = tiktokens `byte_pair_merge`: nur Byte-OFFSETS in das immutable piece halten,
Paar-Rang via `ranks[string(piece[a:c])]`. **Der Schlüssel-Hebel: `m[string(bytes)]`
wird vom Go-Compiler alloc-frei spezialisiert** (kein String-Alloc, wenn die
Konvertierung nur als Map-Key dient) → NULL Map-Key-Allocs. **6,4×** (502→78,7µs),
Allocs 2084→601. Merge-Reihenfolge + Boundaries bit-identisch → tiktoken-Golden +
2,5M-Fuzz grün. Merke den `m[string(byteSlice)]`-Trick für JEDEN byte-gekeyten
Hot-Path-Map-Lookup (String-Konkatenation als Key verhindert die Optimierung).

**Backward-Pfad (Training) ist auch Basis** (§T632–T636, C25-Direktive „bis aufs
letzte Prozent devirtualisieren"): dieselben Anti-Patterns in autograd/vjp_*.go.
Wins: unaryVJP 14×, broadcastVJP (sum/mean) 10,8×, BLAS-1 (nrm2/axpy) 15×,
max/min/prod 7×, MoE 7×. Muster identisch (dtype-switch einmal → typed []T; für
Reduce ein `axStride`-Odometer statt per-Element-Unravel). Bei SATURIERTEM Kontext
(§C22): an frische-Kontext-Subagenten DELEGIEREN (ein vjp-File pro Agent, im Haupt-
Baum sequenziell, gradcheck als Gate) — hat für vjp_moe.go funktioniert.

**Backward-Sweep ABGESCHLOSSEN** (§T632–§T643): alle per-Layer/per-Token/groß-per-
Element-Backwards devirt (elementwise, reduce×5, BLAS-1, MoE, SSM, MLA, RWKV, Conv1D,
DoRA, IA3), F64+F32-verifiziert. **§C3-GRENZE (wichtig — ehrlich stoppen, nicht
mechanisch grinden):** die GRPO/KTO/SimPO/ORPO/DPO-Loss-Gradienten NICHT devirt — sie
loopen über die BATCH-Dim (b≤64) EINMAL pro Step → Dispatch amortisiert auf ≈0 vs die
Millionen per-Token-Elemente. Devirt davon = non-winning opt (§C3). Der Wert von
Devirt ist proportional zu (Elementzahl × Aufruf-Frequenz); ein `for i := range b`-
Loss-Gradient hat beides klein. Erkenne diese Grenze, statt jedes verbleibende AtF64
zu jagen.

**DELEGATION (§C22) funktioniert für den Sweep:** ein vjp-File pro frischem
Subagenten (im Haupt-Baum, sequenziell), F32-Parity-Test IM BRIEFING mandatieren,
gradcheck als Gate. Danach IMMER selbst reviewen (Diff-Spot-Check + Re-Run) — nicht
blind vertrauen. CAVEAT: ein durch API-Session-Limit ABGEBROCHENER Subagent lässt evtl.
eine unvollständige, unverifizierte Edit zurück (baut evtl., aber ungetestet =
Korrektheitsrisiko) → `git checkout -- <file>` verwerfen und inline/neu machen. LSP-
Diagnostics können nach Subagenten-Edits STALE sein (falsches „undefined X") — dem
echten `go build`/`go test` glauben, nicht dem LSP.

**FALLE — gradcheck ist F64-ONLY** (§T636): `TestGradCheckAllOps` (zentrale finite
Differenzen brauchen f64-Präzision) testet NUR den F64-Pfad. Ein F32-Fast-Path (den
man beim Devirtualisieren fast immer mit hinzufügt) ist damit UNGETESTET — ein
Transkriptionsfehler im F32-Zweig flösse STILL falsche Gradienten in F32-Training
(F32-Modelle treffen diese CPU-VJPs auch beim GPU-Training, da diese Ops keinen
Backend-Backward-Kernel haben). Benchmarks laufen den F32-Pfad zwar (kein Panic),
verifizieren aber nicht die Korrektheit. IMMER einen expliziten F32-vs-F64-Parity-
Test hinzufügen: F32-Pfad ≡ float32(F64-Pfad) ≤1e-3 (gültig, weil der F32-Zweig
intern in float64 rechnet und nur das Speichern rundet). Gilt für delegierte UND
eigene Devirts.

**OPTIMIZER-STEP-LOOPS waren ein übersehener Rung** (§T652/T653, 2026-07-15, C22-
blocked-on-push-window Perf-Audit): der Backward-Sweep devirt die VJPs, aber die
`Optimizer.Step`-Loops in nn/optim.go + nn/lamb/ademamix/adafactor/soap/shampoo.go
liefen NOCH per-Element AtF64/SetF64 (der Update pro Parameter-Element, jeden Step).
Derselbe Fix (flatF64/flatF32 typed contiguous Fast-Path, generic accessor als
Fallback, Arithmetik in float64 in der generic-Reihenfolge = BIT-IDENTISCH) gab
**7–15× auf dem isolierten Step** (Adam 7.8×, SGD 15.4×, Shampoo-diag 10.1×). Auch
der broadcast-reduce-VJP (bcastReduce = Bias-Grad-Backward jeder broadcast Add/Sub/Mul)
war noch per-Element → `bcastSumInto[T]` incremental-offset-Walk, 5×. Guard: neuer
`TestOptimizerFastPathParity` (fast contiguous vs generic Permute-view, EXACT-equal,
F64+F32) statt nur gradcheck. Merke: Step-Loops + VJPs sind BEIDE Hot-Path-Basen.

**Go-LEVEL FLOOR erreicht + wo er liegt** (§R246): nach Optimizer+VJP+broadcast-reduce
+ Compute-Kernel-Devirt ist der gemeinsame Trainings-Step am Go-Floor — verbleibende
AtF64/SetF64 sind KALTE Pfade (awq/gptq/dare/ssl), Architektur-spezifisch (retnet/rwkv/
mamba), ∨ intendierte Generic-Fallbacks. Der DOMINANTE Rest-Cost (pprof, small-batch):
cpu-Backend-Worker-Pool-SYNC (pthread_cond ≈78%) — Parallel-Dispatch-Overhead > Compute
für winzige per-Step-Kernel. Nächster Hebel liegt in backend/cpu (parThreshold anheben /
sub-threshold serialisieren), NICHT in mehr Go-Devirt. Erkenne den Floor, statt kalte
Pfade zu jagen (§C3).

**FORMAT/SERIALISIERUNG war noch ein übersehener Rung** (§T720/T721, 2026-07-16,
„beat all incumbents"-Direktive): der Sweep deckte tensor/backend/inference/tokenizer,
aber NICHT die Datei-I/O. safetensors `Load` + `Save` und internal/npy Read+Write
dekodierten JEDES F32/F64/F16/BF16-Skalar einzeln
(`math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))` / PutUintN je Element) —
ein ≈52M-Iter-Loop für ein 200MB-Modell. **Schlüssel-Einsicht: auf Little-Endian-Hosts
(ALLE GoAI-Targets) SIND die On-Disk-LE-Bytes bereits das In-Memory-Layout eines
verbatim-bit Tensors** → ein einziges `copy` statt Per-Element-Decode. Fix:
`nativeLittleEndian`-Guard + `rawCopyLE[T](dst,src,elemSize)` (read) / `rawStoreLE[T]`
(write) via `unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), n*elem)` + `copy`; Per-Element-
Loop bleibt als Big-Endian-Fallback → bit-identisch auf jedem Host. §V22 A/B: safetensors
load 2.09× (2165→4524 MB/s), npy read 4.9× / write 3.6×. NUR verbatim-bit dtypes
(F32/F64/F16/BF16); FP8/int/quant WIDEN echt (kein bulk-copy). DANN single-copy nachgelegt
(§T723): Load machte NOCH 2 Pässe (io.ReadAll→data + copy→dst) → STREAM (offset-tiling aus
dem HEADER allein validieren, dann jeden Tensor in Offset-Reihenfolge direkt aus r in seinen
Storage lesen; verbatim-LE per io.ReadFull in die Byte-View, single copy). §V22: 42.7→20.7ms
(2.06×, 4913→10153 MB/s). **KORREKTUR eines FRÜHEREN FALSCHEN ANNAHME:** ich hatte notiert
„safetensors ist ZERO-COPY, unschlagbar" — FALSCH. `safetensors.numpy.load_file` hat
OWNDATA=False ABER KOPIERT die Datei in EINEN owned Buffer (+ numpy-Views hinein) = one-copy,
10282 MB/s. GoAI 10153 ≈ 10282 = **PARITY** (beide one-copy, memcpy-bandbreiten-limitiert =
das Ceiling). LEHRE: MISS den Incumbent GENAU (OWNDATA/materialisierung prüfen), nimm nicht
„Rust/zero-copy = unschlagbar" an — die single-copy-Restrukturierung war NICHT „zu riskant":
die hostile-Tests asserten err-ONLY (nicht Message-exakt) → geänderte Error-Pfade sicher,
gated durch full-suite -race + FuzzLoad 5M + FuzzRoundTrip 6M execs. Erst Feasibility-Analyse
(was assertieren die hostile-Tests?), dann restrukturieren.

**WO GoAI die Incumbents SCHLÄGT (nicht nur Compute-Ceilings)** (2026-07-16): die
„beat all incumbents"-Kampagne fand die gewinnbaren Frontiers = wo GoAI STRUKTURVORTEIL
hat (kein Python/Cython/Interpreter-Overhead, kein BLAS/LAPACK/MPS-Ceiling): (1) classical
ML vs sklearn — schlägt/matcht ~alle (KMeans 29×, GBM 9.3×, kNN-fit 6.5×, RF 3.4×, NB 2.1×,
PCA 2×, GMM 1.6×, OLS 1.4×, tree-parity, SVC libsvm-floor, softmax-IRLS 2.5×; nur kNN-
predict/DBSCAN d=20 curse-of-dim trailen); (2) Tokenization vs tiktoken (Rust) — pure-Go
BPE **schlägt 1.15×** bei EXAKTER 216511-Token-Parität (23 vs 20 MB/s), guarded by
BenchmarkGPT2Encode; (3) Format-I/O — 2–5× self-win; safetensors load erreicht PARITY mit dem
Rust-Incumbent (10153 vs 10282 MB/s, beide one-copy) via single-copy streaming, npy 3–5×.
Compute (matmul/conv/GPU) bleibt am Silicon/BLAS/MPS-Ceiling. MERKE: MESSEN hat 2
überraschende Wins geliefert (tiktoken, format-I/O) — nicht annehmen „Rust/C schlägt Go",
sondern A/B messen; strukturvorteil-Domains (Interpreter-frei, nicht-BLAS) sind die
gewinnbaren. Siehe [[perf-gap-vs-python]] und [[integration-audit-method]] (class-audit
eines gefundenen Anti-Patterns über Geschwister: safetensors→npy war derselbe Fix).

**DTYPE-COVERAGE ist eine eigene Re-Sweep-Achse** (§T724, 2026-07-16): nach der Format-I/O-
Kampagne re-swept ich die SAFE-Zonen (`grep AtF64\|SetF64\|Unravel` über nlp/classic/autograd/
vision/rl) um Hot-Path-Erschöpfung KONKRET zu verifizieren (nicht anzunehmen). Fund: gpt2_hf.go's
tied-head-Transpose + c_attn-Split + bias-seg hatten F32/F64 TYPED (base-perf-sweep erledigt) ABER
F16/BF16 fielen in den per-Element-`default:`-Zweig — und F16/BF16-HF-Checkpoints sind der HÄUFIGE
Fall. Head-Transpose = vocab×d≈38M Elemente → ein F16-GPT-2 zahlte ≈327ms per-Element-Dispatch beim
LADEN (>16× die 20ms des safetensors-Reads). Fix: tensor.F16/BF16-cases mit U16-typed-Zugriff (ein
Transpose BEWEGT nur die 16-bit-Werte → bit-exakt), 326.8→21.6ms (15.1×). LEHRE: „schon optimiert"
gilt PRO DTYPE — ein typed F32/F64-Fastpath heißt NICHT dass F16/BF16 auch typed ist. Re-sweep die
BEREITS-optimierten Pfade auf die DTYPE-Achse (F16/BF16 verbatim-U16-move; nur wo output==input-dtype,
NICHT bei widening wie llama_gguf's F64-out-transpose = §C3-skip da F16-GGUF selten). Class-audit
bestätigte gpt2_hf als EINZIGEN high-value F16-gap (Geschwister: kleine per-forward index-Tensoren
[seq] = compute-dominiert §C3-skip; llama_gguf F16 selten). Siehe [[t650-topic-discovery-round]].

**PERF-KAMPAGNE „beat all incumbents" ABGESCHLOSSEN → PIVOT zu Feature-Discovery** (2026-07-16):
nach T718-724 (classical-ML schlägt sklearn, tokenization schlägt tiktoken, format-I/O parity/beats
safetensors, F16-load 15×) ist die SICHERE, in-zone, non-collision, non-ceiling Perf-Front KONKRET
erschöpft (re-swept + class-audited über ALLE Achsen: hot-path per-element, format-I/O, F16-load).
Rest-Perf = collision (backend/cpu pool-sync R246 = Worker aktiv im Paket), HW-ceiling (BLAS/LAPACK/
MPS), ∨ §C3-excluded (klein/selten). Per loop-keep-alive (never idle, advance) → PIVOT zurück zum
STANDING feature-discovery-loop: format/pytorch SAFE .pt/.bin-Loader (T725, restricted unpickler,
KEIN Code-Exec, whitelist-only GLOBAL/REDUCE — GoAIs Safety-Ethos vs torch.load-RCE; golden+reject+
fuzz-getestet; gibt map[string]*tensor.Tensor = direkt in nlp.GPT2FromHF). MERKE: wenn ein SPEZIFISCHES
Direktiv (hier perf) erschöpft ist, ist die richtige autonome Fortsetzung das STANDING loop-default
(project-advancement/feature-discovery auf ungefegten Paketen: format war eins — .pt-Gap gefüllt).

**How to apply:** ein Basis-Perf-Audit ist NICHT fertig nach den Compute-Kernels.
Sweep ALLE Ebenen mit `grep -rnE 'Unravel|AtF64\(|SetF64\(|f64at|f64set|get func'`
über tensor/ + backend/cpu/ + den Inferenz-Hot-Path (decode/sample/kv) + den
Tokenizer (nlp/bpe*, encode) + die FORMAT/SERIALISIERUNG (format/ + internal/npy,
grep zusätzlich `PutUint|Float3?2?bits|LittleEndian.Uint` in per-Element-Loops →
LE-bulk-copy-Kandidat). Jeder Treffer in einem per-ELEMENT-Loop (nicht
Fallback/Einmal-Setup) ist ein Kandidat; bei byte-gekeyten Maps zusätzlich auf
`string(a)+string(b)`-Keys achten. Miss immer (§V22 same-machine A/B) + verifiziere
Parität (full-suite, hoher Blast-Radius). Siehe [[perf-gap-vs-python]].

## T843-KORREKTUR (2026-07-18): dispatch ist NICHT automatisch der dominante term

Bei TRANSPOSE-artigen zugriffsmustern trifft die per-element-AtF64-heuristik daneben.
Gemessen an llamagpu/flat2DT (tied lm_head, 524M elemente):

- eine dispatch-FREIE pure-Go-kontrolle (kein tensor-layer) lief 2.20s vs 2.90s des
  alten AtF64-loops → **AtF64-dispatch war nur ~24% der kosten**
- dominant war der WRITE-STRIDE: 1 MB pro schritt, ein TLB-eintrag PRO element
- nur-dispatch-entfernen brachte enttaeuschende **1.2×**; erst 2D-cache-TILING (256×64)
  brachte den rest → gesamt **5.9×** (230ms→39ms, 1.1→6.8 GB/s)

REGEL: bevor "AtF64 weg" als fix angenommen wird, IMMER eine dispatch-freie kontrolle
messen. Ist die kontrolle kaum schneller als der loop, liegt das problem im
zugriffsmuster (stride/cache/TLB), nicht im dispatch — dann ist tiling/blocking der
hebel, und ein naiver devirtualisierungs-fix waere wertlose churn gewesen.

## FLOOR-KORREKTUR (2026-07-20): "floor erreicht" war auf den TRAINING-STEP-COMPUTE-pfad SKOPIERT

Der R246-"Go-Floor" (oben, Zeile 96) galt für den STEADY-STATE-trainings-schritt (optimizer-
arithmetik + VJPs + compute-kernel). Ein decode-CPU-profil zeigte dann `fillUniform`/`Zeros`
bei **~11% der samples — alles MODELL-SETUP**, nicht die decode-schleife. Ein profiler-getriebener
+ gerichteter sweep (4 subagenten, je measurement-gated, bit-identisch) fand 2–65× in DREI
angrenzenden klassen die R246 NIE fegte:

1. **SETUP/INIT/CONSTRUCTION** (läuft bei modell-bau, nicht pro step): fillGen (Zeros **5.6×**,
   T870), readGen (read-side-mirror), embedding-init, weight-averaging SWA/EMA/materialize
   2.3–2.6×, grad-accumulator 2.3×, Orthogonal-fill 2.63×.
2. **QUANTISIERUNG** (läuft über volle weight-matrizen bei load/quantize): LSQ **14.4×**,
   quant_linear.asF32 **10.7×**, f32Clone **40×**, BitNet 2.7×/2.1×.
3. **ÜBERSEHENE HOT-loops**: SAM.Step (optimizer-HOLDOUT, R246 sagte "steps devirt'd" — SAM war
   der einzige rest, **8.8×**), vjp_reshape (backward-pfad, R246 sagte "VJPs gefegt" — reshape-VJP
   fehlte, **17×**), mixup.PermuteRows (index-slice-alloc-pro-element, **65×** + 256005→5 allocs).

**META-LEHRE:** ein "floor erreicht"-claim MUSS seinen SKOPUS nennen. R246 war der training-STEP-
compute-pfad; er hat setup/init/quant/assembly NICHT gefegt und hatte punkt-holdouts (SAM,
reshape-VJP, mixup). Re-sweep auf der SKOPUS-achse: "was läuft das NICHT der steady-state-
step ist?" (construction, quantization, weight-averaging, checkpointing) ist eine EIGENE perf-
klasse. Der profiler zeigte direkt drauf (11% = setup) — MISS den echten workload, glaube keinem
skopierten floor-claim. §C3 hielt sauber: TruncNormal konvertiert→revertiert bei 1.086×, gumbel
1.05× + remoe-negligible gelassen. Delegation: 4 disjunkte file-sets, sonnet-5 für das mechanische
init-set, opus für die urteils-schweren (fill-vs-transform-vs-route-to-op). Siehe [[goai-autonomous-loop]].

## OPTIMIZER-+AMP-+GGUF-KONTINUATION (2026-07-20, "never throttle, continue grinding", T910-T912+T907)

Der T870-fund "SAM war der einzige optimizer-holdout" war (wie jeder "sweep complete") wieder UNVOLLSTÄNDIG. Ein `awk`-scan (func mit Unravel+AtF64 aber OHNE flatF64) fand SECHS weitere element-wise optimizer-Step-holdouts: **Sophia 3.5×, ScheduleFree 6.3×, CautiousAdamW 4.5×, Grokfast 5.1×, GrokfastMA 4.7× (gr 2× gelesen → in flat gepuffert), Lookahead 7.5×** — alle via den etablierten flatF64/flatF32-fastpath, bit-identisch, ALLE zu TestOptimizerFastPathParity hinzugefügt. §C3-GEMESSEN-UND-GELASSEN: GaLore/APOLLO/QGaLore (per-element loop NUR im 1-D-fallback; 2-D-weights gehen durch SVD-projektion — GaLore Step 1.6s SVD-dominiert, Gap=200) + Muon (Newton-Schulz-matmul-dominiert). **Element-wise optimizer-fastpath-sweep ist jetzt WIRKLICH komplett.**

**NEUE, SCHLIMMERE anti-pattern-klasse als AtF64-dispatch: per-element tensor-ALLOCATION.** nn/amp.go MixedPrecision.Sync rief `roundHalf` PRO ELEMENT — und roundHalf `tensor.FromFloat64([v]).Cast(dt)` ALLOZIERT einen 1-element-tensor + cast pro call → ein 512×512-weight = **2,88 MILLIONEN allocs pro Sync (56ms)**. Fix: das GANZE master in ZWEI bulk-casts runden (`m.Cast(halfDtype).Cast(w.Dtype)` = identisches element-wise rounding = bit-identisch) + contiguous copy → **50×** (2,88M→10 allocs). UnscaleGrads (per-step grad-unscale) flatF32 9.9×. **GREP-ZUSATZ für den sweep: `tensor.New|FromFloat64|scalarTensor|roundHalf|\.Cast\(` INNERHALB eines `for.*Numel()`-loops** — eine alloc/cast pro element ist weit schlimmer als AtF64 und versteckt sich hinter einem helper. Ein präziser scan fand danach NUR roundHalf (rest = per-param/per-row = ok).

**FORMAT-DECODE-straggler:** der T720/T721-rawCopyLE-fix war auf safetensors+npy+pytorch, aber **format/gguf decodeTensor lief NOCH per-element** (`math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))`): F32 2890→9757 MB/s **3.4×** (jetzt ÜBER safetensors' 8.4, ~1.3× hinter gguf-py 12.2), F16-read 1.53× (bulk-memmove in aligned []uint16 dann table-convert; f16→f32 selbst = scattered 256KB-table, read war der rest). rawCopyLE lokal in gguf dupliziert (5 zeilen, zero-dep). **LEHRE: class-audit einen fix über ALLE geschwister — gguf war das eine format-reader das rawCopyLE verpasst hatte** (T907 war als "worker-zone" gebucht, aber kein worker-branch berührt format/gguf → note war stale). Alle 4 format-reader bulk-decoden jetzt F32. Verbleibende gguf-perf = SIMD-dequant (Q4_K 745 MB/s, Q6_K 818 — schon storage-slice-direkt, weiterer win braucht NEON-asm = dedizierte fire, nicht marathon-tail). Siehe [[perf-gap-vs-python]].
