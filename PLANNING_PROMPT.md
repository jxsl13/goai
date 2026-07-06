# GoAI — Master-Planungs- und Implementierungs-Prompt

> Zweck dieser Datei: Ein wiederverwendbares Prompt-Set, um mit den Skills
> `/deep-research`, `/research`, `/spec`, `/review`, `/build` und `/loop` eine
> vollwertige Go-AI-Bibliothek von der Architektur bis zur optimierten
> Implementierung zu planen, zu bauen und zu verifizieren.
>
> Nutzung: Jeder Abschnitt („PHASE …") ist ein eigenständiger Prompt-Block.
> Kopiere den jeweiligen Block hinter den passenden Slash-Command.
> Reihenfolge: PHASE 0 → 1 → 2 → 3 → dann Dauerbetrieb mit `/loop`.

---

## 0. Nordstern (gilt für ALLE Phasen — immer mitgeben)

**Mission:** Eine idiomatische, modulare Go-Bibliothek für das gesamte KI-Spektrum
(lineare Algebra, Autograd, klassisches ML, Deep Learning, NLP/LLM-Inferenz, CV,
RL, probabilistische Modelle), deren Kernoperationen über SIMD-Assembler und
GPU-/NPU-Beschleuniger so nah wie möglich an äquivalente C/C++-Implementierungen
(PyTorch/ATen, llama.cpp, ONNX Runtime, Eigen, oneDNN) herankommen.

**Oberste Prinzipien (nicht verhandelbar):**
1. **Korrektheit vor Geschwindigkeit.** Jede Funktion existiert zuerst als
   referenz-valide, gut getestete Pure-Go-Implementierung. Optimierung ist ein
   *zweiter, separater* Schritt gegen genau diese Referenz.
2. **Mathematische/wissenschaftliche Fundierung.** Jede Einheit zitiert die
   zugrundeliegende Definition (Paper, Lehrbuch, kanonische Referenz-Impl.).
   Numerik-Entscheidungen (Stabilität, Genauigkeit, Overflow) werden explizit
   dokumentiert, nicht implizit getroffen.
3. **Numerische Parität als Abnahmekriterium.** „Fertig" heißt: Ergebnisse
   stimmen innerhalb definierter Toleranzen (ULP/rtol/atol) mit einer
   Referenz-Implementierung (NumPy/PyTorch/Referenz-C) überein — nachgewiesen,
   nicht behauptet.
4. **Performance ist messbar oder existiert nicht.** Kein „schneller" ohne
   Benchmark, Roofline-Einordnung und Vergleich gegen ein C/C++-Baseline.
5. **Plattform- und Hardware-Portabilität von Anfang an.** Native Unterstützung
   für **macOS, Windows, Linux** auf **CPU, GPU und NPU**. Jede beschleunigte
   Operation hat einen Pure-Go-Fallback, der überall läuft.
6. **Eine Verantwortung pro Modul, tiefe Module mit schmaler Schnittstelle.**
   Backend-Details (CUDA/Metal/Vulkan/SIMD) leben hinter stabilen Interfaces.

**cgo-Policy (harter Gate — gilt in jeder Phase und jeder Task):**
- **Standard ist Pure Go.** Jede Op wird zuerst in reinem Go implementiert und
  dann so weit wie möglich in reinem Go optimiert: Algorithmus, Cache-Blocking/
  Layout, `math/bits`, experimentelles **`simd`-Paket** (GOEXPERIMENT, sofern für
  die Ziel-Go-Version verfügbar — Status in §R verifizieren), `avo`-generierter
  x86-Asm, Plan9-Asm (NEON/ARM64), Goroutine-Parallelisierung.
- **cgo ist die Ausnahme, nicht der Default.** cgo (bzw. externe C/C++-Libs wie
  OpenBLAS/cuBLAS/cuDNN/oneDNN) darf erst eingeführt werden, wenn ALLE folgenden
  Bedingungen erfüllt sind:
  1. Die Pure-Go-Version existiert, ist §V-grün und wurde nachweislich bis an
     ihre praktische Decke optimiert (dokumentierte Stufen + Roofline-Einordnung).
  2. Ein Benchmark zeigt eine **deutliche** Performance-Überlegenheit der
     cgo-Variante gegen genau diese ausoptimierte Pure-Go-Version (Schwelle in
     §C festlegen, z. B. „≥ X× oder ≥ Y % näher an C++-Baseline"; „deutlich" wird
     als Zahl definiert, nicht nach Gefühl).
  3. Die cgo-Variante liegt hinter Build-Tags als OPTIONALES Backend; der
     Pure-Go-Pfad bleibt vollständig funktionsfähig und ist der Cross-Compile-
     Default (Fallback überall).
  Ist eine der drei Bedingungen nicht erfüllt ⇒ cgo wird abgelehnt, die Pure-Go-
  Version bleibt Auslieferungsstand, und die cgo-Idee wird als Kandidat in §B/§T
  geparkt statt gemerged.
- Pure-Go vs. generierter Assembler (`avo`) vs. Plan9-Asm — pro Backend abwägen
  (Portabilität, Build-Komplexität, Cross-Compilation, Performance-Decke). All
  diese Wege gelten als „Pure Go" im Sinne der Policy (kein C-Toolchain-Zwang).
- Lizenzmodell und Abhängigkeitspolitik (welche cgo-/Vendor-Libs sind erlaubt?).
- Speicher-/Tensor-Layout (row-major, Strides, Views, Alignment, Arena-Alloc,
  Zero-Copy zu GPU).
- Zahlentypen: f64/f32/f16/bf16/int8-Quantisierung — welche zuerst, wie getestet.
- Threading-Modell (Goroutines vs. fester Worker-Pool, NUMA, GOMAXPROCS-Wechselwirkung).

---

## PHASE 0 — Landschaft & Machbarkeit  →  `/deep-research`

**Slash:** `/deep-research`

**Prompt:**
> Erstelle einen tief recherchierten, quellenbelegten Report als Grundlage für
> den Bau einer vollwertigen KI-Bibliothek in Go, die via SIMD-Assembler und
> GPU/NPU-Beschleunigern an C/C++-Performance heranreichen soll. Beantworte
> faktenbasiert (mit Quellen, Datum, Versionsangaben):
>
> 1. **Stand der Technik in Go-ML:** Was existiert bereits (Gorgonia, gonum,
>    GoMLX, go-torch/cgo-Bindings, tract, candle-Vergleich)? Wo genau scheitern
>    sie an Performance/Portabilität/Wartung? Welche Lücke rechtfertigt Neubau?
> 2. **CPU-SIMD in reinem Go:** Realistisch erreichbare Performance OHNE cgo mit
>    (a) dem experimentellen `simd`-Paket der Go-Stdlib (GOEXPERIMENT — aktuellen
>    Verfügbarkeits-/Reifestatus und Ziel-Go-Version klären), (b) `avo`-generiertem
>    x86-Asm (AVX2/AVX-512), (c) ARM NEON/SVE (Plan9-Asm, Go-Support-Status),
>    (d) Auto-Vektorisierung des Go-Compilers, (e) `math/bits`. Benchmarks/Belege
>    für erreichbaren Anteil an Eigen/oneDNN/OpenBLAS — und ab wo eine Pure-Go-
>    Decke realistisch liegt, jenseits derer nur noch cgo weiterhilft.
> 3. **GPU-Wege aus Go:** cgo→CUDA/cuBLAS/cuDNN; Metal (macOS) via cgo/Objective-C;
>    Vulkan-Compute; WebGPU/wgpu; ROCm/HIP; SYCL. Trade-offs Build-Komplexität vs.
>    Portabilität vs. Peak-Performance. Zero-Copy-Optionen.
> 4. **NPU/Accelerator-Wege:** Apple Neural Engine via CoreML, Windows DirectML,
>    Intel oneDNN/NPU, Qualcomm/ARM. Was ist aus Go heraus überhaupt ansprechbar?
> 5. **Referenz-Baselines für Parität:** Welche C/C++-Bibliotheken dienen als
>    Korrektheits- und Performance-Referenz je Domäne (Eigen, OpenBLAS, oneDNN,
>    llama.cpp/ggml, ONNX Runtime, PyTorch/ATen)? Wie extrahiert man reproduzierbar
>    Golden-Referenzwerte?
> 6. **Modell-Interop-Formate:** ONNX, GGUF, safetensors, HuggingFace — Reifegrad,
>    Aufwand.
> 7. **Verifikations-Methodik im Feld:** Numerische Gradientenprüfung,
>    Property-Based-Testing, ULP-Toleranzen, differential testing gegen Referenz,
>    fuzzing für Numerik.
>
> Liefere am Ende: (a) Empfehlung für die 3–4 tragenden Architektur-Wetten,
> (b) die größten technischen Risiken mit Gegenmaßnahmen, (c) eine begründete
> Reihenfolge, in welcher Domänen/Backends gebaut werden sollten.

**Output-Erwartung:** Ein Report (wird zur Rohquelle für §R im Spec).

---

## PHASE 1 — Gezielte Detail-Recherche  →  `/research`

**Slash:** `/research`  (mehrfach, je offener Entscheidung eine Runde)

Für jede noch offene Kern-Entscheidung aus PHASE 0 eine fokussierte Runde. Beispiele:

> `/research` Vergleiche `avo` (generierter x86-Asm) gegen handgeschriebenen
> Plan9-Assembler gegen cgo→OpenBLAS für GEMM in Go: Wartbarkeit,
> Cross-Compile-Fähigkeit, gemessene GFLOP/s-Decke, Alignment-Anforderungen.
> Empfiehl eine Default-Strategie und benenne, wann man abweicht. Trag die
> Befunde als §R ins Spec.

> `/research` Kläre den aktuellen Reifegrad des experimentellen `simd`-Pakets der
> Go-Stdlib (GOEXPERIMENT): Verfügbarkeit je Go-Version, unterstützte Architekturen
> (x86 AVX2/AVX-512, ARM64 NEON), API-Stabilität, gemessene Performance vs.
> `avo`/Plan9-Asm, Cross-Compile-Verhalten. Definiere daraus die konkrete
> „deutlich"-Schwelle für den cgo-Gate (Speedup-Faktor / % der C++-Baseline).
> Trag Befund + Schwelle als §R/§C ein.

> `/research` Ermittle den kanonischen, numerisch stabilen Algorithmus + die
> übliche Referenz-Toleranz für: Softmax, LayerNorm, GELU, Adam, Conv2d (im2col
> vs. Winograd vs. direct), Attention (naiv vs. FlashAttention). Jeweils mit
> Quelle. Trag als §R ein.

**Regel:** Jede Behauptung mit Quelle; Unbelegtes wird als `?` markiert, nie als
Fakt geschrieben. Ergebnisse landen im **§R (Research-Log)** des Specs.

**Research-Mechanik (Pflicht):** NICHT den eingebauten `/deep-research`-Workflow
verwenden (erzwingt StructuredOutput-Schema → crasht unter Rate-Limits). Immer
`research-lite` (`.claude/workflows/research-lite.js`): eine fokussierte Frage
pro Lauf, schema-frei, komprimierende Sub-Agenten, graceful bei toten Agenten.
Details siehe `LOOP.md` → „Research-Regel".

---

## PHASE 2 — Spec & Architektur  →  `/spec`

**Slash:** `/spec`

**Prompt:**
> Erzeuge `SPEC.md` für die Go-AI-Bibliothek „GoAI" auf Basis des Nordsterns
> (oben) und der §R-Befunde. Struktur:
>
> - **§G (Goals):** Der Nordstern, verdichtet. Messbare Definition von „vollwertig".
> - **§C (Constraints):** Aufgelöste Entscheidungen aus PHASE 0/1 (Backend-Strategie
>   pro Plattform, Zahlentypen-Roadmap, Tensor-Layout, Threading, Lizenz/Deps).
>   MUSS die **cgo-Policy** als messbaren Gate fixieren: Pure Go ist Default;
>   cgo nur nach ausoptimierter Pure-Go-Version + benchmarkbelegter, als konkrete
>   Zahl definierter „deutlicher" Überlegenheit; cgo immer optional hinter
>   Build-Tags mit Pure-Go-Fallback. Lege die „deutlich"-Schwelle hier numerisch
>   fest (z. B. Speedup-Faktor und/oder % der C++-Baseline).
> - **§I (Invariants der Architektur):** Das Schichtenmodell und seine harten
>   Grenzen — z. B.:
>   - `L0 core`: Tensor, Dtype, Device, Speicher/Allocator, Strides/Views.
>   - `L1 compute`: Backend-Interface (`Backend`, `Kernel`) + Pure-Go-Referenz-Backend,
>     das ÜBERALL läuft und Definition der Wahrheit ist.
>   - `L1b accel`: austauschbare Backends (cpu-simd, cuda, metal, vulkan, npu),
>     alle gegen dasselbe Interface, mit Feature-Detection + Fallback.
>   - `L2 autograd`: Tape/Graph, VJP-Regeln je Op.
>   - `L3 nn`: Layers, Init, Optimizer, Loss, Datenpipeline.
>   - `L4 domains`: classic-ML, vision, nlp/llm-inference, rl, probabilistic.
>   - `L5 io`: ONNX/GGUF/safetensors, Serialisierung, Model-Zoo.
>   INVARIANT: Höhere Schichten kennen keine Backend-Interna. Jede Op hat einen
>   Pure-Go-Fallback. Kein `cgo` in `L0`. Public API ist backend-agnostisch.
> - **§V (Verifikations-Invarianten — die Abnahme-Regeln):** z. B.
>   - V-PARITY: Jede Op besteht einen Golden-Test gegen die benannte Referenz
>     innerhalb der in §R fixierten Toleranz (rtol/atol/ULP).
>   - V-GRAD: Jede differenzierbare Op besteht numerische Gradientenprüfung
>     (finite differences) unter definierter Schwelle.
>   - V-CROSS: Backend-X-Ergebnis == Pure-Go-Referenz innerhalb Backend-Toleranz.
>   - V-PLATFORM: CI grün auf {macOS, Windows, Linux} × {CPU-Fallback + verfügbarer
>     Accel}. Fehlender Accel ⇒ Skip mit Log, nie stiller Pass.
>   - V-BENCH: Jede optimierte Op hat einen Benchmark + Baseline-Vergleichszahl;
>     Regressionen brechen CI.
>   - V-PROP: Property-Based-Tests für Invarianten (Shape, Linearität, Assoziativität
>     wo mathematisch garantiert).
>   - V-CGO: Kein cgo im Auslieferungspfad ohne (a) grüne, ausoptimierte Pure-Go-
>     Referenz und (b) eingecheckten Benchmark, der die §C-Schwelle überschreitet.
>     Der Pure-Go-Build (ohne C-Toolchain) muss auf allen Plattformen grün bleiben.
>   - V-STABLE: Public API ändert sich nur über dokumentierten Deprecation-Pfad.
> - **§T (Task-Backlog):** Die eigentliche Arbeitsliste. Jede Task ist EIN
>   auslieferbares, testbares Inkrement, geordnet nach Abhängigkeit. Jede Task
>   trägt: Ziel, betroffene Schicht, Referenz für Parität, Definition-of-Done
>   (welche §V-Regeln sie erfüllen muss). Reihenfolge-Leitlinie:
>   1. `L0` Tensor/Dtype/Device + Allocator + Pure-Go-Referenz-Backend.
>   2. `L1` GEMM/elementwise/reduce als Referenz + Golden-Tests + Bench-Harness.
>   3. **Erst dann** erste Optimierung (SIMD-GEMM) — als separate Task gegen die
>      Referenz aus Schritt 2.
>   4. Autograd-Kern + VJP-Regeln der L1-Ops.
>   5. NN-Basics (Linear, Activation, Loss, SGD/Adam) end-to-end auf CPU.
>   6. Erst danach GPU-Backend, dann Transformer/LLM-Inferenz, dann weitere Domänen.
> - **§B (Backprop-Log):** anfangs leer; Bugs/Fehlschläge werden hier zu neuen
>   §V-Invarianten verdichtet.
>
> Halte §T bewusst in kleinen Schritten: „Korrektheit zuerst, Optimierung als
> eigene Folge-Task" muss sich in der Task-Struktur physisch widerspiegeln.

---

## PHASE 3 — Adversariales Review des Specs  →  `/review`

**Slash:** `/review`

**Prompt:**
> Red-Team das `SPEC.md`, bevor Code entsteht. Prüfe insbesondere:
> - Sind die §V-Regeln WIRKLICH ausreichend, um C++-Parität nachzuweisen, oder
>   erlauben sie stillen Genauigkeitsverlust?
> - Ist die Backend-Abstraktion (§I) tragfähig für CUDA UND Metal UND Vulkan UND
>   NPU, oder leakt ein Backend zwangsläufig in die API?
> - Ist die Task-Reihenfolge frei von versteckten Zyklen (Autograd vs. Kernel vs.
>   Device-Placement)?
> - Wo verleitet der Plan dazu, zu früh zu optimieren?
> - Welche numerischen Fallen (f16-Overflow, Reduktions-Reihenfolge,
>   nicht-assoziative FP-Summation, Determinismus über Backends) fehlen in §V?
> Jede Beanstandung mit Beleg (Datei:Zeile im Spec oder §R-Quelle). Überlebende
> Findings härten §V. Ende mit explizitem Go/No-Go.

Nach Go: SPEC.md ist eingefroren als Wahrheit. Erst jetzt beginnt der Bau.

---

## DAUERBETRIEB — kontinuierliche Implementierung  →  `/loop` + `/build`

Der Bau läuft über den `/build`-Skill (plan-then-execute gegen SPEC.md, mit
automatischem `backprop` bei Test-/Build-Fehlern). `/loop` treibt das
selbstgetaktet Task für Task.

### /loop-Definition (selbstgetaktet, ohne festes Intervall)

**Slash:** `/loop`  (kein Intervall ⇒ das Modell taktet sich selbst pro Task)

**Prompt (genau so übergeben):**
> Arbeite VOLL AUTONOM, ohne Rückfragen an den Nutzer. Implementiere GoAI
> kontinuierlich streng nach `SPEC.md`. Führe pro Iteration GENAU EINE Aufgabe
> zu Ende und stoppe die Schleife erst, wenn alle §T-Tasks den Status „done"
> tragen. Ablauf je Iteration:
>
> 0. **Bootstrap (falls `SPEC.md` fehlt oder unvollständig ist):** Erzeuge zuerst
>    autonom die Planungsgrundlage, EINE Phase pro Iteration, in dieser Reihenfolge:
>    (a) `/deep-research` → Report nach `docs/research/00-landscape.md`; (b)
>    `/research` für die offenen Kern-Entscheidungen → §R/§C; (c) `/spec` →
>    `SPEC.md` mit §G §C §I §V §T §B; (d) `/review` → Findings härten §V, Ergebnis
>    als Go/No-Go-Notiz. Erst wenn `SPEC.md` existiert und ein Review-„Go" trägt,
>    gehe zu Schritt 1 über. Nutze `PLANNING_PROMPT.md` als Vorlage der Phasen-Prompts.
>
> 1. **Auswahl:** Wähle die nächste nicht-erledigte §T-Task, deren Abhängigkeiten
>    erfüllt sind. Nenne ihre ID und Definition-of-Done, bevor du beginnst.
> 2. **Bauen:** `/build` diese Task. Falls es eine Optimierungs-Task ist, MUSS die
>    referenz-valide Pure-Go-Version bereits existieren und grün sein — sonst
>    zuerst deren Korrektheits-Task bauen.
> 3. **Verifizieren (Abnahme nach §V):**
>    - V-PARITY: Golden-Test gegen die in der Task benannte Referenz innerhalb
>      der §R-Toleranz. Fehlt eine Golden-Datei, erzeuge/aktualisiere sie
>      reproduzierbar aus der Referenz und committe sie.
>    - V-GRAD (falls differenzierbar): numerische Gradientenprüfung.
>    - V-CROSS (falls Backend-Task): Ergebnis == Pure-Go-Referenz.
>    - V-PROP: einschlägige Property-Tests.
> 4. **Messen + cgo-Gate (nur bei Optimierungs-Tasks):** Optimiere zuerst in
>    reinem Go bis an die Decke (Algorithmus → Layout/Blocking → `simd`/`avo`/
>    NEON → Goroutines), jede Stufe mit grüner §V und dokumentiertem Benchmark-
>    Delta (GFLOP/s, Speedup, % der C++-Baseline, Roofline). Erst wenn die Pure-
>    Go-Decke erreicht ist, prüfe einen cgo-Kandidaten NUR falls die §C-Schwelle
>    plausibel erreichbar scheint: baue ihn als optionales Build-Tag-Backend,
>    benchmarke gegen die ausoptimierte Pure-Go-Version. Überschreitet er die
>    §C-Schwelle ⇒ V-CGO erfüllt, mergen (Pure-Go bleibt Default-Fallback).
>    Sonst ⇒ verwerfen, Pure-Go bleibt Auslieferungsstand, cgo-Idee als Notiz in
>    §B parken. Keine Optimierung gilt als fertig ohne die Benchmark-Zahl.
> 5. **Fehlerbehandlung:** Bei Test-/Build-/Paritäts-/Regressions-Fehler `backprop`
>    aufrufen: Ursache tracen, prüfen ob ein NEUER §V-Invariant den Rückfall
>    künftig fängt, §B ergänzen, dann Fix. Niemals einen roten Test überspringen
>    oder Toleranzen aufweichen, um grün zu werden — Toleranz-Änderungen brauchen
>    eine §R-Begründung.
> 6. **Plattform-Check:** Stelle sicher, dass die Pure-Go-Pfade plattformneutral
>    bleiben und beschleunigte Pfade sauber hinter Build-Tags/Feature-Detection
>    liegen (macOS/Windows/Linux × CPU/GPU/NPU), mit Fallback. Nicht verfügbare
>    Accel-Backends ⇒ dokumentierter Skip, kein stiller Pass.
> 7. **Abschluss:** Task in §T auf „done" setzen, kurzes Changelog (was, Parität,
>    Benchmark), und — nur wenn der Nutzer Commits erlaubt hat — committen. Dann
>    zur nächsten Iteration.
>
> Invarianten der Schleife: Immer nur eine Task offen. Korrektheit vor
> Performance. Kein Fortschritt ohne bestandene §V-Abnahme. Kein „optimiert" ohne
> Benchmark-Zahl.
>
> **Autonomie-Regel (statt Nutzer-Stopp):** Bei struktureller Unklarheit oder
> einer offenen Design-Entscheidung NICHT stoppen und NICHT den Nutzer fragen.
> Stattdessen: (a) die wissenschaftlich/mathematisch am besten begründete Default-
> Option wählen (mit Quelle, wenn möglich); (b) die Entscheidung als Eintrag in
> `docs/decisions/ADR-<n>.md` festhalten (Kontext, Optionen, Wahl, Begründung,
> Revidierbarkeit) UND als `?`-Merker bzw. Amendment in `SPEC.md` (§C/§B)
> vermerken; (c) mit der so getroffenen Annahme weiterbauen. Nur ECHTE harte
> Blocker (fehlende Toolchain, kaputte Umgebung, die der Loop nicht selbst
> reparieren kann) rechtfertigen einen Halt — dann eine knappe
> `PushNotification` senden und die nächste Iteration abwarten. Niemals
> Toleranzen aufweichen oder Tests überspringen, um Fortschritt vorzutäuschen —
> ein roter Test ist ein `backprop`-Fall, kein Grund zum Lockern.
> Commits/Pushes bleiben untersagt, solange der Nutzer sie nicht ausdrücklich
> erlaubt; der Loop arbeitet bis dahin nur im Working Tree.

### Optional: intervallgetaktete Variante (für Langläufer/Übernacht)

> `/loop 30m` + denselben Prompt — nützlich, wenn einzelne Tasks lange Builds/
> Benchmarks nach sich ziehen und du regelmäßige Fortschritts-Checkpoints willst.
> Für reine Rechenlast ist die selbstgetaktete Variante (kein Intervall) meist
> besser, weil sie erst nach echtem Task-Abschluss weitertaktet.

---

## Querschnitt: Verifikations- & Performance-Standard (gilt in jeder Task)

**Korrektheits-Nachweis (bevor überhaupt optimiert wird):**
- Golden-Tests gegen benannte Referenz (NumPy/PyTorch/Referenz-C), Toleranzen
  aus §R, reproduzierbar erzeugt und eingecheckt.
- Numerische Gradientenprüfung für alle differenzierbaren Ops.
- Property-Based-Tests für mathematisch garantierte Eigenschaften.
- Cross-Backend-Differential-Testing gegen die Pure-Go-Referenz.

**Performance-Methodik (der zweite, separate Schritt):**
1. Roofline/Komplexität analysieren (compute- vs. memory-bound).
2. **In reinem Go** optimieren in Stufen: Algorithmus → Cache-Blocking/Layout →
   SIMD (experimentelles `simd`-Paket / `avo` / NEON) → Multithreading. Nach
   JEDER Stufe: Korrektheit erneut grün + Benchmark-Delta dokumentiert.
3. Vergleich gegen C/C++-Baseline in % der erreichten Peak-Performance.
4. **cgo-Gate** erst nach erreichter Pure-Go-Decke: cgo/externe-Lib nur mergen,
   wenn Benchmark die §C-Schwelle gegen die ausoptimierte Pure-Go-Version reißt;
   sonst verwerfen. GPU/NPU-Offload folgt derselben Logik (optionales Backend,
   Pure-Go-Fallback bleibt).
5. Benchmark-Regressionsschutz in CI (V-BENCH), Pure-Go-Build ohne C-Toolchain
   bleibt auf allen Plattformen grün (V-CGO).

**Plattform-/Hardware-Matrix (Zielabdeckung):**
- OS: macOS, Windows, Linux.
- CPU: x86-64 (AVX2/AVX-512) via `avo`; ARM64 (NEON, ggf. SVE) via Plan9-Asm;
  überall Pure-Go-Fallback.
- GPU: CUDA/cuBLAS/cuDNN (Linux/Windows), Metal (macOS), Vulkan-Compute
  (portabel), ROCm (optional).
- NPU: CoreML/ANE (macOS), DirectML (Windows), oneDNN (Intel) — soweit aus Go
  ansprechbar; sonst als Nicht-Ziel markiert (kein stilles Versprechen).

---

## Reihenfolge auf einen Blick

```
/deep-research   → Landschaft, Wetten, Risiken, Domänen-Reihenfolge   (→ Rohquelle)
/research (×N)   → offene Entscheidungen faktisch klären              (→ §R)
/spec            → SPEC.md: §G §C §I §V §T §B                          (→ Wahrheit)
/review          → Red-Team, §V härten, Go/No-Go                      (→ Freeze)
/loop + /build   → Task für Task bauen, verifizieren, messen          (→ Dauerbetrieb)
   └─ backprop bei jedem Fehlschlag → neue §V-Invarianten
```
