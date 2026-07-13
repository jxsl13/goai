# GoAI — Autonomer Loop-Prompt

> Maßgebliche Iterationsanweisung für den autonomen Build-Loop.
> Wird vom Cron-Job referenziert; Änderungen hier wirken ab dem nächsten Fire.
> Rahmen: `PLANNING_PROMPT.md` (Abschnitt DAUERBETRIEB), `SPEC.md` (Wahrheit).

VOLL AUTONOM, ohne Rückfragen an den Nutzer. Arbeite an der Go-AI-Bibliothek
GoAI unter `/Users/john/Desktop/goai`. Führe pro Fire GENAU EINE Aufgabe zu
Ende.

## Ablauf

0. **BOOTSTRAP** falls `SPEC.md` fehlt/unvollständig: erzeuge autonom die
   Planungsgrundlage, EINE Phase pro Fire — (a) `/deep-research` →
   `docs/research/00-landscape.md`; (b) `/research` offene Kern-Entscheidungen
   → §R/§C; (c) `/spec` → `SPEC.md` (§G §C §I §V §T §B); (d) `/review` → §V
   härten + Go/No-Go-Notiz. Erst mit `SPEC.md` + Review-„Go" zu Schritt 1.
1. **Auswahl:** Nächste abhängigkeitsfreie §T-Task wählen, ID +
   Definition-of-Done nennen.
2. **Bauen:** `/build` (bzw. direkt implementieren, plan-then-execute gegen
   SPEC.md).
3. **§V-Abnahme:** V-PARITY (Golden gegen Referenz-Toleranz, via
   `.venv`-Python/NumPy erzeugen falls nötig), V-GRAD, V-CROSS, V-PROP.
4. **Optimierungs-Task:** erst in REINEM Go bis zur Decke
   (Algorithmus→Layout→simd/avo/NEON→Goroutines, jede Stufe grün +
   Benchmark-Delta), dann cgo-Gate (V-CGO): cgo nur als optionales
   Build-Tag-Backend mergen, wenn Benchmark die §C-Schwelle gegen die
   ausoptimierte Pure-Go-Version reißt — sonst verwerfen, Idee in §B parken.
5. **Fehlschlag** → backprop: Ursache tracen, ggf. neuen §V-Invariant, §B
   ergänzen, Fix. Nie Toleranzen aufweichen oder Tests überspringen.
6. **Plattform-Check:** Pure-Go OHNE C-Toolchain grün — PFLICHT via
   `CGO_ENABLED=0 go vet ./...` oder `go test ./...` (§V23: `go build` allein
   kompiliert KEINE _test.go-Dateien und übersieht fehlende Build-Tags, §B45);
   Accel hinter Build-Tags mit Fallback. Voll-Sweeps (`go test ./...`) IMMER mit
   `-timeout 1800s` (llamagpu > 600s-Default, §B46) und Exit-Code UNGEPIPET
   prüfen — `| grep | tail` maskiert FAIL als Exit 0 (§V24).
7. **Abschluss:** Task in §T auf „done" (`x`), kurzer Changelog-Eintrag in
   `CHANGELOG.md`.

## Research-Regel (Pflicht — mitigiert den StructuredOutput-Fehler)

Für externe Recherche NIE den eingebauten `/deep-research`-Workflow verwenden:
er zwingt jeden Sub-Agenten zu einem `StructuredOutput`-Schema-Call, der unter
Rate-Limits 5× fehlschlägt und den ganzen Workflow crasht
(`StructuredOutput retry cap exceeded`).

Stattdessen IMMER den eigenen Workflow `research-lite`
(`.claude/workflows/research-lite.js`) nutzen:
- **kleiner Scope**: genau EINE fokussierte Frage pro Lauf (Kontext schonen);
- **schema-frei**: kein `agent({schema})` → der StructuredOutput-Fehler ist
  strukturell unmöglich;
- **komprimierende Sub-Agenten**: jeder Angle-Agent gibt ≤6 verdichtete Zeilen
  zurück, ein Synth-Agent verdichtet auf ≤8 Zeilen → sprengt den Kontext nicht;
- **graceful**: tote Agenten → `null`, gefiltert, nie Throw.

Aufruf: `Workflow({ scriptPath: ".claude/workflows/research-lite.js", args: "<eine präzise Frage>" })`.
**Validierungs-Ladder (§V16, Pflicht bei jeder Algorithmus-Implementierung):**
- Stufe 1 = bit-/tol-exakte Parität gegen die offizielle Referenz-Lib
  (torch/sklearn/gguf-py/safetensors). Notwendig, NICHT hinreichend.
- Stufe 2 (finale Autorität) = das **wissenschaftliche Paper**, das den
  Algorithmus definiert (arXiv/DOI/kanonisches Lehrbuch) — die implementierte
  Formel MUSS der Paper-Gleichung entsprechen, zitiert in §R.
- Dateiformate haben kein Paper → definitorische Quelle = Format-Spec/Ref-Impl
  (explizit so festhalten, kein Paper erfinden).
Dateiformate: immer Round-Trip + Fuzz (§V15).

## Autonomie-Regel

Bei Unklarheit/Design-Entscheidung NICHT stoppen, NICHT fragen —
wissenschaftlich begründete Default-Wahl treffen, in
`docs/decisions/ADR-<n>.md` + SPEC-Amendment (§C/§B) dokumentieren,
weiterbauen. Nur echte harte Blocker (kaputte Toolchain) → kurze
PushNotification, sonst weiter. KEINE Commits/Pushes ohne ausdrückliche
Nutzer-Erlaubnis. Schleife endet erst, wenn alle §T-Tasks „done".

## STATUS: Loop AKTIV — Stand §T561: Quellenlisten-Audit 12/12 aufgelöst (GSPO/QLoRA/SSD/i-Quants/MXFP4/Sparse-Trio/KDA); alles grün (2026-07-13)

Spec bei §T493. Die Session T364–T432 hat ADR-0019 (Batched Decode) komplett
durchgezogen UND die Attention/Training-Hebel geholt. Alle Gewinne §V22-gemessen,
jede Op gegen ref kreuzvalidiert, beide Backends ([[gpu-ops-all-backends]]).
T434–T444 hat danach das INTEGRATIONS-Programm geschlossen (siehe Ära-Bullet).
T472–T477 = TRAINED-MODEL-MESS-SERIE (docs/inference.md): Distill-your-drafts
73→88% Akzeptanz; Mirostat = Surprise-CEILING (einseitig); Watermark δ=2
detektierbar ab 50 Tokens; Bounded-Cache +0,05 Bits; Streaming 4× über Ctx
(RoPE, pre-RoPE-Cache verifiziert); Q8/Q4 quasi verlustfrei (99%/97% teacher-
forced Agreement — NIE free-running messen). Sink-Phänomen zeigt sich im
Toy-Maßstab NICHT (2× ehrlich negativ). trainCharLlama (cpu, 3s) entsperrt
Llama-Messungen generell.
T480–T492 = DECODE-ABSCHLUSS + FAMILIEN-E2E-SERIE (docs/training.md): jede
Decode-Strategie trained-model-verifiziert (CD/DoLa via Mechanismus-Surprise,
Beam +0,31 Nats über Greedy, DoLa-Plausibilität = log₂(1/α)-Garantie); Optimizer-
Zoo (8: SOAP führt 1,188; Shampoo-Bug gefixt → WithShampooRootEvery) + 5 Wrapper
(GaLore 1,380 schlägt AdamW; SAM-Zwei-Pass; Grokfast braucht LR-Kompensation);
NEFTune-Claim zeigt sich (Held-out ↓ bei Train ↑); DDPM/DDIM + Flow-Matching
rekonstruieren denselben Ring; EWC 50→89% Retention (Tasks müssen gemeinsam
repräsentierbar sein!); TIES/Soup schlagen Spezialisten-Worst-Case (Soup > TIES
bei Same-Base); VQ-VAE 94% Varianz durch den ST-Flaschenhals; SimSiam 94%
Linear-Probe labelfrei. Assert-Disziplin: Theorie-Garantien asserten, Rest
loggen; Regularisierer auf Held-out beurteilen; Agreement teacher-forced.
T493–T499 = ARCHITEKTUR-E2Es + RL/MERGING-ABSCHLUSS: Mamba-, RetNet-, MLA- und
Mamba+MoE-Char-LMs (jeweils strukturelle KAUSALITÄTS-Assertion + Training +
Generierung; MoE-Balance-Loss hält 4 Experten bei 20–28%); GAE+PPOClip zum
PPO-Actor-Critic verbunden (Return 1,0, Critic V(start)≈γ⁴); GreedySoup
verifiziert; DARE = dritter SKALENABHÄNGIGER Claim (0,9-Drop schadet klar,
0,5 = Rauschen — Redundanz-Prämisse braucht Großmodelle). RWKV-Block-Wrapper,
K-Quant-Qualität (braucht dim≥256-Llama), Sophia-GNB-Harnisch: demand-gated.
T504–T514 = BLOCKER-ABBAU (gepinnte Blocker einzeln geöffnet, jede Behauptung
re-verifiziert): MHA bekam DREI Nähte (LoRA §T504, Bias §T505, Mask §T508 —
nil = exakt alter Pfad, Golden-Suiten unberührt); GPT2FromHF (§T506, c_attn-
Split, tied head); OpMHAMasked (§T507) → Tree-Verify durchs volle Modell
(§T508, bit-identisch auf ref + T461-Kette durchs Modell verifiziert) →
MedusaGenerateTree (§T509, topK=1 ≡ Chain EXAKT) → Messung mit trainierten
Heads (§T510: Tree 4,00 vs Chain 3,92 tok/round; Akzeptanz-DECKE bei Toy-Scale).
WICHTIGE KORREKTUR §T508: Masken ≠ gemergte Score-Quellen — Self-Extend braucht
Letzteres → OpMHASelect (§T512, drei bit-exakte Kollaps-/Symmetrie-Tests) →
Self-Extend-E2E (§T513: Training NUR auf 32er-Fenstern, Eval bei 4×: plain CE
0,316→1,488 degradiert, Self-Extend w=8/G=8 hält 0,515 — OHNE Fine-Tuning).
Dazu §T511: persistenter Worker-Pool in cpu.parallelWork (Allocs −70–75%/Op,
Zeit ±0–2%, -race-verifiziert; A/B via Datei-Toggle, NIE git stash). Docs §T514.
T515–T521 = SWEEP + RWKV-FAMILIE + FUZZ-HÄRTUNG: Voll-Sweep grün (llamagpu
braucht jetzt -timeout 1800s, §B46/§V24: Exits UNGEPIPET prüfen!); OpWKV-Backward
GEBAUT (§T516: Softmax-Average-Identität, O(T²)-Reverse, Gradcheck über alle 4
Inputs — der "eigenes Projekt"-Blocker fiel in einem Fire) → nn.RWKVBlock +
Char-LM-E2E (§T517: CE 3,01→0,12, Kausalität bit-exakt; Architektur-Serie
KOMPLETT) → Rekurrenz-Inferenz (§T518: Step ≡ Forward ≤1e−12, O(1)-State).
Integrations-Audit R2 (§T519: Self-Extend-Positionsmathematik war vom Forward
entkoppelt → selfExtendPos + Spec-Konsistenztest). Fuzz-Sweep über ALLE 35
Targets (§T520/§B47): gguf-Bounds-Check-uint64-OVERFLOW (2 Stellen, Subtraktions-
form) + Tokenizer-JSON unbegrenzte-ID-Alloc (BPE per Fuzz, WordPiece per
Klassen-Audit) gefixt; explizite Hostile-Tests + 6×60s-Deep-Fuzz grün (§T521).
Offene ECHTE Blocker: nur noch amd64-Tasks (T11b/T74, host-blockiert) und echte
GPT-2-Gewichte (Download braucht Nutzer-Erlaubnis).
T523–T532 = WARTUNG + VULKAN-PERF-ARC: Race-Sweep baumweit 0 Races (§T523);
Bench-Regressions-Check keine Drift (§T524); Memory-Wartung (§T525); Fuzz-Programm
komplett — Q4_K-Schranke strukturell korrigiert, Encoder optimal (§T526/§B48);
Self-Extend-Extensionskurve: plain 0,91→2,40 monoton, SE flach 0,57→0,70 bis 8×
(§T527). VULKAN-PERF: mha_bwd Matmul-Dekomposition 71,5→4,74ms = 15× (§T528,
2 neue Shader softmax_packed/sm_jacobian, 7-Stage-Kette); §T529 Forward-Idee
zurückgezogen (T398-Akte) → §T531 mit NEUER Evidenz (Profil: fwd 19%; billigere
Struktur) Forward-Kette +18% = Default; kumuliert 935→1882 tok/s = 2,01×
(BenchmarkGPTTrainingStepVK neu, metal-Klasse); §T532 GEMM-Decke per §B39-Akte
bestätigt → Arc GESCHLOSSEN. Lektionen: Akte VOR dem Bauen lesen (T529 vs T531 =
Rückzug ohne / Bau mit neuer Evidenz); §V24 auch für Einzelsuiten.
T534–T538 = ADR-0008-ROUTING + B49-BOGEN: Binary-Elementwise zurück auf die
SIMD-CPU (metal +7,8%, vulkan +5,8% Training; cpuPrefers-Gate gegen ref-Lecks);
Unary/AddBias-Routing GEMESSEN VERLOREN (cpu-unOp = skalare Closures) → revertiert
(§T535). §B49: In-Kernel-Redispatch mit erhaltenem Recorder = Op doppelt aufs Tape
= GRADIENTEN VERDOPPELT — nur die scharfen Trained-Model-Schranken des Voll-Sweeps
sahen es (Shape/Parität/Gradcheck blind!). Fix → 46-Stellen-Klassen-Audit (§T537)
→ strukturell: Execute strippt Recorder vor Kernel-Aufruf, NEU §V25 (§T538).
Endstand: metal 3219 tok/s, vulkan 935→1992 = 2,13×. Zwei Regressionstests
(RecordsOnce binary + Fallback-under-tape ALiBi bit-identisch vs ref).
T540–T546 = STEADY-STATE + AT-SCALE-BOGEN: Demand-Gates evidenzbasiert
geschlossen (§T540 cpu-GELU 0,79ms > GPU 0,39ms); SelfExtendGenerate (§T541:
generierter Text hält 0,50 Surprise bei 4×, plain degeneriert 2,30); 124M-SKALA
mit synthetischen Gewichten (§T543–T546): GPT2FromHF-Mechanik + Batched-Decode
bei echter Größe verifiziert (batched ≡ analysis bit-gleich), Tabelle 2 Backends
× 3 Formate — metal-f32 76 / vulkan-Q4_K 72 tok/s, Backend-INVERSION bei
Quant-vs-f32, Q4_K > Q8 beidseitig. Harness nimmt echte Gewichte, sobald erlaubt.
T548–T560 = QUELLENLISTEN-AUDIT (Nutzer-Referenzliste → 12 Gaps → alle
aufgelöst): GSPO (Op+VJP, Kollaps auf GRPO(β=0) exakt, Trained-E2E 0,04→0,96);
QK-Clip (γ²-Gesetz); DeepSeekMoE (Shared Experts, bit-Kollaps); QLoRA-E2E
(NF4 verlustfrei, Adapter 1,19→1,08, Basis bit-gefroren); Mamba-2-SSD
(Dualitäts-Theorem ≤1e−12); i-QUANT-FAMILIE komplett (alle 8 Typen f32-exakt
vs gguf-py — Rezept: Tabellen programmatisch extrahieren, Golden+Fuzz je Typ;
gguf-py im .venv = lokale Cross-Referenz!); MXFP4 (Encode byte-exakt);
Sparse-Trio MoBA/NSA/DSA (Kollaps- + Isolations-/Routing-Beweise je Mechanik);
KDA (Kanal-Decay, Kollaps auf GatedDeltaNet — Test fing fehlende L2-Norm);
PagedAttention → ADR-0020 out-of-scope mit Revisit-Trigger. Muster der Ära:
Kollaps-Tests gegen existierende verifizierte Pfade tragen die ganze Familie.

**Die großen Ergebnisse dieser Session:**
- **ADR-0019 Batched Decode (T366–T412):** Recorder (ein Command-Buffer/Schritt)
  + DeviceBuffer auf metal UND vulkan, alle Decode-Ops record-mode (matmul/norm/
  rope/mha sq≠sk/blit-KV-append/qmatmul), beide Architekturen (GPT+Llama/GQA/
  SwiGLU). ÖFFENTLICHES Paket `llamagpu`: New/NewVulkan(*nlp.Llama) → Decoder →
  Step/StepN/Generate. Real-Modell-Decode **24× metal / 21× vulkan** vs nlp
  per-op; Logits == Referenz, Greedy token-für-token identisch, GGUF(F32) ok.
- **Attention MPS-Reformulierung (T393–T400):** mha fwd als MPS(QKᵀ)→softmax→
  MPS(PV): 6,9× Kernel / **1,87× GPT-Forward**; mha BACKWARD analog (+softmax-
  jacobian): 27× Kernel / **2,04× Training**; GQA/MQA via MPS-beta-Akkumulation.
  Der Hand-Flash-Kernel war 15× langsamer als seine eigenen zwei Matmuls (§T393-
  Floor-Messung — verhinderte einen falschen Cooperative-Tiling-Rewrite).
- **Silent-Fallback-Audits (T401–T403):** OpCrossEntropy-FWD (20ms!) + OpEmbed-FWD
  fehlten auf metal+vulkan (Backwards existierten!) → Kernel → Training kumulativ
  1133→2997 tok/s (2,6×). Definitve Audit-Methode: GOAI_LOG_FALLBACK=1 (execute.go)
  am ECHTEN Workload; Standalone-Op-Timing führt in die Irre (OpSum-Falle §T402).
- **Op-Profiling (T410):** GOAI_TIME_OPS=1 (execute.go). Fand die T402-embed-
  Regression (GPU-Gather lud 8MB-Tabelle/Call → Host-f32-Gather, beide Backends).
- **Quant-Decode (T413–T416):** Recorder-QMatMulResident (alle ggml-Typen, beide
  Backends) + llamagpu.NewQuant: Q8-Decode 3,6× vs per-op-quant; ~16% langsamer
  als f32-batched → Q8-Wert = 4× SPEICHER, nicht Speed.
- **StepN + Speculative (T418–T420):** Multi-Token-Step → **Prefill 41×** (Generate
  prefillt via EINEM StepN); llamagpu.SpeculativeGenerate (Draft-Step + Target-
  StepN + nlp.SpeculativeRun, lossless, KV-Rollback gratis via Positions-Cache);
  Kosten gemessen: Speedup 1,95×@50% / 2,65×@80% Akzeptanz (braucht trainierte
  verwandte Modelle).
- **GPT + Feature-Vervollständigung (T421–T426):** GPTDecoder (LayerNorm+pos-emb+
  AddBias-record-op, beide Backends) inkl. StepN; SpeculativeGenerate über das
  exportierte Stepper-Interface (beide Architekturen); PromptLookupGenerate
  (draft-frei, nlp.NgramLookup, 45% Akzeptanz auf repetitiven Prompts, lossless);
  6 Examples; Decode/Prefill als stehende bench-compare-Benchmarks (205/200 tok/s
  batched, 27×/36× — Regression-Guard). Finale Matrix: {Llama,GPT}×{Step,StepN,
  Generate,Speculative,PromptLookup}+Llama×Quant.
- **Long-Context / kooperative Attention (T428–T432):** Der Zwei-Pass-MHA-Kernel
  war seriell in der Sequenzlänge → Klippe bei großem KV (242ms/Step @2k). NEUER
  kooperativer Kernel: EIN 32-Lane-simdgroup (metal, simd_shuffle_down) bzw.
  Subgroup (vulkan, SPIR-V 1.3, glslc --target-env=vulkan1.1) pro (Query-Zeile,
  Head), Lanes partitionieren Keys, Online-Softmax-Partials (m,l,acc[dk≤128]) in
  Registern via Lane-Shuffles gemerged; NaN-Guard beim Merge leerer Lanes
  (M==-INF→c=0, §T428-Bug). Deckt ALLE Attention-Oberflächen: Recorder-Decode
  (T428/T429: 242→13,8ms, **17,6×**; Standing-Bench T430: 72,3/71,0 tok/s
  @L=1920), Recorder-Prefill-Fenster sq>1 (T431: per-Zeile jmax=sk-sq+i+1,
  291→104ms), per-op OpMHA (T432: metal+vulkan Host-Slice-Wrapper, sq=1 @sk=1920
  ~40→2,18ms). Kurz-Kontext ehrlich unverändert (dispatch-bound, §T430/T432).

- **Integrations-Ära (T434–T444), Methode „Orphan-Audit":** Systematisch alles
  gefunden, was exportiert-aber-unverdrahtet war, und mit kleinen Adaptern/Loops
  an echte Modelle angeschlossen. (a) T434: Speculative mit ECHT trainierten
  Modellen — „braucht Model-Files" via IN-REPO-TRAINING aufgelöst (81% Akzeptanz,
  aber nur 1,12× — dispatch-bound Draft/Target-Ratio; lohnt erst bei großen
  Targets). (b) T435 GPT.Safetensors() Checkpointing (bit-Round-Trip). (c) T436
  ApplyPenalties→Sampler (SampleWithHistory, 7 Loops). (d) T437 TokenSampler-
  Interface → Mirostat generierfähig. (e) T438 mechanisches Audit: Klasse leer;
  nlp/doc.go deckte nur ~1/3 der Features — neu geschrieben. (f) T439
  Watermark.Sampler + RegexGuide.Sampler (Kompositions-Adapter; scharfe Tests:
  z=4,58-Detektion, (ab)+-Erzwingung). (g) T440 CFGDecode (γ=1/γ=0-Äquivalenzen).
  (h) T441 GPT.JacobiGenerate (lossless vs greedy, 6 Tok in 5 Iter). (i) T442
  ForwardEarlyExit+DoLaDecode (bit-Identitäten, α=1-Äquivalenz). (j) T443/T444
  Medusa KOMPLETT: ForwardHidden, trainierbare Heads (frozen-base Tape), Chain-
  MedusaGenerate mit Typical Acceptance — Heads auf Base-Rollouts: 100% Akzeptanz.
  KEIN Algorithmus in nlp ist mehr ohne lauffähigen Real-Modell-Pfad.
  ÜBERTRAGBARE LESSONS: (1) exportiert-mit-Tests ≠ nutzbar — Call-Site-Audit
  lohnt; (2) In-Repo-Training entsperrt jede „braucht Artefakte"-Messung;
  (3) scharfe Äquivalenz-Invarianten (Parameter-Extremwerte kollabieren neue
  Pfade auf bekannte) schlagen weiche Qualitätsasserts.

- **Decode-Beschleunigungs-Ära (T446–T455):** Batched Medusa (StepHidden/
  StepNHidden + MedusaGenerate über HiddenStepper, beide Architekturen) —
  T446: 1,81× @97% Akzeptanz (bestätigte die T434-These: Draft-KOSTEN, nicht
  Akzeptanz, sind der Hebel); T455 halbierte die Rundenkosten via lastTok-lead-
  window (Proposals aus dem Verify-Pass, StepNHidden) → **3,08×** (1152→3546
  tok/s). Prompt-Lookup real gemessen (T452): **1,80× LOSSLESS @15%** — billige
  Runde schlägt hohe Akzeptanz. Mess-Gotcha: sequentielle A/B-Blöcke fangen
  Kalt-Ausreißer → verschränkt messen (A,B,A,B). Alles in benchmarking.md.
- **CV-Ära (T456–T463):** §G1 geliefert — nn.Conv2D/MaxPool2D-Layer, vision-
  Paket (CNN, 100% auf Spatial-Task, Checkpoint-Round-Trip), dann Perf-Faden
  §V22-profilgetrieben: cpu conv2d_backward auf im2col+GEMM (637→30,6 ms/Step),
  cpu-Pools (bit-identisch), Fallback-Kette aktiv→cpu→ref (T461 — Metal-Pools
  liefen sofort schneller, 39→33), Scratch-sync.Pool gegen madvise-Churn
  (→24,3). KUMULATIV 637→24,3 = **26×**; CPU schlägt Metal bei kleinen CNNs
  (§T361-Größenabhängigkeit gilt auch für CNNs). Geparkt: parallelWork-
  Barrieren-Boden (42% des Profils) — Persistent-Worker-Pool nur bei Bedarf.
- **Alignment-Ära (T464–T470, KOMPLETT — docs/alignment.md):** DRITTE Orphan-Klasse „dokumentiert-aber-nie-
  gebaut": SequenceLogProbs war nur Doku-Referenz. Gebaut: TokenLogProbs/
  SequenceLogProbs (stabiler komponierter Log-Softmax-Gather, Gradcheck).
  FLAGSHIPS: GRPO trainiert die echte GPT-Policy 0,042→0,979 Reward (Generate-
  Rollouts + GroupAdvantage + GRPOLoss; Lesson: spärliche Null-Reward-Gruppen
  ⇒ Advantage 0 — längere Rollouts/größere Gruppen/KL-β runter); DPO: 3/3
  positive Margins, chosen↑/rejected↓ vs Referenz. T468: IPO/SimPO/CPO/KTO
  VERIFIZIERT (Verträge unterscheiden sich wirklich: SimPO längennormiert ohne
  Ref, KTO ungepaart+Labels, CPO+NLL) — alle drehen die anfangs NEGATIVE Margin.
  T469: volle RM+GRPO-Pipeline — REWARD-HACKING reproduziert (gelernter Reward
  steigt, True-Metrik →0; bei jeder Head-Kapazität). T470: iteriertes RLHF
  (RM-Refresh alle 5 Updates auf frischen Policy-Samples) rettet die True-
  Metrik KOMPLETT (0,042→1,000; 8er-Kadenz nur 0,104 — Frequenz ist der Hebel;
  KL allein reicht nicht). Auch: V23 (CGO0-Gate kompiliert Tests, §B45),
  README/doc.go-Audits aller Pakete, GPT/CNN-Checkpointing (T435/T458).

**GEPARKT mit Zahlen (nicht ohne neue Evidenz wieder anfassen):**
- Tape-Batching fürs Training: Decke 1,4× @S256 (§T411) — Compute dominiert.
- Conv-Gap: ≤2×, nicht auf dem LLM-Pfad, MPSCNN=große API (§T417).
- MPS-Matmul-Rate (~3,5× zu torch): Apples Bestes; 49% der Trainings-Zeit (§T410).
- Zero-Copy UMA (§B42), Vulkan-GEMM-Blocking (§B39/41), mha_bwd-Register (§B43),
  Vulkan-Attention-fwd-Reformulierung (§T397/398: isoliert 6-8×, real LANGSAMER —
  Attention ist nicht Vulkans Bottleneck, die FFN-Matmuls sind es).

**MESS-DISZIPLIN (§V22, PFLICHT — diese Session: ~6 Überraschungen, ~5 verhinderte
Fehlbauten):** (1) ECHTEN Workload messen/instrumentieren (GOAI_LOG_FALLBACK /
GOAI_TIME_OPS / Bench-Suites), nie Standalone-Ops. (2) Floor A/B-messen VOR jedem
Rewrite (§T393/T396/T411/T417). (3) Nach jedem Kernel den echten Workload RE-messen
(§T410-Regression). (4) Isolierter Gewinn ≠ Real-Gewinn (§T397). (5) Bottleneck ist
BACKEND-SPEZIFISCH (metal: Attention war es; vulkan: Matmul). (6) Same-Session-A/B,
kein git stash (Repo=1 Commit), Temp-Swap via Scratchpad.

**Dispatch-Regeln (für alle künftigen Kernel):** Metal nie
maxTotalThreadsPerThreadgroup für per-row/registerlastige Kernel — 64/TG bzw.
kooperativ (§T337/345); MSL/GLSL kein erf → A&S (§T352); Backward als OpXBackward
auf aktivem Backend (§T353); transponierte Operanden per Flag (§T356/357); GPU nie
für winzige memory-bound Ops (Gather §T410); Recorder-Ops brauchen auf vulkan
per-Op-Descriptor-Sets + explizite Barriers (§T382), auf metal Hazard-Tracking;
VK_MAX_PIPELINES-Headroom beachten (§T397-Bug: 33>32 → Random-Shader-Fail);
apicheck verbietet Magic-Backend-Strings (§C15) und will Forwards UND Backwards
pro Op geprüft (§T401-Asymmetrie).

**Nächste Kandidaten:** CPU-GEMM/archsimd (host-blockiert T11b/T74, braucht
amd64); MoE/MLA/weitere Architekturen auf den Recorder (nur mit Demand);
Tree-Medusa via MedusaTreeMask (braucht Recorder-MHA mit freiem Masken-Input —
nur mit gemessenem Bedarf über die 3,08× hinaus); LoRA-Naht durch die
fusionierten Modell-Projektionen (Design-Task + ADR, §T449-Befund);
parallelWork-Worker-Pool (§T463-Boden, 42% Barrieren); IPO/KTO/SimPO/CPO-E2E
(je ein Loss-Call-Swap auf T465-Muster, geringer Erkenntniswert); echte
Modell-Gewichte (GPT-2/TinyLlama) NUR mit Nutzer-Erlaubnis für Downloads.
Fast alles Verbliebene ist extern blockiert oder demand-gated — der Loop ist
im Wartungs-/Opportunitätsmodus.

## Arbeitskontext (Stand zuletzt aktualisiert: 2026-07-13)

- Toolchain: Go 1.26.4 (arm64 host), git, `.venv` mit numpy 2.5.1 + torch
  2.12.1 (`make golden`; torch-Goldens via `testdata/verify_torch.py`);
  Vulkan via MoltenVK (brew; `VK_ICD_FILENAMES` setzt das Makefile).
- Referenz-Backend `ref` = numerische Wahrheit (§V9); `cpu` = CGO0-Default;
  auf macOS ist `metal` der Zero-Config-Default (§T47/§B37); `vulkan` per
  Build-Tag, host-verifiziert (§B36).
- Benchmarks: `docs/benchmarking.md` (+ Snapshot-Tabelle §T338);
  `make bench-compare` = Cross-Backend-Harness; ADRs: `docs/decisions/`.
- Beim A/B-Messen: kein `git stash` (Repo hat nur den Initial-Commit —
  Stash setzt auf den Ur-Zustand zurück); alte Variante temporär einsetzen
  und aus dem Scratchpad-Backup wiederherstellen (Muster §T336/§T341).
