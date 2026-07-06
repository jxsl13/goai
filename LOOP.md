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
6. **Plattform-Check:** Pure-Go-Build ohne C-Toolchain (`CGO_ENABLED=0`) grün
   auf macOS/Windows/Linux; Accel hinter Build-Tags mit Fallback.
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

## STATUS: Loop-Endzustand auf diesem Host erreicht (2026-07-06)

28/29 §T-Tasks done. T11b (`~`) ist die einzige offene Task und auf diesem
arm64-Host strukturell nicht ausführbar: der archsimd-Teil braucht eine
amd64-Laufzeit für V-CROSS (§B13); der NEON-Teil wurde per Messung geparkt
(§B27: Loop läuft bereits bei 55 GB/s ≈ 1 elem/cycle; Netto-Gewinn ≤7 %).

**Wiederaufnahme-Bedingungen** (dann Loop neu starten):
1. amd64-CI-Runner o. amd64-Host verfügbar → T11b archsimd bauen + V-CROSS.
2. Go-Release mit archsimd-ARM64-Support (§R3) → NEON-Frage neu bewerten.
3. Neue §T-Tasks im Spec (z. B. aus §B-Kandidaten: B14 pooled outputs, B18
   broadcasting, B22 LayerNorm-VJP, GGUF-LLM-E2E, Vulkan/CUDA-Backend).

## Arbeitskontext (Stand zuletzt aktualisiert: 2026-07-06)

- Toolchain: Go 1.26.4 (arm64 host), git, `.venv` mit numpy 2.5.1 + torch
  2.12.1 (`make golden`; torch-Goldens via `testdata/verify_torch.py`).
- Referenz-Backend `ref` = numerische Wahrheit (§V9); optimiertes `cpu` =
  Default; archsimd-amd64 nur CI-verifizierbar (§B13, T11b).
- Benchmarks: `docs/benchmarking.md`; ADRs: `docs/decisions/`.
