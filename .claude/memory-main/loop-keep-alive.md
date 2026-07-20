---
name: loop-keep-alive
description: Never self-stop the GoAI /loop AND never idle/hold — blocked means optimize or delegate, not wait
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
---

Der GoAI-`/loop 1m` darf NICHT eigenständig beendet werden — UND es darf auf gar
keinen Fall ge-idlet/gehalten werden. Beides ist eine harte Nutzer-Vorgabe
(jetzt SPEC.md **C22: NEVER IDLE**).

**Why:** (1) Loop-Stop: zweimal (2026-07-13) bei leerem Backlog gestoppt → Nutzer
startete beide Male sofort manuell neu; will den Loop als stehenden Kanal. (2)
Idle-Verbot: am 2026-07-14 hielt ich ~25 Fires lang nur „bis zum Push-Fenster"
(C16-Throttle), Wakeup nach Wakeup ohne Arbeit — der Nutzer unterbrach emphatisch:
„was für halte, kein halten kein idle, kein garnix. mache performance
optimierungen" und „auf gar keinen fall ge-idlet werden darf". Der C16-Throttle
begrenzt nur das PUSHEN, nicht das ARBEITEN — lokal committen ist unbegrenzt.

**How to apply:**
- NIEMALS ein bloßes „hold bis zum Fenster"-Wakeup ohne Arbeit — das ist ein
  C22-Verstoß. Jeder Fire, der blockiert ist (Push-Fenster/CI/Wakeup), MUSS echte
  Arbeit starten oder voranbringen, DANN erst reschedulen.
- DEFAULT-Arbeit wenn blockiert = PERFORMANCE-OPTIMIERUNGEN (Python/C++/torch
  schlagen — §perf-gap, T620-Conv-Rungs; jeder gemessene A/B-Rung = §T + docs-Zeile).
  Sonst Research/Tests/Docs/Audits.
- Kontext GESÄTTIGT ist KEIN Idle-Grund, sondern ein DELEGATIONS-Grund: worktree-
  isolierte Fresh-Context-Subagenten (bewährtes Muster §T620: vulkan fused conv
  +20%, gemessener A/B). Bei GPU-Arbeit nicht >1 GPU-Agent parallel (B55-Kontention
  verfälscht A/B).
- „nichts Sicheres/Unblockiertes zu tun" ist KEINE gültige Ausrede — es gibt immer
  einen Perf-Rung zum Versuchen+Messen.
- Stoppen NUR auf ausdrückliche Nutzer-Anweisung („stopp").
- Siehe [[goai-autonomous-loop]], [[perf-gap-vs-python]], [[integration-audit-method]].
