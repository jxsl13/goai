---
name: pre-push-gofmt-gate
description: Pre-push gate MUST include `gofmt -l` over the whole tree — go vet + mdlint don't catch formatting, and agent-added files are often unformatted
metadata:
  node_type: memory
  type: feedback
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
---

Vor jedem Push den GANZEN Baum auf gofmt prüfen:
`gofmt -l $(git ls-files '*.go' | grep -v testdata)` — muss LEER sein.

**Why:** am 2026-07-14 scheiterte die CI-Lane `cgo+race / ubuntu` (die `gofmt -l .`
VOR den Tests laufen lässt) an einer un-formatierten, von einem Subagenten
hinzugefügten `backend/metal/metal_test.go` — obwohl mein lokaler Pre-Push-Gate
(CGO0-build, `go vet -tags ...`, mdlint, cross-ref-Tests) alles grün meldete. `go
vet` prüft KEIN Formatting; mdlint prüft nur Markdown; der pre-commit-Hook läuft
nur mdlint. So rutschte die un-gofmt'te Agent-Datei durch → rote CI → ein extra
Push (C16-Ausnahme „rote main fixen") nötig.

**How to apply:**
- gofmt-Check IMMER in den Pre-Push-Gate, besonders nach dem Cherry-picken von
  Subagenten-Commits (Agenten gofmt'en ihre neuen Dateien oft nicht — der
  metal-Agent formatierte metal_test.go nicht).
- Bequem: `gofmt -w` auf alle geänderten Dateien direkt nach dem Merge/Cherry-pick,
  bevor der nächste Commit/Push.
- Der volle Pre-Push-Gate = CGO0-build + `gofmt -l` (ganzer Baum) + `go vet` +
  mdlint + apicheck + die relevanten (targeted, §B55) Cross-Ref-Tests.
- Siehe [[goai-autonomous-loop]], [[worktree-agent-stale-base]].
