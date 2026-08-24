# ADR-0030 — External perfscan is canonical with a parity-gated compatibility lane

- Status: accepted
- Date: 2026-08-24
- Relates: `R-01M0S05JDTENN`, `P-01M0S0D3ZPFCB`, `T-01M0S0F690EA7`

## Context

GoAI grew a repository-local performance scanner whose checks and fixtures were
being maintained independently from `github.com/jxsl13/perfscan`. The external
tool now has the larger registry, shared documentation, graded fixes, baselines,
and SARIF output. Continuing to default to the fork makes every detector fix and
new check pay twice.

A registry comparison prevents a simplistic replacement. The in-tree registry
has 146 IDs and external v1.81.0 has 375. They share 93 IDs, so the external tool
adds 282 but still lacks 53 benchmark-derived GoAI checks. A larger total is not
proof that specialized coverage did not regress.

The old `github.com/jxsl13/perfscan/perfscan` submodule path stops at v1.71.0.
The current executable is the repository-root module
`github.com/jxsl13/perfscan`, released as v1.81.0. GoAI resolves that exact
version with `GOPROXY=direct`.

## Decision

1. `make perfscan`, CI, and the autofix workflow use
   `github.com/jxsl13/perfscan@v1.81.0` with `GOPROXY=direct`.
2. `perfscan.yaml` at repository root is the single project vocabulary.
3. CI has a dedicated whole-tree external scanner lane. Existing findings are
   advisory; tool, configuration, and registry drift are hard failures.
4. `make perfscan-compat` runs only the IDs in
   `perfscan-compat-checks.txt`. That file must equal the computed set difference
   between the in-tree and external registries.
5. The legacy engine and fixtures remain only until that set difference is
   empty. Upstream issue <https://github.com/jxsl13/perfscan/issues/877> tracks
   the remaining ports.
6. The nested Spectackle history and the exactness mutation utility remain in
   place during the staged retirement.

## Consequences

- GoAI immediately gains 282 checks and shared upstream maintenance without
  dropping the 53 local checks.
- The local CI cost increases. On the M2 Pro validation host, the external scan
  had a 25.08-second median, compatibility 5.14 seconds, and the registry audit
  0.83 seconds. This is accepted tooling overhead, not presented as a speedup.
- Every upstream port shrinks `perfscan-compat-checks.txt`; an unreviewed registry
  change fails the audit instead of silently changing coverage.
- The legacy package is no longer forced into every selective Go test lane. Its
  role is explicit in the dedicated scanner lane and ordinary full-suite tests.

## Revisit

Delete the compatibility engine, fixtures, selector, and parity script only
when the computed internal-only registry contains zero IDs and external fixtures
cover the behavior previously pinned by GoAI.
