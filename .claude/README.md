# In-repo Claude Code harness (portable)

Everything a Claude Code (or Agent-SDK) session needs to work on GoAI lives here, so
the setup travels to other hosts (Linux/Windows for the CUDA/AMD/Intel accel work)
without depending on a machine's global `~/.claude`.

## What's vendored

| Path | Purpose |
|------|---------|
| `.claude/commands/` | Generated slash commands for the spectackle spec server: `/spectackle` (the loop itself), `/spectackle-state`, `/spectackle-find`, `/spectackle-get`, `/spectackle-research`, `/spectackle-swarm`, `/spectackle-export`, `/spectackle-merge`. Regenerate after a server upgrade with `spectackle call commands '{"op":"gen","harness":["claude"]}'`. |
| `.claude/workflows/research-lite.js` | The schema-free 3-agent web-research workflow for tier-2 paper verification (never the built-in `/deep-research`; see `LOOP.md`). Invoke: `Workflow({scriptPath: ".claude/workflows/research-lite.js", args: "<one focused question>"})`. |
| `.claude/skills/find-skills/` | Helper for discovering and installing additional agent skills. |
| `LOOP.md` (repo root) | The autonomous per-iteration build-loop prompt the cron/loop references. |

To use on a fresh host, install the spectackle server (`brew install spectackle`)
and run Claude Code with this repo as the project root so the project-local
`.claude/` is picked up. The `loop` skill itself is a built-in Claude Code skill;
only `LOOP.md` needs to travel, and it lives at the repo root.

## Where the spec lives

The spec is **not** a markdown file in this repo. It lives in server-owned
`.spectackle/` bundles — one at the repo root plus one per context directory
(`backend/cpu`, `backend/cuda`, `classic`, `format`, `internal/simd`, `nlp`,
`nn`) — and is read and written **only** through the spectackle server. Never
hand-edit those files; the server owns them and a manual edit will be
overwritten or rejected.

Each bundle holds:

- `spec.md` — the living contracts as EARS rules, with stamped code anchors so
  drift against the real symbols is detectable.
- `work.md` — active lifecycle items (proposals, tasks, ADRs, research).
- `journal.ndjson` — append-only history, including every rejection.

Entry points:

```bash
make spec-state
```

```bash
make spec-check
```

```bash
make spec-index
```

## Token-optimization decisions (evidence-based)

These are deliberately conservative — verified benchmarks (JetBrains SkillsBench
A/B, SkillReducer arXiv:2603.29919, Anthropic pricing/context-editing docs) show
a verbosity-reducer prompt style saves only **~8.5% of output tokens** on
agentic tasks (code, diffs, tool calls dominate and are left verbatim), not the
advertised 65–75%. So we lean on the real levers instead.

The spectackle server is itself a token lever, and the larger one: it returns
IDs and spans rather than file contents, so a lookup costs on the order of the
result set instead of the codebase. Prefer `find`/`get` over shell `grep` when
you only need locations or structure.

### The real levers, for whoever runs this

- **Prompt caching** — Claude Code caches its own system prompt / tool defs / history
  automatically (cache read = 10% of input price). Keep prefixes stable.
- **Context hygiene** — `/clear` on task switch, `/compact` (with a focus note) at
  ~40–50% context; the Agent-SDK `clear_tool_uses` (context editing) cut tokens 84%
  in Anthropic's 100-turn eval.
- **Model routing** — a strong model orchestrates (drafts proposals, writes task
  briefs, reviews), cheaper fresh-context models implement one approved task each.
  This is the division of labor the server is built around.
- **Tool-output bloat** is usually the biggest hidden cost — delegate verbose work
  (tests, log dumps) to subagents that return summaries; prefer CLI over MCP.
- Measure first: `/context`, `/usage`, `ccusage`.

## History

Until 2026-07-25 this repo ran a hand-rolled spec system that has since been retired. Its tables were migrated into the server-owned `.spectackle/` bundles; the narrative ledgers it produced live under `docs/history/`, kept only because live citation contracts reference their ids. All spec authoring now goes through spectackle.

All of that was migrated into spectackle and removed. The verification
invariants, constraints and architecture invariants became EARS contracts; the
architecture decision records became ADR items; the open tasks became lifecycle
items; the goals and layer model became intent prose. The historical corpus that
did not map onto a live contract — the completed task log, the research log and
the bug ledger — stays in git history and in `CHANGELOG.md`.
