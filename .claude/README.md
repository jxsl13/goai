# In-repo Claude Code harness (portable)

Everything a Claude Code (or Agent-SDK) session needs to work on GoAI lives here, so
the setup travels to other hosts (Linux/Windows for the CUDA/AMD/Intel accel work)
without depending on a machine's global `~/.claude`.

## What's vendored

| Path | Purpose |
|------|---------|
| `.claude/workflows/research-lite.js` | The schema-free 3-agent web-research workflow for §V16 tier-2 paper verification (never the built-in `/deep-research`; see `LOOP.md`). Invoke: `Workflow({scriptPath: ".claude/workflows/research-lite.js", args: "<one focused question>"})`. |
| `.claude/skills/` | The full workflow-skill suite, so the whole build loop is portable: `spec` (mutate SPEC.md), `build` (implement one §T task), `research` (external facts → §R), `review` (red-team before build), `grill` (sharpen a fuzzy idea), `deepen`, `check`, `backprop` (bug → §B), `find-skills`, and `caveman` (the SPEC.md encoding skill, slimmed — see below). |
| `FORMAT.md` (repo root) | The **authoritative** caveman encoding — grammar, symbol table, `§V`/`§T`/`§B` row shapes. The skills point here; nothing restates it. |
| `LOOP.md` (repo root) | The autonomous per-iteration build-loop prompt the cron/loop references. |

To use on a fresh host, symlink or copy `.claude/skills/` into the machine's
`~/.claude/skills/` (or run Claude Code with this repo as the project root so the
project-local `.claude/` is picked up). The `loop` skill itself is a built-in Claude
Code skill; only `LOOP.md` (its per-iteration instructions) needs to travel, and it
lives at the repo root.

## Token-optimization decisions (evidence-based)

These are deliberately conservative — verified benchmarks (JetBrains SkillsBench
A/B, SkillReducer arXiv:2603.29919, Anthropic pricing/context-editing docs) show
the "caveman"/verbosity-reducer style saves only **~8.5% of output tokens** on
agentic tasks (code, diffs, tool calls dominate and are left verbatim), not the
advertised 65–75%. So we keep caveman lean and lean on the real levers.

1. **Caveman level = `lite`.** The `cavemem` hook re-injects the style on every
   `UserPromptSubmit` (~1–1.5k input tokens/turn). `lite` injects less; quality is
   statistically indistinguishable from `full` (JetBrains sign-test p=0.82). Set via
   a `.caveman-active` file containing `lite` next to the skills dir.
2. **Skill body slimmed by progressive disclosure.** `SKILL.md` no longer duplicates
   `FORMAT.md`; it keeps only the non-derivable guardrails and defers the full
   encoding to `FORMAT.md` (loaded once by `/spec`). ~72% body reduction, matching
   SkillReducer's finding that most skill body is non-actionable.
3. **`FORMAT.md` stays the single source** of the encoding (24 lines, well under
   Anthropic's <200-line guidance for always-loaded context).

### The real levers (bigger than caveman), for whoever runs this

- **Prompt caching** — Claude Code caches its own system prompt / tool defs / history
  automatically (cache read = 10% of input price). Keep prefixes stable.
- **Context hygiene** — `/clear` on task switch, `/compact` (with a focus note) at
  ~40–50% context; the Agent-SDK `clear_tool_uses` (context editing) cut tokens 84%
  in Anthropic's 100-turn eval.
- **Model routing** — Sonnet for most coding, Opus only for hard architecture/debug,
  Haiku (~1/5 the input price) for mechanical/subagent work.
- **Tool-output bloat** is usually the biggest hidden cost — delegate verbose work
  (tests, log dumps) to subagents that return summaries; prefer CLI over MCP.
- Measure first: `/context`, `/usage`, `ccusage`.

### Optional, host-specific (NOT committed — would break cross-OS)

The `cavemem` hooks (`SessionStart`/`UserPromptSubmit`/`PostToolUse`/`Stop`) are wired
in the machine's global `~/.claude/settings.json` with an absolute node path, so they
are intentionally left out of the repo. If a host wants the per-turn caveman inject,
add them there and drop a `~/.claude/.caveman-active` file containing `lite`. Given the
~8.5% ceiling, running without the hook (relying on `FORMAT.md` + the skill on demand)
is a defensible, lower-token default.
