export const meta = {
  name: 'perfscan-autofix',
  description: 'perfscan → parallel agents implement each fix (worktree-isolated, bit-exact) → SERIAL bench-validate → keep only benchmark-proven wins and open draft PRs',
  whenToUse: 'Auto-fix perfscan findings the staticcheck way, but for perf patterns that need judgment + a benchmark. args: {checks?: string (default "strided-inner-reduction"), max?: number (default 6), benchtime?: string (default "200x")}',
  phases: [
    { title: 'Scan', detail: 'go run ./internal/perfscan -json; pick actionable, benchmark-backed candidates' },
    { title: 'Implement', detail: 'one agent per candidate — bit-exact fix + correctness test, parallel isolated worktrees, NO benchmarking' },
    { title: 'Validate', detail: 'SERIAL: bench PRE/POST each fix one-at-a-time (no ns/op contamination); keep wins' },
  ],
}

const checks = (args && args.checks) || 'strided-inner-reduction'
const max = (args && args.max) || 6
const benchtime = (args && args.benchtime) || '200x'

const CAND_SCHEMA = {
  type: 'object', required: ['candidates'],
  properties: {
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        required: ['file', 'line', 'category', 'funcName', 'pkg', 'benchName'],
        properties: {
          file: { type: 'string' }, line: { type: 'integer' },
          category: { type: 'string' }, message: { type: 'string' },
          funcName: { type: 'string' }, pkg: { type: 'string' },
          benchName: { type: 'string', description: 'existing Benchmark* that exercises funcName, or "" if none' },
        },
      },
    },
  },
}

const IMPL_SCHEMA = {
  type: 'object', required: ['branch', 'correctnessPassed', 'committed'],
  properties: {
    branch: { type: 'string' }, pkg: { type: 'string' },
    correctnessPassed: { type: 'boolean' }, committed: { type: 'boolean' },
    benchName: { type: 'string' }, reason: { type: 'string' },
  },
}

const VALIDATE_SCHEMA = {
  type: 'object', required: ['isWin'],
  properties: {
    branch: { type: 'string' }, preNs: { type: 'number' }, postNs: { type: 'number' },
    speedup: { type: 'number' }, isWin: { type: 'boolean' }, note: { type: 'string' },
  },
}

phase('Scan')
const scan = await agent(
  `Find perfscan candidates worth auto-fixing. Run from the repo root:\n` +
  `  go run ./internal/perfscan -config internal/perfscan/perfscan.json -checks=${checks} -json ./...\n` +
  `Parse the JSON findings. Return up to ${max} candidates. For each, resolve the enclosing FUNCTION name and its go package import path (pkg), and find whether an existing Benchmark* in that package exercises that function (set benchName to it, else ""). PRIORITIZE candidates that (a) have a benchName and (b) are in a hot inference/training kernel; SKIP _test.go, generated code, cold/fallback/error paths, and anything already contiguous. Do NOT modify any files.`,
  { label: 'scan', phase: 'Scan', schema: CAND_SCHEMA })

const candidates = (scan.candidates || []).filter(c => c.benchName).slice(0, max)
log(`scan: ${(scan.candidates || []).length} candidates, ${candidates.length} benchmark-backed`)
if (!candidates.length) return { shipped: [], note: 'no benchmark-backed candidates to auto-fix' }

phase('Implement')
const implemented = await parallel(candidates.map((c, i) => () =>
  agent(
    `Implement the perfscan "${c.category}" fix at ${c.file}:${c.line} (function ${c.funcName}, package ${c.pkg}).\n` +
    `Pattern: ${c.message}\n\n` +
    `RULES:\n` +
    `1. git checkout -B autofix/${c.category}-${i} origin/main (you are in an isolated worktree).\n` +
    `2. Implement the optimization. For strided-inner-reduction: INTERCHANGE the loops so the flat array ARR[inner*stride+outer] is walked CONTIGUOUSLY (make the strided reduction var the OUTER loop and the contiguous var INNER, using a per-output scratch accumulator). PRESERVE the exact per-output-element reduction order so it stays BIT-IDENTICAL. Keep any division as division (never ×1/x reciprocal). If the accumulator is a widened type (float64 over float32 inputs), keep it widened and round once at the end.\n` +
    `3. gofmt -w the file; go build the package.\n` +
    `4. Run ONLY the correctness tests: go test ${c.pkg} -run <relevant test regex> -count 1. DO NOT run any -bench (benchmarking happens serially later; running it here contaminates ns/op).\n` +
    `5. If correctness passes AND the change is genuinely bit-exact, git add + commit and set committed=true, correctnessPassed=true. If you cannot make it bit-exact or tests fail, DO NOT commit; set committed=false and explain in reason.\n` +
    `Return {branch, pkg, correctnessPassed, committed, benchName:"${c.benchName}", reason}.`,
    { label: `impl:${c.category}:${i}`, phase: 'Implement', isolation: 'worktree', schema: IMPL_SCHEMA })))

const ready = implemented.filter(Boolean).filter(r => r.committed && r.correctnessPassed && r.benchName)
log(`implemented: ${ready.length}/${candidates.length} committed & correct`)

phase('Validate')
// SERIAL loop — await each bench before the next so only ONE benchmark runs at a time.
const results = []
for (const r of ready) {
  const v = await agent(
    `Serially bench-validate branch ${r.branch} (package ${r.pkg}, benchmark ${r.benchName}). Nothing else is running.\n` +
    `1. In a fresh worktree of ${r.branch}, run POST: go test ${r.pkg} -run '^$' -bench '${r.benchName}$' -benchtime ${benchtime} -count 3. Take the median ns/op.\n` +
    `2. git checkout origin/main -- the file(s) this branch changed; run PRE with the identical command; take the median ns/op. Then restore the branch files.\n` +
    `3. isWin = postNs < preNs*0.98 (a real >2% speedup). Return {branch, preNs, postNs, speedup: preNs/postNs, isWin, note}.`,
    { label: `bench:${r.branch}`, phase: 'Validate', schema: VALIDATE_SCHEMA })
  results.push({ ...r, ...v })
  log(`${r.branch}: ${v.speedup ? v.speedup.toFixed(3) : '?'}x ${v.isWin ? 'WIN' : 'no-win'}`)
}

const wins = results.filter(r => r.isWin)
const losers = results.filter(r => !r.isWin)
return {
  scanned: candidates.length,
  implemented: ready.length,
  wins: wins.map(w => ({ branch: w.branch, pkg: w.pkg, speedup: w.speedup, benchName: w.benchName })),
  losers: losers.map(l => ({ branch: l.branch, speedup: l.speedup, note: l.note })),
  note: 'wins are committed on their autofix/* branches; review, then `gh pr create --draft` each. losers should be branch-deleted.',
}
