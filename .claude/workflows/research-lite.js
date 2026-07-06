export const meta = {
  name: 'research-lite',
  description: 'Small-scope, schema-free web research: fan out 3 compressing sub-agents on ONE focused question, synthesize a short cited verdict. Mitigates the deep-research StructuredOutput retry-cap failure by never using a schema.',
  phases: [
    { title: 'Angles', detail: '3 independent sub-agents, each returns <=6 compressed lines' },
    { title: 'Synthesize', detail: 'one agent compresses to a final verdict' },
  ],
}

// ONE focused question per run (keep scope small → protect main-loop context).
const question =
  typeof args === 'string' && args.trim()
    ? args.trim()
    : (args && args.question) || 'No question provided.'

// Fixed independent angles. Small on purpose.
const ANGLES = [
  'the primary spec or official reference IMPLEMENTATION source code',
  'the original paper or a canonical textbook definition',
  'a second independent corroborating source (official docs / well-known library)',
]

// --- MITIGATION OF THE StructuredOutput FAILURE ---
// The built-in deep-research forces every agent to emit a StructuredOutput
// schema; those calls hit the 5-retry cap under rate limits and the whole
// workflow throws. Here NO agent() call passes `schema`, so the StructuredOutput
// tool is never invoked and that error is structurally impossible. Sub-agents
// self-compress to a handful of lines (context protection). Dead agents return
// null (never throw) and are filtered — graceful degradation instead of a crash.

phase('Angles')
const findings = (
  await parallel(
    ANGLES.map((angle, i) => () =>
      agent(
        `Research ONE focused question, ONLY from this angle: ${angle}.\n\n` +
          `QUESTION:\n${question}\n\n` +
          `Use web search / fetch as needed, then COMPRESS to AT MOST 6 lines. ` +
          `Each line format: <claim> | VERDICT(confirmed|refuted|unclear) | <source url or citation>. ` +
          `No prose, no preamble, no restating the question — output only those <=6 lines. ` +
          `If you cannot verify, output one line: "unclear | <one-clause reason>".`,
        { label: `angle:${i + 1}`, phase: 'Angles' },
      ),
    ),
  )
).filter(Boolean)

if (findings.length === 0) {
  // Every angle agent died (e.g. rate limit). Return cleanly — NEVER throw.
  return {
    question,
    verdict:
      'UNVERIFIED — all angle agents failed to return (likely rate limit). No StructuredOutput crash; safe to retry later.',
    findings: [],
  }
}

log(`${findings.length}/${ANGLES.length} angles returned; synthesizing`)

phase('Synthesize')
const synthesis =
  (await agent(
    `Compress independent research findings into ONE final verdict.\n\n` +
      `QUESTION:\n${question}\n\n` +
      `FINDINGS FROM ${findings.length} INDEPENDENT AGENTS:\n` +
      findings.map((f, i) => `--- agent ${i + 1} ---\n${f}`).join('\n') +
      `\n\nOutput AT MOST 8 lines. Line 1 = overall VERDICT (CONFIRMED|REFUTED|PARTIAL|UNVERIFIED) + one-clause reason. ` +
      `Following lines = the key claims each with its single best source. Compress hard, no prose. ` +
      `Flag any claim only one agent supports as [single-source].`,
    { label: 'synthesize', phase: 'Synthesize' },
  )) || 'Synthesis agent failed; rely on the raw angle findings below.'

return { question, verdict: synthesis, rawFindings: findings }
