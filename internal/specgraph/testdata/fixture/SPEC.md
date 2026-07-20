# SPEC — fixture

Synthetic mini-corpus for the specgraph extractor tests. Mirrors the real
SPEC.md section shapes (FORMAT.md): defs for §G/§C/§I/§V, tables for §R/§B/§T,
worker prose ids (Tw-X, Iw8) that must NEVER match, and a §T cites column.

## §G — goals

G1: tiny fixture goal — exercise §I.L0 and backend/cpu.

## §C — constraints

C1: pure Go, references §V1.

C2: multi-line def start,
  continuation line citing §G1.

## §I — architecture invariants

I.L0 core: tensor layer, no cgo.

I1: registry pattern, see §C1.

## §R — research log

| id | claim | source | conf |
| --- | --- | --- | --- |
| R1 | fixture claim about softmax | paper 2020 | high |

## §V — verification invariants

V1 FIXTURE-TAG: every op has a reference test. Prevents §B1-class waste.

V2 MULTI WORD TAG: tolerates spaces in the tag like the real V26.

## §B — backprop log

| id | date | cause | fix |
| --- | --- | --- | --- |
| B1 | 2026-01-02 | per-element allocation in backend/cpu softmax loop caused NaN poisoning (escape `a \| b`, commit abc1234) | bulk cast fixed it. Non-vacuous: guard test added. V1,T2. |
| B2 | 2026-01-03 | per-element allocation in nn.Fixture optimizer step loop caused same NaN class | flat fast path. V1. |

## §T — task backlog

| id | status | task | cites | state | priority |
| --- | --- | --- | --- | --- | --- |
| T1 | x | scaffold backend/cpu softmax (commit abc1234) | C1,V1,ADR-0001 | done | high |
| T2 | x | PERF nn.Fixture optimizer fast path, T1 class | V1,B1 | done | med |
| T3 | . | future work referencing §I.L0 | G1 | . | low |
