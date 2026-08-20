# M2 ViT Constant-Dispatch Boundary Rejection (2026-08-20)

## Outcome

Rejected and reverted. The prototype reduced the B=8 preparation/assembly/class-extraction
boundary from 51 backend operations to 5 and preserved exact F64 logits at B=1/4/8. It did not
clear the frozen end-to-end M2 gates: B=8 training was approximately flat, forward speedup was
not repeatable at 1.25x, and the control arm exceeded the 1.15x spread ceiling. No ViT executable
change remains.

The experiment exposed an independent general win: host-resident Metal `OpEmbedBackward` should
not upload, submit an atomic scatter, synchronize, and download its full output table. That finding
is retained and gated separately by Spectackle proposal `P-01M0FNKC7DEJZ`.
The rejected batch-loop pattern is reported as
[jxsl13/perfscan#772](https://github.com/jxsl13/perfscan/issues/772).

## Frozen contract

- Base: `fd6d08bb03188224ce3e63761b71b8581f8752bb`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Candidate boundary: host-only batch patchification; one class/patch table concat; two
  differentiable `OpEmbed` gathers plus one add; one final class-row `OpEmbed` gather.
- Exactness: exact F64 forward parity at B=1/4/8; all non-Class/Pos gradients bit-identical;
  Class/Pos gradient relative RMS at most 1e-6.
- Structure: remove at least 46 B=8 boundary operations.
- End-to-end B=8 gates: forward at least 1.25x, train at least 1.15x, control spread at most 1.15x.
- Any failed gate requires reverting executable changes.

## Correctness and structural result

The temporary mutation-proven control/candidate tests passed:

- F64 logits: exact at B=1/4/8.
- F64 backward: exact for every parameter other than repeated Class/Pos rows; those rows stayed
  below the 1e-6 relative-RMS budget.
- B=8 target boundary: 51 operations in the control versus 5 in the candidate (46 removed).
- Candidate operation mix: one concat, three embeds, one add; zero slices and reshapes.

The tests and candidate implementation were removed after the performance rejection, as required.

## M2 measurements

Command used for each independent campaign:

```text
go test -tags=vulkan ./vision -run '^$' \
  -bench '^BenchmarkViTBatchBoundary/B8/metal/(Forward|TrainStep)/(control|candidate)$' \
  -benchtime=1x -count=<campaign count>
```

Medians are nanoseconds per full B=8 call; speedup is control/candidate.

| Campaign | Count | Phase | Control median | Candidate median | Speedup | Control spread | Gate |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 3 | Forward | 24,699,584 | 19,810,250 | 1.247x | 1.071x | fail (<1.25x) |
| 1 | 3 | Train | 77,443,208 | 93,798,958 | 0.826x | 1.388x | fail |
| 2 | 5 | Forward | 25,347,209 | 20,059,875 | 1.264x | 1.188x | fail (spread) |
| 2 | 5 | Train | 94,052,958 | 93,238,208 | 1.009x | 1.420x | fail (<1.15x) |
| 3 | 7 | Forward | 23,636,084 | 19,251,625 | 1.228x | 1.431x | fail |
| 3 | 7 | Train | 93,215,000 | 93,387,833 | 0.998x | 1.434x | fail |

Campaigns 2 and 3 used the independently faster deterministic host embedding-backward route. It
removed that primitive as a confounder but did not make the ViT boundary a training lever.

## Decision

The boundary dispatch count was real, but the packed encoder backward dominates a full training
step. Removing preparation commands alone cannot support the frozen 1.15x training claim. Future
ViT work must attack a larger resident encoder/attention boundary rather than relabel this
inference-local improvement as end-to-end leadership.
