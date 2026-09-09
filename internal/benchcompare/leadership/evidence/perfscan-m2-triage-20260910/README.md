# M2 perfscan check and source triage

Read-only diagnostic on 2026-09-10. No auto-fix, production change, benchmark,
profiling run, performance gain or promotion is claimed.

## Scope and results

The analyzed Go/native source, module declaration and vocabulary are identical
to commit `e57d49aafa700e6428a1fcb6f2fad7b95a95e1ae`. Only diagnostic reports and
Spectackle records changed while scans ran. This is the existing draft GELU
branch, not a claim that its rejected runtime candidate qualifies for main.

| Check | Result |
| --- | --- |
| `make perfscan-check`, canonical v1.81.0 | PASS, 53 internal-only compatibility checks, focused PS1001 scan clean, legacy fixture tests pass |
| Installed v1.110.0, `-config perfscan.yaml -json ./...` | 2,035 advisories from 62 checks; 85 CPU-backend findings |
| Canonical v1.81.0, same whole-tree inputs | 1,890 advisories |
| Installed v1.110.0, `GOEXPERIMENT=simd`, `./backend/cpu` | 84 advisories from 11 checks |
| Go 1.27.1 autograd BCE diagnostics and ReLU disassembly | Residual in-loop bounds checks confirmed; compile and objdump exit 0 |

Whole-tree scans use darwin/arm64, CGO disabled and no `-tests`. These are static
findings, not proof of GPU-route execution or runtime cost. The scanner can also
report cross-build source evidence; an inactive-file finding is not an active
M2 SIMD-path defect. Direct invocation returns 1 when advisories exist; the
canonical integration gate returns 0. No baseline was rewritten or suppressions
added. Version v1.110.0 is the installed executable, not an assertion of latest.

Matching records by `(id,file,line,col)`, v1.110.0 adds 196 locations and no longer
reports 51. Seven newly represented check IDs are PS6080, PS6083, PS6086, PS6088,
PS6092, PS6093 and PS6095. PS4008 adds 17 locations under an existing ID. These are
scanner-coverage differences on unchanged code, not newly introduced regressions.

## Prioritized findings

1. **PS6093: autograd bounds checks.** The new version emits 132 such advisories
   repository-wide. In `autograd/vjp_elementwise.go`, `unaryVJP` indexes four
   slices using `x.Numel()`; `reluVJP` indexes three without a dominating
   slice-length proof. Pinned compiler diagnostics report `IsInBounds` at lines
   30/39 and 123/124/134/135. The optimized ARM64 ReLU loops contain separate
   CMP/BCS checks for the input and positive-branch gradient/output accesses,
   with targets calling `runtime.panicBounds`. A separately specified candidate
   should establish valid storage extents once and preserve dtype, layout,
   alias, special-value, gradient and invalid-input behavior. Benchmark both
   active and inactive ReLU values plus small/large public backward routes;
   assembly evidence alone does not establish leverage. Extend perfscan #904.
2. **PS6086: distillation fan-out.** `backend/cpu/distill_cpu.go:176` launches
   every chunk and waits, leaving caller participation as a candidate. There are
   14 PS6086 findings overall. Preserve the current `b*c < 1<<13` threshold,
   chunking, arithmetic and exact outputs during any experiment. Prior QMatMul
   evidence in perfscan #829 rejected the same general idea despite fewer
   allocations because decode latency regressed. Do not infer a win here.
3. **PS6095: Shampoo invariant quotient.** `nn/shampoo.go:312` repeats
   `-1.0/float64(power)` syntactically inside the eigenvalue loop. Inspect actual
   code generation before proposing a change: the compiler may already hoist
   it and the enclosing eigensolve may dominate. No compiler or timing result
   for this particular site was produced in this check.

## Findings deliberately not promoted

- PS4008 at Cholesky line 104 and Conv1D line 126 is newly reported by v1.110.0,
  but both are scalar residual loops after four-output register tiles. MoE's
  PS1010 sites at lines 92/157 are likewise at most three output columns.
  Existing exact output-interleaving evidence remains binding (perfscan #906).
  Tail size and production shape matter; the general advice is not a reason to
  undo the measured main loops or reassociate their reductions.
- Cross-entropy PS4002 sites execute a log once per row, not once per class.
  The prior scalar-log vectorization rejection is already in Spectackle.
- PS6078 calls `f32NativeKernels` and `normF32ForwardFast` architecture gaps,
  but their false definitions are default-build policies. Both have true ARM64
  SIMD siblings; the normal default keeps different accumulation semantics.
  Report build-feature partitions accurately before recommending a missing
  architecture implementation (follow-up to perfscan #795).
- PS6092's reference-backend `op.apply` sites are the generic/exotic or broadcast
  fallback paths. F32/F64 contiguous hot loops already use one-time concrete
  operation dispatch. Do not repeat the optimization recorded in perfscan #905.
- `llamagpu/decoder.go:4115` casts one model-upload row block (256 rows), not one
  token; contiguous F32 bypasses conversion. This is not a new decode allocation.
- GELU PS6077 remains covered by the existing rejected candidates and allocation
  investigation. This scan does not override those rejections.

## Reproduction and retained evidence

All commands used the pinned SDK environment:

```sh
PATH=/private/tmp/goai-go1271-j1cLvq/go/bin:$PATH
GOCACHE=/private/tmp/gocache-goai
GOMODCACHE=/private/tmp/goai-go1271-j1cLvq/modcache
GOTOOLCHAIN=local
CGO_ENABLED=0
GOPROXY=direct
```

The installed scanner is `/Users/john/go/bin/perfscan`, module v1.110.0, built
with Go 1.27.0. Its package-loading metadata reports Go 1.27.1. CI's scanner was
resolved directly with `go run github.com/jxsl13/perfscan@v1.81.0`.

`commands.json` records argv, environments, exits and artifact hashes. Compressed
files are byte-preserving gzip (`-n`) of complete raw reports, not filtered
summaries. Decompress with `gzip -dc <artifact.gz>`.

The first help attempt used ambient Go and failed a toolchain-download permission
check. The first whole-tree JSON result could not be delivered through the tool
transport; its exit and findings are unavailable, so it is not counted. The
retained scan was explicitly rerun to fresh disk artifacts. Two initial canonical
commands failed restricted-network name resolution and are retained separately
from successful direct-network retries. No failure is reported as a clean scan.

The final pinned Spectackle check has the same 136 inherited warnings and two
record-only context errors (`performance`, `testing`), with no new drift, anchor
or toolchain-load errors. This is not a globally clean Spectackle claim.
