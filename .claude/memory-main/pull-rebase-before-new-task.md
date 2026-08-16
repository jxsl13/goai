---
name: pull-rebase-before-new-task
description: "User directive — sync with origin/main (pull --rebase or --ff) BEFORE starting a new task, not only before pushing"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 73f72698-b352-4a1d-a7bc-9aca1549b901
  modified: 2026-08-14T23:44:45.395Z
---

Sync the branch against `origin/main` — `git pull --rebase` or `git pull --ff` — **before starting each new task**, not merely before each push.

**Why:** the user works in parallel on backend/cuda-amd64 branches, and main moves under this branch continuously. Starting work from a stale base is what produces the expensive failures: on 2026-07-28 a session's whole branch drifted far enough that PR #394 went CONFLICTING across 15 files, including a genuine **ID collision** — both sides had independently minted `PS6001` in `internal/perfscan` for entirely different checks. Rebasing before starting would have surfaced main's PS6001 while the new rule was still being named, when renumbering costs nothing. Resolved after the fact it cost a rename across the registry, detector, 13 tests, PATTERNS.md and a live suppression in `tensor/tensor.go`.

**How to apply:**
- First action of a new task: fetch and rebase/ff onto `origin/main`. Do it even when the previous task just pushed — main may have advanced in between.
- **Verify the base is actually current — never trust the session's "Recent commits" snapshot.** On 2026-08-15 local `main` sat at `a2f27462` (2026-07-22) while `origin/main` was **907 commits ahead**, having meanwhile deleted `SPEC.md` and the whole `spec/` fragment tree in favour of tracked `.spectackle/`. A full session's work (T984–T998) was built on that 3-week-stale base and had to be re-landed piecemeal. `git rev-list --count <base>..origin/main` and `git merge-base --is-ancestor` answer this in one second.
- **A failing `git status` is a work-stranding emergency, not a nuisance.** The same session found `core.bare=true` set in `.git/config` of a repo that plainly had a work tree (a stray worktree-command side effect). Git then refused every status/commit/checkout with "this operation must be run in a work tree", so ~1500 lines of finished work silently accumulated uncommitted for a day, and the affected tasks recorded the symptom without diagnosing it ("cannot archive because this workspace has no Git worktree metadata", "the local bare Git repository cannot pin the new GoAI source"). Fix is `git config --local core.bare false`; then classify every modified file as stale-vs-local by comparing `git hash-object` against the blob in each commit that touched it, *before* restoring anything — files absent from disk may be intentional deletions, not missing checkouts.
- When a stale-base body of work must be salvaged, split it: push the whole thing to a preservation branch first so nothing is lost, then extract the subsets that merge cleanly onto current main as focused PRs. `git merge-tree --write-tree --name-only origin/main <branch>` lists the conflicted paths without touching the tree, so the cleanly-appliable subset is identifiable up front (here `backend/metal/` was conflict-free and became PR #1061 on its own).
- Before minting any new identifier that must be globally unique (perfscan `PSxxxx`, spec rule stems, task IDs), check it against **main and every open PR**, not just the local branch. This is the existing `PERF-ID-COLLISION-001` rule; the collision above happened because it was skipped.
- A merge must never silently revert main. When both sides changed the same code, diff each side against the merge base and confirm which is a superset before choosing — and if main carries an orthogonal optimization the branch lacks, take main's and book the remainder as measurable follow-up rather than hand-composing the two in a hot path.
- Verify a pre-existing test failure really is pre-existing by running it in a clean worktree at `origin/main`, rather than inferring it from commit history.

Related: [[goai-autonomous-loop]] (per-push discipline: fetch-rebase, §V27 selector validation, CI watch), [[worktree-agent-stale-base]] (the same staleness problem for delegated subagents).
