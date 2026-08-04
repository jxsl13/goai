// Package cpu is the optimized Pure-Go CPU backend (layer L1b). It computes the
// same results as backend/ref (the truth, §V9) but fast: contiguous typed-slice
// loops with no per-element allocation, goroutine parallelism above a threshold,
// and SIMD-class primitives from internal/simd (amd64 archsimd override in §T11b,
// ADR-0005). It registers as the preferred Default; ref stays the fallback (§I4).
package cpu

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type kernelKey struct {
	op    backend.Op
	dtype tensor.Dtype
}

// Backend is the optimized CPU backend. Synchronous, so Synchronize is a no-op.
type Backend struct {
	table map[kernelKey]backend.Kernel
}

var std = &Backend{table: make(map[kernelKey]backend.Kernel)}

func init() { backend.RegisterDefault(std) }

func (b *Backend) add(op backend.Op, dtype tensor.Dtype, k backend.Kernel) {
	//perfscan:ignore PS3004 kernel-table registration; one-time init, cold
	b.table[kernelKey{op, dtype}] = k
}

func (b *Backend) Name() backend.Name    { return backend.CPU }
func (b *Backend) Device() tensor.Device { return tensor.CPU() }
func (b *Backend) Synchronize() error    { return nil }
func (b *Backend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	//perfscan:ignore PS3004 one map lookup per whole-op dispatch; negligible vs kernel
	k, ok := b.table[kernelKey{op, dtype}]
	return k, ok
}

// parThreshold is the total work (items × work-per-item) below which parallelism is not
// worth the overhead of handing chunks to the worker pool. Boundary behavior: set it too
// LOW and small ops pay pool-dispatch + cache-sync cost that exceeds the compute they save
// (a net slowdown on the many tiny elementwise/norm calls a decode step makes); too HIGH and
// medium-sized ops that would benefit stay serial. 1<<15 (32768) is the measured crossover on
// M-series cores (§T511 pool design, §V22 A/B) — the point where GOMAXPROCS-way splitting
// starts to win. Not a user knob (internal, §C13-exempt); revisit via benchmarks (§V5) if the
// pool dispatch cost changes.
//
//perfscan:ignore PS6023 tuned const declaration, not a loop
const parThreshold = 1 << 15 // 32768

// parWorkPerWorker is the minimum share of the total work a band must carry for its worker to be
// worth waking. The worker count is total/parWorkPerWorker, capped at GOMAXPROCS, so an op grows
// its fan-out with its size instead of jumping straight to full width the moment it clears
// parThreshold. See the sweep recorded at the use site.
const parWorkPerWorker = 1 << 15

// poolBarrier is the completion state of one parallelWork call. pending is the
// countdown the CALLER spin-waits on (§V22: parking in wg.Wait cost a full
// M-stop/restart — pthread_cond_wait + _signal + scheduler steal churn — on
// every one of the ~16k barriers a 500-token decode issues); wg is the blocking
// fallback for barriers that outlive the caller's spin budget (big training
// ops), where one park is amortized over ms of compute. Workers decrement
// pending LAST, so a caller that observes pending==0 knows both the body and
// the wg.Done have fully happened (seq-cst atomics give the happens-before).
type poolBarrier struct {
	pending atomic.Int32
	wg      sync.WaitGroup
}

// poolTask is one chunk handed to the persistent workers (§T511).
type poolTask struct {
	body   func(lo, hi int)
	lo, hi int
	b      *poolBarrier
}

// The worker pool (§T511 persistence + §V22 latency redesign). Structure:
//
//   - poolChs[w] is worker w's MAILBOX: a small buffered channel used ONLY
//     with non-blocking sends and receives. Tasks therefore always sit in the
//     channel BUFFER — never in a recvq direct handoff — so any hot worker,
//     or the caller itself, can steal a chunk whose designated worker is slow
//     to arrive. One single-consumer-ish mailbox per worker instead of one
//     shared channel: with a shared channel every submission made all spinning
//     workers stampede onto the channel's internal runtime mutex, whose
//     contention backoff is usleep — measured 40% usleep + 22% lock2 of a
//     decode profile, ~70µs per barrier even with every worker hot.
//
//   - poolPark[w] is worker w's park signal. A worker that runs out of spin
//     budget blocks HERE, never on its mailbox, so a task destined for a
//     parked worker stays stealable and the thread-wake latency stays OFF the
//     barrier's critical path: the caller signals poolPark after filling the
//     mailbox, keeps computing, and if the worker is slow to wake the chunk is
//     simply drained by a sibling or the caller. Wakes still cost the sender a
//     goready (~µs) but no longer serialize the barrier.
//
//   - Spin-before-park (dense regime only, see poolDense): a parked thread
//     costs a full OS wake to restart (measured ~100µs of pthread_cond_signal/
//     _wait per barrier on an M2 Pro — 71% of the whole
//     BenchmarkLlamaGenerate500RowBuf profile), which dwarfs the µs-scale
//     chunks a single-token decode dispatches. In the dense regime each worker
//     polls for poolSpinPolls empty receives (~200µs) before parking — enough
//     to bridge the serial glue between decode barriers (norm/rope/sampling,
//     ≲25µs each, plus per-token sampling) so back-to-back barriers run
//     signal-free after the first. The spin must NOT yield: a Gosched variant
//     melted down on sched.lock (12 spinners × ~1M yields/s → lock2+usleep
//     >50% of the profile). Yield-free is safe: the pool holds GOMAXPROCS-1
//     workers, the caller works its own chunk, so spinners never outnumber
//     free Ps, and the poll is a function call so async preemption (GC, STW)
//     works normally. In the sparse regime workers park immediately on the
//     mailbox itself (recvq handoff — the old pool's exact wake path).
//
//   - Submission is NON-blocking everywhere: when a mailbox is full (nested
//     parallelWork from inside a worker, or GOMAXPROCS grew after init) the
//     chunk runs inline on the caller, so the pool can never deadlock.
//
// Constants are not user knobs (internal, §C13-exempt); recalibrate via §V22
// benchmarks if the pool design changes.
var (
	poolChs  []chan poolTask
	poolPark []chan struct{}
)

const (
	poolSpinPolls  = 131_072 // dense-regime polls (~200µs+) before a worker parks
	poolStealEvery = 128     // polls between steal sweeps over the sibling mailboxes
)

// Dense-regime detection: the aggressive latency machinery (worker spin,
// stealing, caller help, width cap) is only worth its side effects for
// decode-like streams — many SMALL barriers separated by µs-scale serial
// glue. Everywhere else it is actively harmful, and not only as burn:
// always-spin variants measured +10-40% on warm mid-size ops and +25% on a
// 256³ gemm (continuously-burning worker threads lose their macOS scheduler
// standing — priority decay, E-core placement, no wake boost — and every
// barrier then waits on a badly-placed straggler), and sparse-mode stealing
// measured +30-50% on 11-way elementwise barriers (eleven waking workers
// sweeping each other's mailboxes serialize on the channel mutexes and
// migrate tasks away from their wakers). So an op runs in dense mode only if
//
//	its total work is decode-sized (< poolDenseMaxWork — a [1,dim]·[dim,vocab]
//	head gemv at this model scale is ~256k units, warm mid-size throughput
//	ops start at ~262k), AND
//	the serial glue since the previous barrier ENDED is short
//	(< poolDenseGapNs — decode's norm/rope/sampling glue is ≲25µs, while a
//	sparse caller leaves the pool idle much longer). Measuring end→start
//	glue rather than start→start cadence keeps the verdict independent of
//	how fast the barriers themselves run (a start→start rule locked decode
//	into sparse mode: sparse barriers are slow, so the gap never shrank).
//
// Sparse mode is exactly the old pool's discipline: full width, park on
// arrival, designated delivery, one blocking wait. §V22 measured the split:
// decode 2.05× faster in dense mode, warm grind benches at baseline in
// sparse mode.
const (
	poolDenseGapNs   = 40_000
	poolDenseMaxWork = 1 << 18
)

var (
	poolLastBarrierEnd atomic.Int64 // UnixNano when the last parallelWork returned
	poolDense          atomic.Bool  // regime of the most recent dispatch
)

// poolWideMinWork is the total work above which parallelWork fans out to the
// full machine width; below it, chunks are µs-scale and the barrier is
// latency-bound, so the fan-out is capped at 2/3 of GOMAXPROCS (see the probe
// numbers at the cap site). At ~1ns/unit this puts full width at ≳350µs-sized
// chunks, where an E-core or OS-descheduled straggler costs a fraction of the
// chunk instead of several times it.
const poolWideMinWork = 1 << 22

// The pool holds GOMAXPROCS-1 workers: parallelWork splits [0,n) into at most
// GOMAXPROCS chunks and the caller always works the first one itself, so a
// full-width barrier hands out exactly GOMAXPROCS-1 chunks and the hot steady
// state (every participant on its own core) has no spare thread to churn.
func init() {
	n := max(runtime.GOMAXPROCS(0)-1, 1)
	poolChs = make([]chan poolTask, n)
	poolPark = make([]chan struct{}, n)
	for w := range n {
		poolChs[w] = make(chan poolTask, 2)
		poolPark[w] = make(chan struct{}, 1)
	}
	for w := range n {
		go poolWorker(w)
	}
}

// poolWorker is the body of pool worker w: run tasks from the own mailbox,
// spin-poll for more, STEAL from sibling mailboxes while spinning, park on the
// signal channel when the budget runs out. Stealing restores the dynamic load
// balance a shared queue would give (a fixed chunk→worker binding made every
// barrier wait on its unluckiest designated worker — E-core or descheduled —
// measured at +24-37% on warm mid-size ops); the empty-mailbox checks on both
// the own poll and the sweep stay on the runtime's lock-free fast path.
func poolWorker(w int) {
	ch := poolChs[w]
	for {
		var t poolTask
		var run bool
		if poolDense.Load() {
			// Dense regime: spin long, steal from siblings while spinning.
		spin:
			for i := 0; i < poolSpinPolls; i++ {
				select {
				case t = <-ch:
					run = true
					break spin
				default:
				}
				if i%poolStealEvery == poolStealEvery-1 {
					for v := range poolChs {
						if v == w {
							continue
						}
						select {
						case t = <-poolChs[v]:
							run = true
							break spin
						default:
						}
					}
				}
			}
			if !run {
				<-poolPark[w] // park; a stale signal costs one more spin round
				continue
			}
		} else {
			// Sparse regime (the old pool's discipline): park directly on the
			// mailbox. A submission then wakes this worker by recvq handoff
			// with the task attached — the exact baseline wake path, one
			// goready, no signal/poll indirection. (Stealability is lost for
			// tasks delivered this way, which sparse mode does not need.)
			t = <-ch
		}
		t.body(t.lo, t.hi)
		t.b.wg.Done()
		t.b.pending.Add(-1) // last: pending==0 ⇒ body+Done complete
	}
}

// parallelWork splits [0,n) into chunks across the persistent worker pool and
// runs body on each — the caller works the first chunk itself — or runs body
// once inline when the estimated total work (n × workPerItem) is small.
// body must be safe on disjoint sub-ranges.
func parallelWork(n, workPerItem int, body func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	total := n * workPerItem
	if workers <= 1 || total < parThreshold {
		body(0, n)
		return
	}
	// Classify the regime (see poolDenseGapNs) and publish it for the workers.
	dense := total < poolDenseMaxWork &&
		time.Now().UnixNano()-poolLastBarrierEnd.Load() < poolDenseGapNs
	poolDense.Store(dense)
	// DENSE small ops must not fan out to the whole machine. Probed on the
	// M2 Pro host (12 = 8P+4E cores) with ~5µs chunks arriving back to
	// back: barriers scale to ~4.6× at 8 participants but COLLAPSE below
	// serial (0.8×) at 11-12 — once every core is claimed, each barrier waits
	// on the unluckiest thread (E-core chunks run ~2× longer, and with zero
	// spare cores the OS constantly has a participant descheduled). 2/3 of
	// GOMAXPROCS lands on the 8 P-cores here and keeps a straggler margin on
	// other hosts. The cap is NOT applied in the sparse regime: an isolated
	// mid-size op is throughput-bound, and capping it off the E cluster cost
	// a measured +20-34% on bandwidth-bound elementwise ops (Mul/AddBias)
	// whose full-width split the old pool handled fine.
	if dense {
		workers = max(workers*2/3, 2)
	}
	// EVERY WORKER MUST BE GIVEN ENOUGH WORK TO PAY FOR ITS OWN BARRIER. parThreshold decides
	// whether to fan out AT ALL; without a second gate on the work each participant receives, an
	// op only just above it is split GOMAXPROCS ways and every band is a fraction of the amount
	// that justified splitting once. A per-token decode step issues a long run of exactly those
	// ops, and the pool spends its time waking and parking: on BenchmarkLlamaPromptStepwise the
	// profile was 42% runtime.usleep, 22% cond_wait and 9% cond_signal against 14% arithmetic,
	// and the whole benchmark ran 2.3x SLOWER at 12 cores than at 1.
	if w := total / parWorkPerWorker; w < workers {
		workers = max(w, 1)
	}
	if workers <= 1 {
		body(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var b poolBarrier
	first := min(chunk, n)
	w := 0
	for lo := first; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		b.pending.Add(1)
		b.wg.Add(1)
		var sent bool
		if w < len(poolChs) {
			select {
			case poolChs[w] <- poolTask{body: body, lo: lo, hi: hi, b: &b}:
				sent = true
				select { // wake the worker if it parked; a hot one ignores this
				case poolPark[w] <- struct{}{}:
				default:
				}
			default: // that worker's mailbox is saturated (nested call): run inline
			}
			w++
		}
		if !sent {
			body(lo, hi)
			b.wg.Done()
			b.pending.Add(-1)
		}
	}
	body(0, first)
	// Spin-wait for the workers (§V22): a decode-sized chunk finishes within a
	// few µs of the caller's own, and parking in wg.Wait for that tail costs an
	// M-stop/restart per barrier. While waiting, HELP: drain unclaimed tasks
	// from the mailboxes and run them here — if a chunk landed on a parked or
	// descheduled worker, the caller executes it instead of stalling on the
	// thread wake (the cold-pool small-op case degrades to ~serial execution
	// with no waking at all). Barriers that outlive the spin budget (big
	// training ops) fall back to the blocking wait, whose one park is noise.
	if dense {
		// Dense wait: spin (parking would dominate the µs-scale tail) and
		// HELP — drain unclaimed tasks from the mailboxes and run them here,
		// so a chunk destined for a parked or descheduled worker never puts a
		// thread wake on the barrier's critical path.
		for i := 0; i < poolSpinPolls && b.pending.Load() != 0; i++ {
			if i%poolStealEvery == poolStealEvery-1 {
				for _, ch := range poolChs {
					select {
					case t := <-ch:
						t.body(t.lo, t.hi)
						t.b.wg.Done()
						t.b.pending.Add(-1)
					default:
					}
				}
			}
		}
	}
	if b.pending.Load() != 0 {
		// One drain sweep before blocking: any still-UNCLAIMED task runs here
		// (among them a nested call's chunk that landed in this very worker's
		// own mailbox, which would otherwise deadlock the blocking wait);
		// every claimed task is executing and will Done() on its own.
		for _, ch := range poolChs {
			select {
			case t := <-ch:
				t.body(t.lo, t.hi)
				t.b.wg.Done()
				t.b.pending.Add(-1)
			default:
			}
		}
		b.wg.Wait() // one park, amortized by the chunk length
	}
	poolLastBarrierEnd.Store(time.Now().UnixNano())
}

// parallel splits [0,n) with unit work per item (elementwise use).
func parallel(n int, body func(lo, hi int)) { parallelWork(n, 1, body) }
