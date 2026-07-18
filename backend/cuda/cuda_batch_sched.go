//go:build cuda && cgo && (linux || windows)

package cuda

import "fmt"

// §ROADMAP FRONT B / B3 — iteration-level continuous-batch scheduler over the paged KV pool.
// This is the mechanism vLLM/SGLang use to turn a single-stream engine into a high-throughput
// server: a running batch of sequences all advance ONE decode step per iteration, sequences
// join (admit) and leave (evict) the batch every step (not per-request), and the paged pool
// (B1) + batched paged attention kernel (B2) let the batch grow to fill VRAM with near-zero
// fragmentation. This slice provides the scheduler mechanics + drives the batched attention;
// the full batched model forward (embed -> layers -> sample) is B4.

// genSeq is one in-flight generation: its paged K/V and how many decode steps remain.
type genSeq struct {
	kv        *SeqKV
	remaining int // decode steps left before the request finishes
	produced  int
}

// BatchScheduler runs a continuous batch over a PagedKVPool.
type BatchScheduler struct {
	pool            *PagedKVPool
	qHeads, kvHeads int
	active          []*genSeq
	pending         []*genSeq
	maxActive       int
}

// NewBatchScheduler creates a scheduler bounded to maxActive concurrent sequences.
func NewBatchScheduler(pool *PagedKVPool, qHeads, kvHeads, maxActive int) *BatchScheduler {
	return &BatchScheduler{pool: pool, qHeads: qHeads, kvHeads: kvHeads, maxActive: maxActive}
}

// Enqueue registers a new request: a sequence (already seeded with its prompt K/V via SeqKV)
// that should decode `steps` more tokens.
func (bs *BatchScheduler) Enqueue(kv *SeqKV, steps int) {
	bs.pending = append(bs.pending, &genSeq{kv: kv, remaining: steps})
}

// Admit moves pending requests into the active batch up to maxActive. Call it at the top of
// each iteration BEFORE sizing the step inputs to ActiveLen (the continuous-batching loop shape).
func (bs *BatchScheduler) Admit() {
	for len(bs.active) < bs.maxActive && len(bs.pending) > 0 {
		bs.active = append(bs.active, bs.pending[0])
		bs.pending = bs.pending[1:]
	}
}

// reap evicts finished sequences (remaining==0), returning their blocks to the pool — the
// per-step eviction continuous batching needs so freed VRAM immediately admits new requests.
func (bs *BatchScheduler) reap() {
	kept := bs.active[:0]
	for _, g := range bs.active {
		if g.remaining <= 0 {
			g.kv.Release()
		} else {
			kept = append(kept, g)
		}
	}
	bs.active = kept
}

// ActiveLen / Pending report batch occupancy.
func (bs *BatchScheduler) ActiveLen() int { return len(bs.active) }
func (bs *BatchScheduler) Pending() int   { return len(bs.pending) }

// Step advances the whole active batch by one decode iteration (call Admit first): run the batched
// paged attention for the current query tokens `q` ([activeLen, qHeads*hd], row i = active[i]'s
// query), append each sequence's new K/V, decrement, and reap finished. Returns the attention
// output [activeLen, qHeads*hd]. `newKV` supplies per-sequence [1, kvHeads*hd] K and V for the
// token just produced (the B4 forward will compute these; here they are provided/mocked).
func (bs *BatchScheduler) Step(q *DeviceF32, newK, newV []*DeviceF32) (*DeviceF32, error) {
	if len(bs.active) == 0 {
		return nil, nil
	}
	if q.rows != len(bs.active) {
		return nil, fmt.Errorf("cuda: BatchScheduler.Step q rows %d != active %d", q.rows, len(bs.active))
	}
	if len(newK) != len(bs.active) || len(newV) != len(bs.active) {
		return nil, fmt.Errorf("cuda: BatchScheduler.Step newK/V len mismatch")
	}
	seqs := make([]*SeqKV, len(bs.active))
	for i, g := range bs.active {
		seqs[i] = g.kv
	}
	out, err := bs.pool.BatchedDecodeAttn(q, seqs, bs.qHeads, bs.kvHeads)
	if err != nil {
		return nil, err
	}
	for i, g := range bs.active {
		if err := g.kv.Append(newK[i], newV[i]); err != nil {
			out.Free()
			return nil, err
		}
		g.remaining--
		g.produced++
	}
	bs.reap()
	return out, nil
}
