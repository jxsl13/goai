//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// The continuous-batch scheduler must admit pending sequences up to capacity, advance the whole
// active batch one decode step per iteration, evict finished ones (returning their blocks), and
// keep producing until every request is done.
func TestCUDABatchScheduler(t *testing.T) {
	skipNoGPU(t)
	const blockSize, hd, qHeads, kvHeads, maxActive = 16, 64, 8, 2, 4
	wkv := kvHeads * hd
	qWidth := qHeads * hd
	pool, err := cuda.NewPagedKVPool(256, blockSize, wkv)
	must(t, err)
	defer pool.Free()
	sched := cuda.NewBatchScheduler(pool, qHeads, kvHeads, maxActive)

	rng := rand.New(rand.NewSource(61))
	dev := func(rows, cols int) *cuda.DeviceF32 {
		f := make([]float32, rows*cols)
		for i := range f {
			f[i] = float32(rng.NormFloat64()) * 0.3
		}
		d, _ := cuda.NewDeviceF32(rows, cols)
		must(t, d.UploadF32(f))
		return d
	}
	// 6 requests with different decode budgets; each seeded with a short prompt.
	freeAll := pool.FreeBlocks() // full pool before any allocation
	budgets := []int{3, 7, 2, 5, 4, 6}
	want := 0
	for _, steps := range budgets {
		s := pool.NewSeqKV()
		prompt := dev(8, wkv)
		must(t, s.Append(prompt, prompt))
		prompt.Free()
		sched.Enqueue(s, steps)
		want += steps
	}

	totalProduced := 0
	for step := 0; sched.ActiveLen() > 0 || sched.Pending() > 0; step++ {
		sched.Admit()
		na := sched.ActiveLen()
		if na == 0 {
			break
		}
		q := dev(na, qWidth)
		nk := make([]*cuda.DeviceF32, na)
		nv := make([]*cuda.DeviceF32, na)
		for i := 0; i < na; i++ {
			nk[i] = dev(1, wkv)
			nv[i] = dev(1, wkv)
		}
		out, err := sched.Step(q, nk, nv)
		must(t, err)
		if out != nil {
			totalProduced += out.Rows()
			out.Free()
		}
		q.Free()
		for i := 0; i < na; i++ {
			nk[i].Free()
			nv[i].Free()
		}
		if step > 100 {
			t.Fatal("scheduler did not drain")
		}
	}
	t.Logf("scheduler drained: produced %d attention rows over the batch (want ~%d total tokens)", totalProduced, want)
	if got := pool.FreeBlocks(); got != freeAll {
		t.Fatalf("blocks not fully returned: free %d, full pool %d", got, freeAll)
	}
	t.Logf("all blocks returned (free %d)", pool.FreeBlocks())
}

// Aggregate decode throughput scaling: the batched paged attention at batch=1/16/64, fixed
// cached length. This is the continuous-batching lever — one kernel launch serves the whole
// batch, so aggregate tok/s should rise steeply with batch size (vs a single-stream engine
// whose aggregate == its single-sequence rate). Target: vLLM's ~4655 tok/s at n=64.
func benchBatchedDecode(b *testing.B, batch, seqLen int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const blockSize, hd, qHeads, kvHeads = 16, 64, 8, 2
	wkv := kvHeads * hd
	qWidth := qHeads * hd
	needBlocks := batch * ((seqLen + blockSize - 1) / blockSize)
	pool, err := cuda.NewPagedKVPool(needBlocks+batch, blockSize, wkv)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Free()
	rng := rand.New(rand.NewSource(7))
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		f := make([]float32, seqLen*wkv)
		for j := range f {
			f[j] = float32(rng.NormFloat64()) * 0.3
		}
		d, _ := cuda.NewDeviceF32(seqLen, wkv)
		d.UploadF32(f)
		seqs[i].Append(d, d)
		d.Free()
	}
	qf := make([]float32, batch*qWidth)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.3
	}
	q, _ := cuda.NewDeviceF32(batch, qWidth)
	q.UploadF32(qf)
	defer q.Free()
	o, err := pool.BatchedDecodeAttn(q, seqs, qHeads, kvHeads)
	if err != nil {
		b.Fatal(err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		out, err := pool.BatchedDecodeAttn(q, seqs, qHeads, kvHeads)
		if err != nil {
			b.Fatal(err)
		}
		out.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
	// aggregate tokens/s = batch tokens per launch / per-launch time
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkBatchedDecode_b1_s128(b *testing.B)  { benchBatchedDecode(b, 1, 128) }
func BenchmarkBatchedDecode_b16_s128(b *testing.B) { benchBatchedDecode(b, 16, 128) }
func BenchmarkBatchedDecode_b64_s128(b *testing.B) { benchBatchedDecode(b, 64, 128) }
