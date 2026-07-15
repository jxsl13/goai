//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"sort"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// The prefill gap to llama.cpp (2187 vs 8389 tok/s pp128) needs a target: this profile
// times the real TinyLlama prefill per op class (sync after each op — the ~10 µs sync
// noise is negligible against the ms-scale classes at seq=128) so the fused-attention
// arc (lever b) starts from measured shares, not the assumption that attention dominates
// (§V22: the bottleneck is backend-specific; §T402: never trust standalone op timing).
func TestCUDAPrefillProfile(t *testing.T) {
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	rl := buildResidentLlama(t, model)
	defer rl.free()

	const seq = 128
	ids := make([]int32, seq)
	for i := range ids {
		ids[i] = int32((i*131 + 1) % model.Config.Vocab)
	}

	buckets := map[string]time.Duration{}
	tick := func(name string, fn func() error) {
		start := time.Now()
		must(t, fn())
		must(t, cuda.GraphSync())
		buckets[name] += time.Since(start)
	}

	run := func() {
		dx, err := rl.emb.Embed(ids)
		must(t, err)
		for _, l := range rl.layers {
			rq := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.heads}
			rk := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.kv}
			var dh, dq, dk, dv, da, dh2, dgate, dup *cuda.DeviceF32
			tick("rmsnorm", func() error { var e error; dh, e = dx.RMSNormTo(l.gAttn, float32(l.eps)); return e })
			tick("qkv-gemm", func() error {
				var e error
				if dq, e = l.wq.MatMulDevice(dh); e != nil {
					return e
				}
				if dk, e = l.wk.MatMulDevice(dh); e != nil {
					return e
				}
				dv, e = l.wv.MatMulDevice(dh)
				return e
			})
			dh.Free()
			tick("rope", func() error {
				if e := dq.RoPE(rq); e != nil {
					return e
				}
				return dk.RoPE(rk)
			})
			tick("attention", func() error {
				var e error
				da, e = cuda.GroupedQueryAttention(dq, dk, dv, l.heads, l.kv, true)
				return e
			})
			dq.Free()
			dk.Free()
			dv.Free()
			tick("o-gemm", func() error { return l.wo.MatMulAccInto(da, dx) })
			da.Free()
			tick("rmsnorm", func() error { var e error; dh2, e = dx.RMSNormTo(l.gFFN, float32(l.eps)); return e })
			tick("ffn-gemm", func() error {
				var e error
				if dgate, e = l.wg.MatMulDevice(dh2); e != nil {
					return e
				}
				dup, e = l.wu.MatMulDevice(dh2)
				return e
			})
			dh2.Free()
			tick("swiglu", func() error { return dgate.SwiGLU(dup) })
			dup.Free()
			tick("ffn-gemm", func() error { return l.wd.MatMulAccInto(dgate, dx) })
			dgate.Free()
		}
		tick("final-norm+head", func() error {
			if e := dx.RMSNorm(rl.norm, float32(rl.eps)); e != nil {
				return e
			}
			dl, e := rl.out.MatMulDevice(dx)
			if e != nil {
				return e
			}
			dl.Free()
			return nil
		})
		dx.Free()
	}

	run() // warm-up (kernel compiles, cuBLAS heuristics)
	for k := range buckets {
		delete(buckets, k)
	}
	const reps = 5
	t0 := time.Now()
	for i := 0; i < reps; i++ {
		run()
	}
	total := time.Since(t0)

	type kv struct {
		name string
		d    time.Duration
	}
	var rows []kv
	var sum time.Duration
	for k, v := range buckets {
		rows = append(rows, kv{k, v / reps})
		sum += v / reps
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].d > rows[j].d })
	t.Logf("TinyLlama prefill seq=%d, %d reps, avg total %.2f ms (walls incl. syncs)", seq, reps, float64(total.Microseconds())/1000/reps)
	for _, r := range rows {
		t.Logf("  %-16s %8.2f ms  %5.1f%%", r.name, float64(r.d.Microseconds())/1000, 100*float64(r.d)/float64(sum))
	}
}
