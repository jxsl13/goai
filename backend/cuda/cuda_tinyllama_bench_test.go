//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type residentLlama struct {
	emb, out   *cuda.ResidentB
	norm       *cuda.ResidentVec
	layers     []*resLayer
	dim, vocab int
	eps        float64
}

func buildResidentLlama(tb testing.TB, m *nlp.Llama) *residentLlama {
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	rl := &residentLlama{dim: cfg.Dim, vocab: cfg.Vocab, eps: cfg.Eps}
	rl.emb, _ = cuda.NewResidentB(cast(m.TokEmb))
	rl.norm, _ = cuda.NewResidentVec(cast(m.Norm.Gamma))
	rl.out, _ = cuda.NewResidentB(cast(m.Out))
	rl.layers = make([]*resLayer, len(m.Blocks))
	for i, blk := range m.Blocks {
		ga, _ := cuda.NewResidentVec(cast(blk.AttnNorm.Gamma))
		gf, _ := cuda.NewResidentVec(cast(blk.FFNNorm.Gamma))
		mk := func(w *tensor.Tensor) *cuda.ResidentB { r, _ := cuda.NewResidentB(cast(w)); return r }
		rl.layers[i] = &resLayer{
			gAttn: ga, gFFN: gf,
			wq: mk(blk.Wq), wk: mk(blk.Wk), wv: mk(blk.Wv), wo: mk(blk.Wo),
			wg: mk(blk.FFN.Wgate), wu: mk(blk.FFN.Wup), wd: mk(blk.FFN.Wdown),
			heads: cfg.Heads, kv: kv, dim: cfg.Dim, hidden: cfg.Hidden,
			ropeBase: cfg.RopeBase, eps: cfg.Eps,
		}
	}
	return rl
}

func (rl *residentLlama) free() {
	rl.emb.Free()
	rl.norm.Free()
	rl.out.Free()
	for _, l := range rl.layers {
		l.free()
	}
}

// forward runs embed → layers → final norm → output and returns host logits.
func (rl *residentLlama) forward(tb testing.TB, ids []int32) *tensor.Tensor {
	dx, err := rl.emb.Embed(ids)
	if err != nil {
		tb.Fatal(err)
	}
	for _, l := range rl.layers {
		if dx, err = l.forward(dx); err != nil {
			tb.Fatal(err)
		}
	}
	if err := dx.RMSNorm(rl.norm, float32(rl.eps)); err != nil {
		tb.Fatal(err)
	}
	dl, err := rl.out.MatMulDevice(dx)
	if err != nil {
		tb.Fatal(err)
	}
	dx.Free()
	out, err := dl.ToHost()
	if err != nil {
		tb.Fatal(err)
	}
	dl.Free()
	return out
}

func loadTinyLlama(tb testing.TB) *nlp.Llama {
	if !cuda.Available() {
		tb.Skip("no gpu")
	}
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		tb.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		tb.Fatal(err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

// Prefill throughput: forward a prompt of `seq` tokens, tokens/s = seq / t.
func benchPrefill(b *testing.B, seq int) {
	m := loadTinyLlama(b)
	rl := buildResidentLlama(b, m)
	defer rl.free()
	ids := make([]int32, seq)
	for i := range ids {
		ids[i] = int32((i*131 + 1) % m.Config.Vocab)
	}
	b.ResetTimer()
	for range b.N {
		rl.forward(b, ids)
	}
	b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaPrefillGPU_32(b *testing.B)  { benchPrefill(b, 32) }
func BenchmarkTinyLlamaPrefillGPU_128(b *testing.B) { benchPrefill(b, 128) }

// CPU baseline (goai nlp.Llama forward on the optimized cpu backend).
func benchPrefillCPU(b *testing.B, seq int) {
	m := loadTinyLlama(b)
	cpu, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(cpu)
	toks := make([]int, seq)
	for i := range toks {
		toks[i] = (i*131 + 1) % m.Config.Vocab
	}
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(ctx, toks); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaPrefillCPU_32(b *testing.B) { benchPrefillCPU(b, 32) }
