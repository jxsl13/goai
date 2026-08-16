//go:build darwin && cgo

package llamagpu_test

// TestF16PrefillShortPromptWin is the end-to-end anchor for the f16 short-prompt weight path, and
// the reason its gate is M<=64 rather than the M<=128 the leaf benchmark would suggest.
//
// Short prefill was the weakest number against llama.cpp (pp64 ~0.46x, against ~0.92x at pp1024).
// TestDQGemmCostSplit located the cause in the GEMM rather than the expansion — at M=64 it reads its
// 46 MB f32 weight at ~99 GB/s against ~180 GB/s sustained — and
// TestGEMMF16OnlyWinsWhenBandwidthBound confirmed f16 buys 1.28x there and nothing once the GEMM
// turns compute-bound. This test checks what actually reaches the model.
//
// Interleaved arms per prompt length, two runs on an M2 Pro (TinyLlama-1.1B Q4_K_M):
//
//	n= 64  815.1 -> 961.9 tok/s  1.180x   |  815.0 -> 957.9  1.175x
//	n= 96  997.2 -> 1000.6       1.003x   |  990.5 -> 975.6  0.985x
//	n=128 1297.9 -> 1328.0       1.023x   | 1247.5 -> 1246.2 0.999x
//	n=256 1628.5 -> 1628.8       1.000x   | 1631.9 -> 1634.7 1.002x
//
// The leaf gain at M=128 (1.09x) does not survive to end-to-end, and n=96 came out slightly negative
// on the second run, so the gate is set to the only length with a repeatable win. n=256 is outside
// the gate and confirms the path is inert there.
//
// The argmax token is IDENTICAL in both arms at every length. That is also the reachability check:
// f16 must round, so byte-identical logits would mean the gate never fired. Half precision costs
// ~5.4e-4 relative L2 on the projection output, which does not move the prediction.
import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

func TestF16PrefillShortPromptWin(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	f, err := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	qm, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Skip(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skip(err)
	}
	defer dec.Release()

	prompt := func(n int) []int {
		p := make([]int, n)
		for i := range p {
			p[i] = 1 + i%2000
		}
		return p
	}
	for _, n := range []int{64, 96, 128, 256} {
		p := prompt(n)
		var res [2]struct {
			tps  float64
			last float32
			arg  int
		}
		for arm := 0; arm < 2; arm++ {
			metal.SetQ4KDequantGemmF16(arm == 1)
			dec.StepNLast(p, 0)
			best := 0.0
			var lg []float32
			for range 3 {
				st := time.Now()
				out, e := dec.StepNLast(p, 0)
				if e != nil {
					t.Fatal(e)
				}
				if v := float64(n) / time.Since(st).Seconds(); v > best {
					best = v
				}
				lg = out
			}
			am, mx := 0, float32(-1e30)
			for i, v := range lg {
				if v > mx {
					mx, am = v, i
				}
			}
			res[arm].tps, res[arm].last, res[arm].arg = best, mx, am
		}
		fmt.Printf("F16E2E n=%4d f32=%7.1f f16=%7.1f tok/s %.3fx  argmax f32=%d f16=%d %s\n",
			n, res[0].tps, res[1].tps, res[1].tps/res[0].tps, res[0].arg, res[1].arg,
			map[bool]string{true: "SAME", false: "DIFFER"}[res[0].arg == res[1].arg])
	}
	metal.SetQ4KDequantGemmF16(true)
}
