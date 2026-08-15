//go:build darwin && cgo

package llamagpu_test

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

func TestZZPP2(t *testing.T) {
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
	for _, n := range []int{64, 128} {
		long := make([]int, n)
		for i := range long {
			long[i] = 1 + i%2000
		}
		dec.StepNLast(long, 0)
		bw, bg := 1e18, 1e18
		for range 5 {
			st := time.Now()
			dec.StepNLast(long, 0)
			w := time.Since(st).Seconds()
			g := metal.LastGPUSeconds()
			if w < bw {
				bw = w
			}
			if g < bg {
				bg = g
			}
		}
		fmt.Printf("PP2 n=%3d  wall=%6.2f ms  gpuBusy=%6.2f ms  gpu%%=%4.1f  host=%5.2f ms  perLayerGPU=%.3f ms  %.0f tok/s\n",
			n, bw*1000, bg*1000, 100*bg/bw, (bw-bg)*1000, bg*1000/22, float64(n)/bw)
	}
}
