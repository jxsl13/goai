//go:build darwin && cgo

package llamagpu_test

// TestNarrowProjectionGate covers the N threshold on the f16 expand-then-GEMM path, and why it has
// to depend on M rather than being a constant.
//
// TinyLlama's k and v projections are N=256, under the N>=512 gate, so they fell to the cooperative
// quantized kernel. That kernel re-reads the weight per row-group, which is fine at decode (M=1) and
// expensive at prefill batch sizes. Once the weight cache removed the expansion cost, there was no
// longer an obvious reason to keep them off the GEMM path — but measuring found the answer depends
// on the batch. Interleaved arms (N>=512 vs N>=128), two runs, cache on:
//
//	n= 64  1510.1 -> 1482.7  0.982x  |  1499.9 -> 1484.5  0.990x
//	n=256  1850.4 -> 1924.7  1.040x  |  1848.0 -> 1931.6  1.045x
//	n=512  1906.8 -> 1990.9  1.044x  |  1896.5 -> 1981.2  1.045x
//
// A consistent sign flip, not noise: four readings at 1.040-1.045 do not scatter the way this
// machine's ~4% noise does, and n=64 is reproducibly on the other side of 1.0. At M=64 a 256-wide
// GEMM plus its activation/result conversions costs more than the cooperative kernel it replaces;
// by M=256 the cooperative kernel's per-row weight re-read dominates and the GEMM wins.
//
// So the threshold is 128 for M>=128 and stays 512 below that. With that gate:
//
//	n= 64  1460.3 -> 1461.2  1.001x  |  1468.5 -> 1479.2  1.007x
//	n=256  1839.1 -> 1917.2  1.043x  |  1832.3 -> 1903.5  1.039x
//	n=512  1872.5 -> 1991.6  1.064x  |  1883.9 -> 1982.0  1.052x
//
// n=64 is returned to neutral while the gains at 256 and 512 are kept. The reported cache size is
// the reachability check: 1.92 GB at n=64 where k/v stay off the path, 1.94 GB at n>=256 where the
// two extra tensors per layer join it.
//
// The argmax token is identical in both arms at every length.
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

func TestNarrowProjectionGate(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	f, _ := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if f == nil {
		t.Skip("no model")
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	qm, _ := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skip(err)
	}
	defer dec.Release()
	metal.SetWeightCacheGB(4.0)
	for _, n := range []int{64, 256, 512} {
		var tps [2]float64
		var arg [2]int
		for arm, minN := range []int{512, 128} { // arm0 = old behaviour, arm1 = M-dependent
			metal.SetF16MinN(minN)
			p := make([]int, n)
			for j := range p {
				p[j] = 1 + j%2000
			}
			dec.StepNLast(p, 0)
			best, am := 0.0, 0
			for range 3 {
				st := time.Now()
				out, _ := dec.StepNLast(p, 0)
				if v := float64(n) / time.Since(st).Seconds(); v > best {
					best = v
				}
				mx := float32(-1e30)
				for i, v := range out {
					if v > mx {
						mx, am = v, i
					}
				}
			}
			tps[arm], arg[arm] = best, am
		}
		_, _, b := metal.WeightCacheStats()
		if arg[0] != arg[1] {
			t.Errorf("n=%d: narrow-projection gate changed the predicted token (%d vs %d)", n, arg[0], arg[1])
		}
		fmt.Printf("NPROBE n=%4d N>=512 %7.1f  N>=128 %7.1f  %.3fx  argmax %d/%d %s  %.2fGB\n",
			n, tps[0], tps[1], tps[1]/tps[0], arg[0], arg[1],
			map[bool]string{true: "SAME", false: "DIFFER"}[arg[0] == arg[1]], b/1e9)
	}
	metal.SetF16MinN(512)
	metal.SetWeightCacheGB(0)
}
