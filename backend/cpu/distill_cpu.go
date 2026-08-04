package cpu

import (
	"math"
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// distillKernelCPU is the knowledge-distillation KL loss with (a) both softmax exp passes
// vectorised via simd.ExpSumF64 and (b) the two per-element math.Log eliminated: since
// log p[j] = zt[j]/T − mt − log Σt and log q[j] = zs[j]/T − ms − log Σs, the KL term
// collapses to p[j]·((zt[j]−zs[j])/T + C) with a per-ROW constant C = (ms−mt)+(logΣs−logΣt)
// — 2 logs/row instead of 2c. Tolerance-gated (ADR-0021 f64 vexp), gated by vexpF64Fast;
// F32 / non-rank-2 / bad temp fall back to the reference kernel.
func distillKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && vexpF64Fast &&
		in[0].Dtype() == tensor.F64 && in[1].Dtype() == tensor.F64 &&
		in[0].Ndim() == 2 && in[1].Ndim() == 2 {
		zs, zt := in[0], in[1]
		b, c := zs.Shape()[0], zs.Shape()[1]
		pa, _ := attrs.(backend.DistillAttrs)
		pa = pa.WithDefaults()
		temp := pa.Temperature
		if b > 0 && zt.Shape()[0] == b && zt.Shape()[1] == c && temp > 0 {
			ss := zs.Contiguous().Storage().F64()
			ts := zt.Contiguous().Storage().F64()
			invT := 1 / temp
			p := make([]float64, c)
			qs := make([]float64, c)
			tsc := make([]float64, c)
			ssc := make([]float64, c)
			var total float64
			for i := 0; i < b; i++ {
				zti := ts[i*c : i*c+c]
				zsi := ss[i*c : i*c+c]
				mt := math.Inf(-1)
				for j, z := range zti {
					v := z * invT
					tsc[j] = v
					if v > mt {
						mt = v
					}
				}
				sumT := simd.ExpSumF64(p, tsc, mt) // p[j] = exp(zt[j]/T − mt)
				ms := math.Inf(-1)
				for j, z := range zsi {
					v := z * invT
					ssc[j] = v
					if v > ms {
						ms = v
					}
				}
				sumS := simd.ExpSumF64(qs, ssc, ms) // Σs only (qs is scratch)
				invSumT := 1 / sumT
				cc := (ms - mt) + (math.Log(sumS) - math.Log(sumT))
				var kl float64
				for j := 0; j < c; j++ {
					if pj := p[j] * invSumT; pj > 0 {
						kl += pj * ((zti[j]-zsi[j])*invT + cc)
					}
				}
				total += temp * temp * kl
			}
			out := tensor.NewOn(ctx.Device(), zs.Dtype(), tensor.Shape{})
			out.SetF64(total / float64(b))
			return []*tensor.Tensor{out}, nil
		}
	}
	// F32 fast path: close the dtype-gap (OpDistill was cpu-registered F64-only, so F32
	// fell to backend/ref's serial scan). The rows are independent KL contributions, so we
	// compute each row's temp²·kl in parallel over disjoint row blocks and sum them back in
	// index order — BYTE-IDENTICAL to the serial ref F32 path (which widens F32→F64 up front
	// and runs the same z/temp softmax + log-KL sequence per row). We read F32 directly and
	// widen per element, skipping ref's b·c F64 materialization.
	if len(in) == 2 &&
		in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 &&
		in[0].Ndim() == 2 && in[1].Ndim() == 2 {
		zs, zt := in[0], in[1]
		b, c := zs.Shape()[0], zs.Shape()[1]
		pa, _ := attrs.(backend.DistillAttrs)
		pa = pa.WithDefaults()
		temp := pa.Temperature
		if b > 0 && zt.Shape()[0] == b && zt.Shape()[1] == c && temp > 0 {
			ss := zs.Contiguous().Storage().F32()
			ts := zt.Contiguous().Storage().F32()
			contrib := make([]float64, b)
			distillRowsF32(ss, ts, contrib, b, c, temp)
			var total float64
			//perfscan:ignore PS3010 already simd+typed log-elim fast path (shipped distill win)
			for i := 0; i < b; i++ { // serial ordered sum ≡ ref's total += ...
				total += contrib[i]
			}
			out := tensor.NewOn(ctx.Device(), zs.Dtype(), tensor.Shape{})
			out.SetF64(total / float64(b))
			return []*tensor.Tensor{out}, nil
		}
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpDistill, in, attrs)
}

// distillRowScanF32 fills contrib[lo:hi] with each row's temp²·KL(teacher‖student),
// mirroring backend/ref's F32 path exactly: F32 reads widened to F64, the stable softmax
// uses z/temp (division, not ×invT) so the bits match ref's softmaxRowFlat, and the KL is
// Σ p·(log p − log q) over p>0. p/q scratch is per-invocation (per worker) so no sharing.
func distillRowScanF32(ss, ts []float32, contrib []float64, c int, temp float64, lo, hi int) {
	p := make([]float64, c)
	q := make([]float64, c)
	for i := lo; i < hi; i++ {
		zti := ts[i*c : i*c+c]
		zsi := ss[i*c : i*c+c]
		// teacher soft targets p
		mt := math.Inf(-1)
		for _, z := range zti {
			if v := float64(z) / temp; v > mt {
				mt = v
			}
		}
		var sumT float64
		for j, z := range zti {
			e := math.Exp(float64(z)/temp - mt)
			p[j] = e
			sumT += e
		}
		for j := range p {
			p[j] /= sumT
		}
		// student soft distribution q
		ms := math.Inf(-1)
		for _, z := range zsi {
			if v := float64(z) / temp; v > ms {
				ms = v
			}
		}
		var sumS float64
		for j, z := range zsi {
			e := math.Exp(float64(z)/temp - ms)
			q[j] = e
			sumS += e
		}
		for j := range q {
			q[j] /= sumS
		}
		var kl float64
		for j := 0; j < c; j++ {
			if p[j] > 0 {
				kl += p[j] * (math.Log(p[j]) - math.Log(q[j]))
			}
		}
		contrib[i] = temp * temp * kl
	}
}

// distillRowsF32 runs distillRowScanF32 across GOMAXPROCS goroutines over disjoint row
// blocks. Rows are independent (each writes its own contrib[i]) so the result is identical
// to the serial scan regardless of scheduling; the KL is transcendental-heavy (c exp + 2c
// log per row) → compute-bound → parallelizes near-linearly. Small work stays serial.
func distillRowsF32(ss, ts []float32, contrib []float64, b, c int, temp float64) {
	nw := runtime.GOMAXPROCS(0)
	//perfscan:ignore PS3011 declined-dtype reference fallback; stale line past EOF
	chunk := (b + nw - 1) / nw
	if nw <= 1 || chunk >= b || b*c < 1<<13 {
		distillRowScanF32(ss, ts, contrib, c, temp, 0, b)
		return
	}
	var wg sync.WaitGroup
	for lo := 0; lo < b; lo += chunk {
		hi := lo + chunk
		if hi > b {
			hi = b
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			distillRowScanF32(ss, ts, contrib, c, temp, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

func init() {
	std.add(backend.OpDistill, tensor.F64, distillKernelCPU)
	std.add(backend.OpDistill, tensor.F32, distillKernelCPU)
}
