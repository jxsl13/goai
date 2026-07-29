package autograd

import (
	"math"
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Knowledge-distillation VJP (Hinton et al. 2015). With p = softmax(z_t/T),
// q = softmax(z_s/T) and loss = mean_b T²·KL(p‖q), the gradient of the soft
// cross-entropy w.r.t. the student logits is (1/T)(q−p); the T² prefactor makes
// it T·(q−p), so the soft loss keeps a magnitude comparable to the hard loss
// across temperatures (paper §2, §R51):
//
//	∂loss/∂z_sᵢ = (g/B)·T·(qᵢ − pᵢ)
//
// The teacher logits are frozen → nil gradient.
func init() {
	RegisterVJP(backend.OpDistill, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		zs, zt := in[0], in[1]
		pX, _ := attrs.(backend.DistillAttrs)
		temp := pX.WithDefaults().Temperature
		b, c := zs.Shape()[0], zs.Shape()[1]
		gv := g.AtF64()
		scale := gv * temp / float64(b)

		gs := tensor.New(zs.Dtype(), zs.Shape())
		// Fast paths: softmaxRowT re-read each logit through z.AtF64 twice (max pass +
		// exp pass) and the result was written back cell-by-cell via gs.SetF64 —
		// interface dispatch dominating an O(b·c) kernel. When teacher and student
		// share a contiguous dtype, walk the row slices directly. Bit-identical: the
		// same v/temp, max, exp(·−m), sum and division, in the same j order.
		switch {
		case zs.Dtype() == tensor.F64 && zt.Dtype() == tensor.F64:
			zss := zs.Contiguous().Storage().F64()
			zts := zt.Contiguous().Storage().F64()
			ds := gs.Storage().F64()
			// Rows are independent (each writes only ds[i*c:i*c+c] and reads frozen
			// zss/zts), so split the batch across workers. p/q are PRIVATIZED per chunk —
			// pure per-row scratch, fully overwritten before read, never shared across rows.
			// Bit-identical: each row's softmax + scale·(q−p) uses the same operands in the
			// same j order regardless of which goroutine runs it.
			distillParallelRows(b, c, func(rlo, rhi int) {
				p := make([]float64, c)
				q := make([]float64, c)
				for i := rlo; i < rhi; i++ {
					base := i * c
					softmaxRowTInto(p, zts[base:base+c], temp)
					softmaxRowTInto(q, zss[base:base+c], temp)
					for j := 0; j < c; j++ {
						ds[base+j] = scale * (q[j] - p[j])
					}
				}
			})
			return []*tensor.Tensor{gs, nil}, nil
		case zs.Dtype() == tensor.F32 && zt.Dtype() == tensor.F32:
			zss := zs.Contiguous().Storage().F32()
			zts := zt.Contiguous().Storage().F32()
			ds := gs.Storage().F32()
			distillParallelRows(b, c, func(rlo, rhi int) {
				p := make([]float64, c)
				q := make([]float64, c)
				row := make([]float64, c) // privatized: teacher/student widen buffer, per-row only
				for i := rlo; i < rhi; i++ {
					base := i * c
					for j := 0; j < c; j++ {
						row[j] = float64(zts[base+j])
					}
					softmaxRowTInto(p, row, temp)
					for j := 0; j < c; j++ {
						row[j] = float64(zss[base+j])
					}
					softmaxRowTInto(q, row, temp)
					for j := 0; j < c; j++ {
						ds[base+j] = float32(scale * (q[j] - p[j]))
					}
				}
			})
			return []*tensor.Tensor{gs, nil}, nil
		}

		for i := range b {
			p := softmaxRowT(zt, i, c, temp)
			q := softmaxRowT(zs, i, c, temp)
			for j := range c {
				gs.SetF64(scale*(q[j]-p[j]), i, j)
			}
		}
		return []*tensor.Tensor{gs, nil}, nil // teacher frozen
	})
}

// distillParallelRows splits b independent batch rows across GOMAXPROCS workers (sibling
// of conv1dParallelChannels): each worker gets a contiguous [rlo,rhi) row range and
// allocates its own scratch. Serial below a small work threshold (b·workPerRow) so tiny
// batches skip the fork/join. Row writes are disjoint, so this is bit-identical to the
// serial scan.
func distillParallelRows(b, workPerRow int, body func(rlo, rhi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > b {
		nw = b
	}
	if nw <= 1 || b*workPerRow < 1<<15 {
		body(0, b)
		return
	}
	var wg sync.WaitGroup
	chunk := (b + nw - 1) / nw
	for lo := 0; lo < b; lo += chunk {
		hi := lo + chunk
		if hi > b {
			hi = b
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// softmaxRowTInto writes the stable softmax of the logit row (scaled by 1/temp)
// into out (len(out) == len(zrow)). Same passes as softmaxRowT on a typed slice.
func softmaxRowTInto(out, zrow []float64, temp float64) {
	m := math.Inf(-1)
	for j := range zrow {
		if v := zrow[j] / temp; v > m {
			m = v
		}
	}
	var sum float64
	for j := range zrow {
		e := math.Exp(zrow[j]/temp - m)
		out[j] = e
		sum += e
	}
	for j := range out {
		out[j] /= sum
	}
}

// softmaxRowT returns the stable softmax of row i of z scaled by 1/temp.
func softmaxRowT(z *tensor.Tensor, i, c int, temp float64) []float64 {
	m := math.Inf(-1)
	for j := range c {
		if v := z.AtF64(i, j) / temp; v > m {
			m = v
		}
	}
	out := make([]float64, c)
	var sum float64
	for j := range c {
		e := math.Exp(z.AtF64(i, j)/temp - m)
		out[j] = e
		sum += e
	}
	for j := range c {
		out[j] /= sum
	}
	return out
}
