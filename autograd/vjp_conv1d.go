package autograd

import (
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// conv1dParallelChannels runs body over the D channels in GOMAXPROCS chunks. In the conv1d
// VJP every write (dxs[..*D+c], dws[c*K+..], dbs[c]) is indexed by channel c, so channels
// touch DISJOINT memory — chunk-parallel is race-free, and keeping t-outer/k-inner within
// each channel preserves the per-element ascending-t accumulation → bit-identical to the
// serial loop. Serial below a small work threshold.
func conv1dParallelChannels(d, workPerChan int, body func(clo, chi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > d {
		nw = d
	}
	if nw <= 1 || d*workPerChan < 1<<15 {
		body(0, d)
		return
	}
	var wg sync.WaitGroup
	chunk := (d + nw - 1) / nw
	for lo := 0; lo < d; lo += chunk {
		hi := lo + chunk
		if hi > d {
			hi = d
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

func init() {
	// softplus'(x) = σ(x). Dispatch OpSoftplusBackward on the tape's active backend so the
	// F64 case runs the vectorized vsoftplusGradF64 (4-wide AVX2 exp) instead of a scalar
	// math.Exp loop — the Mamba/Jamba Δ-projection backward, the un-accelerated sibling of
	// the shipped vsoftplusF64 forward (like SiLU/GELU §T362/§T353). Non-F64/F32 and other
	// backends fall back to the reference kernel.
	RegisterVJP(backend.OpSoftplus, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpSoftplusBackward, []*tensor.Tensor{in[0], g}, nil)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out[0]}, nil
	})

	// Causal depthwise conv1d VJP (§R77). out[t,c]=Σ_k w[c,k]·x[t−(K−1)+k,c]+b[c]:
	//	db[c]   = Σ_t g[t,c]
	//	dw[c,k] = Σ_t g[t,c]·x[t−(K−1)+k, c]
	//	dx[j,c] = Σ_{(t,k): t−(K−1)+k=j} g[t,c]·w[c,k]
	RegisterVJP(backend.OpConv1D, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, w := in[0], in[1]
		hasBias := len(in) == 3
		L, D := x.Shape()[0], x.Shape()[1]
		K := w.Shape()[1]

		dx := tensor.New(x.Dtype(), x.Shape())
		dw := tensor.New(w.Dtype(), w.Shape())
		var db *tensor.Tensor
		if hasBias {
			db = tensor.New(in[2].Dtype(), in[2].Shape())
		}

		switch x.Dtype() {
		case tensor.F64:
			if w.Dtype() == tensor.F64 && g.Dtype() == tensor.F64 && (!hasBias || in[2].Dtype() == tensor.F64) {
				xs := x.Contiguous().Storage().F64()
				ws := w.Contiguous().Storage().F64()
				gs := g.Contiguous().Storage().F64()
				dxs := dx.Storage().F64()
				dws := dw.Storage().F64()
				var dbs []float64
				if hasBias {
					dbs = db.Storage().F64()
				}
				// t-outer / c-inner: gs/xs/dxs are indexed [t*D+c] / [j*D+c], so walking c
				// contiguously per fixed t (within this goroutine's [clo,chi) channel block)
				// streams those flat arrays sequentially instead of striding by D each step.
				// Bit-identical: dbs[c], dws[c*K+k], dxs[j*D+c] are distinct per-c cells, each
				// still accumulated in ascending-t order (which c we visit at a given t is
				// irrelevant to a per-c accumulator's summation order).
				conv1dParallelChannels(D, L*K, func(clo, chi int) {
					for t := 0; t < L; t++ {
						for c := clo; c < chi; c++ {
							gv := gs[t*D+c]
							if hasBias {
								dbs[c] += gv
							}
							for k := 0; k < K; k++ {
								j := t - (K - 1) + k
								if j >= 0 {
									dws[c*K+k] += gv * xs[j*D+c]
									dxs[j*D+c] += gv * ws[c*K+k]
								}
							}
						}
					}
				})
				grads := []*tensor.Tensor{dx, dw}
				if hasBias {
					grads = append(grads, db)
				}
				return grads, nil
			}
		case tensor.F32:
			if w.Dtype() == tensor.F32 && g.Dtype() == tensor.F32 && (!hasBias || in[2].Dtype() == tensor.F32) {
				xs := x.Contiguous().Storage().F32()
				ws := w.Contiguous().Storage().F32()
				gs := g.Contiguous().Storage().F32()
				dxs := dx.Storage().F32()
				dws := dw.Storage().F32()
				var dbs []float32
				if hasBias {
					dbs = db.Storage().F32()
				}
				// t-outer / c-inner interchange (see F64 path above): contiguous c-walk over the
				// [t*D+c] / [j*D+c] flat arrays, bit-identical per-c ascending-t accumulation.
				conv1dParallelChannels(D, L*K, func(clo, chi int) {
					for t := 0; t < L; t++ {
						for c := clo; c < chi; c++ {
							gv := float64(gs[t*D+c])
							if hasBias {
								dbs[c] = float32(float64(dbs[c]) + gv)
							}
							for k := 0; k < K; k++ {
								j := t - (K - 1) + k
								if j >= 0 {
									dws[c*K+k] = float32(float64(dws[c*K+k]) + gv*float64(xs[j*D+c]))
									dxs[j*D+c] = float32(float64(dxs[j*D+c]) + gv*float64(ws[c*K+k]))
								}
							}
						}
					}
				})
				grads := []*tensor.Tensor{dx, dw}
				if hasBias {
					grads = append(grads, db)
				}
				return grads, nil
			}
		}

		// generic fallback for exotic dtypes / mixed inputs
		for t := range L {
			for c := range D {
				gv := g.AtF64(t, c)
				if hasBias {
					db.SetF64(db.AtF64(c)+gv, c)
				}
				for k := range K {
					j := t - (K - 1) + k
					if j >= 0 {
						dw.SetF64(dw.AtF64(c, k)+gv*x.AtF64(j, c), c, k)
						dx.SetF64(dx.AtF64(j, c)+gv*w.AtF64(c, k), j, c)
					}
				}
			}
		}
		grads := []*tensor.Tensor{dx, dw}
		if hasBias {
			grads = append(grads, db)
		}
		return grads, nil
	})
}
