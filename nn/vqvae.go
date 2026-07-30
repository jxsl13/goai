package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// VectorQuantizer implements the VQ-VAE vector-quantization layer (van den Oord,
// Vinyals & Kavukcuoglu 2017, "Neural Discrete Representation Learning",
// arXiv:1711.00937). It maps each continuous encoder vector z_e to the NEAREST
// entry of a learned codebook, producing a discrete latent — the basis of discrete
// representation learning (image/audio tokenizers, neural codecs). The argmin has no
// gradient, so training relies on two devices:
//
//   - Straight-through estimator: the vector handed to the decoder is
//     z_q_st = z_e + stop_gradient(z_q − z_e), whose forward value is the codebook
//     vector z_q but whose gradient is copied straight back to z_e (∂z_q_st/∂z_e = 1).
//   - A two-part VQ loss (Eq. 3): the codebook loss ‖sg[z_e] − e‖² pulls the codebook
//     toward the encoder (encoder detached), and the commitment loss β‖z_e − sg[e]‖²
//     pulls the encoder toward the codebook (codebook detached), so each side is
//     trained by exactly one term. β defaults to 0.25.
//
// Codebook is [K, D] (K entries of dim D); Beta is the commitment cost.
type VectorQuantizer struct {
	Codebook *tensor.Tensor // [K, D] learned embedding table
	Beta     float64        // commitment cost β (default 0.25)
}

// NewVectorQuantizer builds a K×D codebook (zero-initialized — the caller normally
// seeds it, e.g. from data or a uniform init) with commitment cost beta.
func NewVectorQuantizer(dtype tensor.Dtype, k, d int, beta float64) *VectorQuantizer {
	return &VectorQuantizer{Codebook: tensor.New(dtype, tensor.Shape{k, d}), Beta: beta}
}

// Params returns the codebook for the optimizer.
func (vq *VectorQuantizer) Params() []*tensor.Tensor { return []*tensor.Tensor{vq.Codebook} }

// Quantize assigns each row of ze [batch, D] to its nearest codebook entry and
// returns: the straight-through quantized output zqST [batch, D] (feed this to the
// decoder), the VQ loss (codebook + β·commitment; add it to the reconstruction
// loss), and the chosen codebook indices [batch]. zqST is differentiable w.r.t. ze
// (straight-through) and the loss w.r.t. both ze and the codebook (each via its own
// term).
func (vq *VectorQuantizer) Quantize(ctx *backend.Context, ze *tensor.Tensor) (zqST, loss *tensor.Tensor, indices []int, err error) {
	if ze.Ndim() != 2 {
		return nil, nil, nil, fmt.Errorf("nn: VQ.Quantize wants rank-2 [batch,D], got %v", ze.Shape())
	}
	batch, d := ze.Shape()[0], ze.Shape()[1]
	k := vq.Codebook.Shape()[0]
	if vq.Codebook.Shape()[1] != d {
		return nil, nil, nil, fmt.Errorf("nn: VQ codebook dim %d != ze dim %d", vq.Codebook.Shape()[1], d)
	}
	// nearest-neighbor assignment k_i = argmin_j ‖ze_i − e_j‖² (non-differentiable).
	indices = make([]int, batch)
	sel := tensor.New(ze.Dtype(), tensor.Shape{batch, k}) // one-hot [batch,K] for a differentiable gather
	// Nearest-codebook assignment argmin_j ‖ze_i−e_j‖². The old scan paid an AtF64 dispatch
	// d× per (i,j) = batch·k·d dispatches; grab contiguous typed rows once and jam 4 codebook
	// entries so their argmin distances accumulate on independent chains (mirrors AQLM #467).
	// Bit-identical: the read floats and the ascending-c f64 accumulation are unchanged, and
	// the 4-wide `< bd` keeps the lowest index on ties exactly like the sequential scan.
	zc := ze.Contiguous()
	cbc := vq.Codebook.Contiguous()
	switch ze.Dtype() {
	case tensor.F64:
		zs, cb := zc.Storage().F64(), cbc.Storage().F64()
		ss := sel.Storage().F64()
		// Each latent's argmin is independent (reads only its own row + the shared read-only
		// codebook, writes only indices[i] and the disjoint one-hot row i of sel), so the batch
		// fans out over GOMAXPROCS. Each worker keeps the 4-row codebook tiling within its own
		// range; a row's argmin is identical regardless of tile boundary → bit-identical.
		parallelRows(batch, k*d, func(lo, hi int) {
			i := lo
			for ; i+4 <= hi; i += 4 { // tile 4 ze rows per codebook pass (4× less codebook traffic)
				b0, b1, b2, b3 := vqNearest4F64(zs[i*d:i*d+d], zs[(i+1)*d:(i+1)*d+d], zs[(i+2)*d:(i+2)*d+d], zs[(i+3)*d:(i+3)*d+d], cb, k, d)
				indices[i], indices[i+1], indices[i+2], indices[i+3] = b0, b1, b2, b3
				ss[i*k+b0], ss[(i+1)*k+b1], ss[(i+2)*k+b2], ss[(i+3)*k+b3] = 1, 1, 1, 1
			}
			for ; i < hi; i++ { // remainder rows
				best := vqNearestF64(zs[i*d:i*d+d], cb, k, d)
				indices[i] = best
				ss[i*k+best] = 1
			}
		})
	case tensor.F32:
		zs, cb := zc.Storage().F32(), cbc.Storage().F32()
		ss := sel.Storage().F32()
		parallelRows(batch, k*d, func(lo, hi int) {
			i := lo
			for ; i+4 <= hi; i += 4 {
				b0, b1, b2, b3 := vqNearest4F32(zs[i*d:i*d+d], zs[(i+1)*d:(i+1)*d+d], zs[(i+2)*d:(i+2)*d+d], zs[(i+3)*d:(i+3)*d+d], cb, k, d)
				indices[i], indices[i+1], indices[i+2], indices[i+3] = b0, b1, b2, b3
				ss[i*k+b0], ss[(i+1)*k+b1], ss[(i+2)*k+b2], ss[(i+3)*k+b3] = 1, 1, 1, 1
			}
			for ; i < hi; i++ {
				best := vqNearestF32(zs[i*d:i*d+d], cb, k, d)
				indices[i] = best
				ss[i*k+best] = 1
			}
		})
	default:
		for i := range batch { // exotic dtype: keep the per-element dispatch
			best, bestDist := 0, math.Inf(1)
			for j := range k {
				var dist float64
				for c := range d {
					diff := ze.AtF64(i, c) - vq.Codebook.AtF64(j, c)
					dist += diff * diff
				}
				if dist < bestDist {
					best, bestDist = j, dist
				}
			}
			indices[i] = best
			sel.SetF64(1, i, best)
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, e := backend.Execute(ctx, op, in, at)
		if e != nil {
			return nil, e
		}
		return o[0], nil
	}
	zq, err := ex(backend.OpMatMul, nil, sel, vq.Codebook) // [batch,D] = gather, differentiable w.r.t. codebook
	if err != nil {
		return nil, nil, nil, err
	}
	// straight-through: z_q_st = ze + sg(zq − ze) ⇒ forward zq, gradient flows to ze.
	stDelta, err := ex(backend.OpStopGradient, nil, mustSub(ctx, zq, ze))
	if err != nil {
		return nil, nil, nil, err
	}
	if zqST, err = ex(backend.OpAdd, nil, ze, stDelta); err != nil {
		return nil, nil, nil, err
	}
	// per-element squared errors, both [batch,D]: codebook ‖sg[ze] − zq‖² (encoder
	// detached) and commitment ‖ze − sg[zq]‖² (codebook detached).
	sgZe, err := ex(backend.OpStopGradient, nil, ze)
	if err != nil {
		return nil, nil, nil, err
	}
	sqE, err := sq(ctx, mustSub(ctx, sgZe, zq))
	if err != nil {
		return nil, nil, nil, err
	}
	sgZq, err := ex(backend.OpStopGradient, nil, zq)
	if err != nil {
		return nil, nil, nil, err
	}
	sqC, err := sq(ctx, mustSub(ctx, ze, sgZq))
	if err != nil {
		return nil, nil, nil, err
	}
	betaSqC, err := ex(backend.OpMul, nil, sqC, scalarTensor(ze.Dtype(), vq.Beta)) // [batch,D]*[1,1]
	if err != nil {
		return nil, nil, nil, err
	}
	combined, err := ex(backend.OpAdd, nil, sqE, betaSqC) // mean(sqE+β·sqC) = mean(sqE)+β·mean(sqC)
	if err != nil {
		return nil, nil, nil, err
	}
	if loss, err = ex(backend.OpMean, backend.ReduceAttrs{Axes: []int{0, 1}}, combined); err != nil {
		return nil, nil, nil, err
	}
	return zqST, loss, indices, nil
}

// mustSub returns a − b through ctx (panics only on an internal op error, which for
// same-shape rank-2 tensors cannot happen).
func mustSub(ctx *backend.Context, a, b *tensor.Tensor) *tensor.Tensor {
	o, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		panic(err)
	}
	return o[0]
}

// sq returns x² elementwise, differentiably.
func sq(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	o, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, x}, nil)
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

// vqNearestF64 returns argmin_j ‖row−cb[j]‖² over k codebook rows of width d (cb is the
// flat [k·d] codebook), 4-wide unroll-and-jam. `< bd` (strict) keeps the lowest index on ties.
func vqNearestF64(row, cb []float64, k, d int) int {
	best, bd := 0, math.Inf(1)
	j := 0
	for ; j+4 <= k; j += 4 {
		c0 := cb[j*d : j*d+d : j*d+d]
		c1 := cb[(j+1)*d : (j+1)*d+d : (j+1)*d+d]
		c2 := cb[(j+2)*d : (j+2)*d+d : (j+2)*d+d]
		c3 := cb[(j+3)*d : (j+3)*d+d : (j+3)*d+d]
		var d0, d1, d2, d3 float64
		for t, xv := range row {
			e0 := xv - c0[t]
			d0 += e0 * e0
			e1 := xv - c1[t]
			d1 += e1 * e1
			e2 := xv - c2[t]
			d2 += e2 * e2
			e3 := xv - c3[t]
			d3 += e3 * e3
		}
		if d0 < bd {
			bd, best = d0, j
		}
		if d1 < bd {
			bd, best = d1, j+1
		}
		if d2 < bd {
			bd, best = d2, j+2
		}
		if d3 < bd {
			bd, best = d3, j+3
		}
	}
	for ; j < k; j++ {
		cj := cb[j*d : j*d+d : j*d+d]
		var dist float64
		for t, xv := range row {
			e := xv - cj[t]
			dist += e * e
		}
		if dist < bd {
			bd, best = dist, j
		}
	}
	return best
}

// vqNearest4F64 is vqNearestF64 for a TILE of 4 ze rows at once: each codebook row cj is
// loaded once and reused across the 4 rows (4 independent distance accumulators), so the
// k·d codebook streams once per 4-row tile instead of once per row — a 4× cut in codebook
// load traffic on this memory-bound (~0.4 flop/byte) scan, while the 4 independent chains
// keep the same ILP the old 4-wide-j jam had. Bit-identical to four vqNearestF64 calls:
// each row's distance to cj is the same ascending-t f64 sum over the same (row[t]−cb[j][t])
// terms, and each row's argmin scans j ascending with the same strict `< bd` tie-break.
func vqNearest4F64(r0, r1, r2, r3, cb []float64, k, d int) (int, int, int, int) {
	b0, b1, b2, b3 := 0, 0, 0, 0
	bd0, bd1, bd2, bd3 := math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)
	for j := 0; j < k; j++ {
		cj := cb[j*d : j*d+d : j*d+d]
		var d0, d1, d2, d3 float64
		for t, cjt := range cj {
			e0 := r0[t] - cjt
			d0 += e0 * e0
			e1 := r1[t] - cjt
			d1 += e1 * e1
			e2 := r2[t] - cjt
			d2 += e2 * e2
			e3 := r3[t] - cjt
			d3 += e3 * e3
		}
		if d0 < bd0 {
			bd0, b0 = d0, j
		}
		if d1 < bd1 {
			bd1, b1 = d1, j
		}
		if d2 < bd2 {
			bd2, b2 = d2, j
		}
		if d3 < bd3 {
			bd3, b3 = d3, j
		}
	}
	return b0, b1, b2, b3
}

// vqNearest4F32 is vqNearest4F64 over float32 storage (widening each read to float64 exactly
// as AtF64 does), tiling 4 ze rows per codebook pass.
func vqNearest4F32(r0, r1, r2, r3, cb []float32, k, d int) (int, int, int, int) {
	b0, b1, b2, b3 := 0, 0, 0, 0
	bd0, bd1, bd2, bd3 := math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)
	for j := 0; j < k; j++ {
		cj := cb[j*d : j*d+d : j*d+d]
		var d0, d1, d2, d3 float64
		for t := 0; t < d; t++ {
			cjt := float64(cj[t])
			e0 := float64(r0[t]) - cjt
			d0 += e0 * e0
			e1 := float64(r1[t]) - cjt
			d1 += e1 * e1
			e2 := float64(r2[t]) - cjt
			d2 += e2 * e2
			e3 := float64(r3[t]) - cjt
			d3 += e3 * e3
		}
		if d0 < bd0 {
			bd0, b0 = d0, j
		}
		if d1 < bd1 {
			bd1, b1 = d1, j
		}
		if d2 < bd2 {
			bd2, b2 = d2, j
		}
		if d3 < bd3 {
			bd3, b3 = d3, j
		}
	}
	return b0, b1, b2, b3
}

// vqNearestF32 is vqNearestF64 over float32 storage, widening each read to float64 exactly
// as AtF64 on an F32 tensor does — so the accumulated distance (and the argmin) is identical.
func vqNearestF32(row, cb []float32, k, d int) int {
	best, bd := 0, math.Inf(1)
	j := 0
	for ; j+4 <= k; j += 4 {
		c0 := cb[j*d : j*d+d : j*d+d]
		c1 := cb[(j+1)*d : (j+1)*d+d : (j+1)*d+d]
		c2 := cb[(j+2)*d : (j+2)*d+d : (j+2)*d+d]
		c3 := cb[(j+3)*d : (j+3)*d+d : (j+3)*d+d]
		var d0, d1, d2, d3 float64
		for t := range row {
			xv := float64(row[t])
			e0 := xv - float64(c0[t])
			d0 += e0 * e0
			e1 := xv - float64(c1[t])
			d1 += e1 * e1
			e2 := xv - float64(c2[t])
			d2 += e2 * e2
			e3 := xv - float64(c3[t])
			d3 += e3 * e3
		}
		if d0 < bd {
			bd, best = d0, j
		}
		if d1 < bd {
			bd, best = d1, j+1
		}
		if d2 < bd {
			bd, best = d2, j+2
		}
		if d3 < bd {
			bd, best = d3, j+3
		}
	}
	for ; j < k; j++ {
		cj := cb[j*d : j*d+d : j*d+d]
		var dist float64
		for t := range row {
			e := float64(row[t]) - float64(cj[t])
			dist += e * e
		}
		if dist < bd {
			bd, best = dist, j
		}
	}
	return best
}
