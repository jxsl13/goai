package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// nf4Codes is the NF4 (4-bit NormalFloat) codebook from QLoRA (Dettmers et al.
// 2023, arXiv:2305.14314 §3, §R75) — the exact 16 values used by bitsandbytes.
// They are the quantiles of a standard normal N(0,1) (8 negative + 9 positive,
// sharing an exact 0.0 at index 7) normalized to [-1,1], so each of the 16 bins
// holds roughly equal probability mass — information-theoretically optimal for
// zero-centered normally-distributed weights. Sorted ascending.
var nf4Codes = [16]float64{
	-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
	-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
	0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
	0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
}

// NF4Code returns the NF4 codebook value for a 4-bit index (0..15).
func NF4Code(idx int) float64 { return nf4Codes[idx] }

// nearestNF4 returns the index of the codebook value closest to x (argmin, lowest
// index on ties). x is expected in [-1,1] (block-normalized).
func nearestNF4(x float64) uint8 {
	best, bestD := 0, math.Inf(1)
	for i, c := range nf4Codes {
		if d := math.Abs(x - c); d < bestD {
			best, bestD = i, d
		}
	}
	return uint8(best)
}

// QuantizeNF4 block-quantizes w to NF4 (QLoRA §3): each block of blockSize values
// is divided by its absolute maximum into [-1,1], and each normalized value is
// stored as the 4-bit index of the nearest codebook entry (packed two per byte).
// Returns the packed 4-bit indices and the per-block fp64 absmax scales. This is
// the naive single-quantization; double-quantization of the scales (a further
// memory saving) is a documented follow-up. blockSize ≤ 0 defaults to 64.
func QuantizeNF4(w []float64, blockSize int) (packed []uint8, absmax []float64) {
	if blockSize <= 0 {
		blockSize = 64
	}
	n := len(w)
	nblocks := (n + blockSize - 1) / blockSize
	absmax = make([]float64, nblocks)
	packed = make([]uint8, (n+1)/2)
	for b := range nblocks {
		lo := b * blockSize
		hi := min(lo+blockSize, n)
		var am float64
		for i := lo; i < hi; i++ {
			if a := math.Abs(w[i]); a > am {
				am = a
			}
		}
		absmax[b] = am
		for i := lo; i < hi; i++ {
			var idx uint8 = 7 // all-zero block → exact 0.0
			if am > 0 {
				idx = nearestNF4(w[i] / am)
			}
			packed[i/2] |= idx << (4 * (i % 2))
		}
	}
	return packed, absmax
}

// DequantizeNF4 reconstructs n values from NF4-packed indices and per-block absmax
// scales: w_hat[i] = absmax[block] · nf4Codes[index] (QLoRA §3).
func DequantizeNF4(packed []uint8, absmax []float64, blockSize, n int) []float64 {
	if blockSize <= 0 {
		blockSize = 64
	}
	out := make([]float64, n)
	for i := range n {
		idx := (packed[i/2] >> (4 * (i % 2))) & 0x0F
		out[i] = absmax[i/blockSize] * nf4Codes[idx]
	}
	return out
}

// QuantizeScalesNF4 is QLoRA "double quantization" (Dettmers et al. 2023,
// arXiv:2305.14314 §3, §R79): it compresses the per-block fp64 absmax scales that
// QuantizeNF4 produced, from ~32 bits each down to ~8. The scales are centered by
// their mean (so they can be signed-quantized), then 8-bit block-quantized with a
// second block size (default 256): per block, a shared fp64 absmax2 and a signed
// 8-bit code per scale. Returns the codes, the per-second-block absmax2, and the
// global offset (mean). NOTE: bitsandbytes uses a nonlinear "dynamic" 8-bit map;
// this uses a symmetric signed-linear map (code/127) — an equivalent independent
// scheme, so it is NOT bit-compatible with bitsandbytes (the scales are internal
// metadata, not an interchange format), just a smaller self-consistent encoding.
func QuantizeScalesNF4(absmax []float64, blockSize2 int) (codes []int8, absmax2 []float64, offset float64) {
	if blockSize2 <= 0 {
		blockSize2 = 256
	}
	n := len(absmax)
	for _, a := range absmax {
		offset += a
	}
	if n > 0 {
		offset /= float64(n)
	}
	nblocks := (n + blockSize2 - 1) / blockSize2
	absmax2 = make([]float64, nblocks)
	codes = make([]int8, n)
	for b := range nblocks {
		lo := b * blockSize2
		hi := min(lo+blockSize2, n)
		var am2 float64
		for i := lo; i < hi; i++ {
			if c := math.Abs(absmax[i] - offset); c > am2 {
				am2 = c
			}
		}
		absmax2[b] = am2
		for i := lo; i < hi; i++ {
			if am2 > 0 {
				q := math.Round((absmax[i] - offset) / am2 * 127)
				codes[i] = int8(math.Max(-127, math.Min(127, q)))
			}
		}
	}
	return codes, absmax2, offset
}

// DequantizeScalesNF4 reconstructs the fp64 absmax scales from their
// double-quantized form: absmax_hat[i] = code[i]/127·absmax2[block] + offset
// (QLoRA §3, §R79).
func DequantizeScalesNF4(codes []int8, absmax2 []float64, offset float64, blockSize2, n int) []float64 {
	if blockSize2 <= 0 {
		blockSize2 = 256
	}
	out := make([]float64, n)
	for i := range n {
		out[i] = float64(codes[i])/127*absmax2[i/blockSize2] + offset
	}
	return out
}

// QLoRALinear is a QLoRA adapter (Dettmers et al. 2023, §R75): a frozen base
// weight stored in 4-bit NF4 plus a trainable low-rank LoRA adapter (§R27). The
// base is dequantized on the fly for the forward pass; only the adapter (A, B)
// trains, so a large model can be fine-tuned in a fraction of the memory. Forward:
//
//	y = x·dequant_NF4(W) + (alpha/r)·(x·A)·B
type QLoRALinear struct {
	packed    []uint8   // NF4-packed base weight, row-major [in·out]
	absmax    []float64 // per-block scales
	blockSize int
	In, Out   int            // input / output features
	A, B      *tensor.Tensor // trainable adapter
	R         int            // LoRA rank
	Alpha     float64        // LoRA scaling α; effective scale α/R
}

// NewQLoRA quantizes the base weight w[in,out] to NF4 and attaches a rank-r LoRA
// adapter (A Kaiming, B zero — so the adapter starts as a no-op). blockSize ≤ 0
// defaults to 64.
func NewQLoRA(w *tensor.Tensor, blockSize, r int, alpha float64, seed uint64) (*QLoRALinear, error) {
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: QLoRA base must be [in,out], got %v", w.Shape())
	}
	in, out := w.Shape()[0], w.Shape()[1]
	if r <= 0 || r > in || r > out {
		return nil, fmt.Errorf("nn: QLoRA rank %d out of range for [%d,%d]", r, in, out)
	}
	flat := make([]float64, in*out)
	for i := range in {
		for j := range out {
			flat[i*out+j] = w.AtF64(i, j)
		}
	}
	packed, absmax := QuantizeNF4(flat, blockSize)
	a := tensor.New(w.Dtype(), tensor.Shape{in, r})
	KaimingUniform(a, in, seed)
	b := tensor.New(w.Dtype(), tensor.Shape{r, out})
	Zeros(b)
	bs := blockSize
	if bs <= 0 {
		bs = 64
	}
	return &QLoRALinear{packed: packed, absmax: absmax, blockSize: bs, In: in, Out: out, A: a, B: b, R: r, Alpha: alpha}, nil
}

// Base returns the dequantized NF4 base weight [in,out] (a fresh constant tensor).
func (q *QLoRALinear) Base(dtype tensor.Dtype) *tensor.Tensor {
	flat := DequantizeNF4(q.packed, q.absmax, q.blockSize, q.In*q.Out)
	w := tensor.New(dtype, tensor.Shape{q.In, q.Out})
	for i := range q.In {
		for j := range q.Out {
			w.SetF64(flat[i*q.Out+j], i, j)
		}
	}
	return w
}

func (q *QLoRALinear) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes y = x·dequant_NF4(W) + (alpha/r)·(x·A)·B. The dequantized base
// is a constant (frozen), so gradients flow only into the adapter A and B.
func (q *QLoRALinear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	base, err := q.exec(ctx, backend.OpMatMul, nil, x, q.Base(x.Dtype()))
	if err != nil {
		return nil, err
	}
	xa, err := q.exec(ctx, backend.OpMatMul, nil, x, q.A)
	if err != nil {
		return nil, err
	}
	delta, err := q.exec(ctx, backend.OpMatMul, nil, xa, q.B)
	if err != nil {
		return nil, err
	}
	return q.exec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: q.Alpha / float64(q.R)}, delta, base)
}

// Params returns the trainable adapter matrices A and B (the NF4 base is frozen).
func (q *QLoRALinear) Params() []*tensor.Tensor { return []*tensor.Tensor{q.A, q.B} }
