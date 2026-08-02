package cpu

import (
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized conv2d (§T24b): im2col + the blocked GEMM band kernel. Columns are
// laid out in (c,ky,kx) order — the exact accumulation order of the reference
// conv — and the GEMM preserves per-element k order, so results are
// bit-identical to backend/ref (§V3, §V11 tol 0). Everything accumulates in
// f64 (§V10); f32 narrows once on store. Pooling stays on the reference
// kernels via fallback (§I4) — conv is where the GEMM payoff lives.

func conv2dKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 && len(in) != 3 {
		return nil, fmt.Errorf("cpu: conv2d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	if x.Ndim() != 4 || w.Ndim() != 4 {
		return nil, fmt.Errorf("cpu: conv2d needs x[N,C,H,W] and w[F,C,KH,KW], got %v/%v", x.Shape(), w.Shape())
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("cpu: conv2d channel mismatch x C=%d vs w C=%d", c, wc)
	}
	var bias *tensor.Tensor
	if len(in) == 3 {
		bias = in[2]
		if bias.Ndim() != 1 || bias.Shape()[0] != f {
			return nil, fmt.Errorf("cpu: conv2d bias must be [%d], got %v", f, bias.Shape())
		}
	}
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()
	s := pa.Stride
	p := pa.Pad
	if s < 1 || p < 0 {
		return nil, fmt.Errorf("cpu: conv2d invalid stride %d / pad %d", s, p)
	}
	ho := (h+2*p-kh)/s + 1
	wo := (wd+2*p-kw)/s + 1
	if ho < 1 || wo < 1 {
		return nil, fmt.Errorf("cpu: conv2d output would be empty (%dx%d)", ho, wo)
	}

	xc, wcont := x.Contiguous(), w.Contiguous()
	k := c * kh * kw
	rows := n * ho * wo

	// Perf builds (f32NativeKernels): route the f32 conv through the f32-native
	// SIMD gemmF32 entry point (AVX/NEON) instead of the scalar f64 band kernel —
	// same ADR-0021/0026 tolerance-for-speed trade as the matmul op. The default
	// build keeps the bit-exact fused f64 path below.
	if f32NativeKernels && x.Dtype() == tensor.F32 {
		return conv2dF32Native(ctx, xc, wcont, bias, convGeo{
			n: n, c: c, h: h, wd: wd, f: f, kh: kh, kw: kw,
			s: s, p: p, ho: ho, wo: wo, k: k, rows: rows,
		})
	}

	// wt[(c,ky,kx), f]: transposed weights matching the column order
	wtP := getF64(k * f)
	defer putF64(wtP)
	wt := *wtP

	// The im2col matrix and the GEMM product are per-WORKER, per-CHUNK scratch, not
	// whole-tensor buffers. The three per-row stages run fused inside one parallelWork
	// band — they were separate barriers once, and the profile showed ~75% of the op in
	// pool wake/park churn between them — but the buffers they hand each other were still
	// sized rows x k and rows x f, so every column written by im2col went out to DRAM and
	// came back for the GEMM. im2col replicates each input element kh*kw times, so that
	// round trip is the dominant traffic in the op: one 512x512 head convolution of the
	// multi-token-attention benchmark materialized 138 MB to multiply it by a 66-element
	// weight vector. Sized to a chunk instead, the columns are still in L2 when the GEMM
	// reads them, and the op's footprint stops scaling with the image.
	//
	// Bit-identical: each row's im2col values, its GEMM accumulation order and its scatter
	// target are untouched — only the buffer they live in while in flight changed.
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, f, ho, wo})
	var bs []float64
	if bias != nil {
		bs = make([]float64, f)
		for fi := range f {
			bs[fi] = bias.AtF64(fi)
		}
	}
	hw := ho * wo
	work := k + k*f + f // per-row: im2col fill + GEMM row + scatter
	nw, cw := convSlots(rows, k, f, 8)
	colsP := getF64(nw * cw * k)
	defer putF64(colsP)
	cols := *colsP
	prodP := getF64(nw * cw * f)
	defer putF64(prodP)
	prod := *prodP
	var slot atomic.Int32
	switch x.Dtype() {
	case tensor.F64:
		xs, os := xc.Storage().F64(), out.Storage().F64()
		wtFill(wt, wcont.Storage().F64(), f, k)
		parallelWork(rows, work, func(lo, hi int) {
			sc, sp := convSlice(cols, &slot, cw, k, f)
			pw := prod[sp : sp+cw*f : sp+cw*f]
			// The pool hands out a ZEROED window, so the FIRST chunk needs no clearing and
			// only the chunks that reuse it do. That distinction is what keeps a conv small
			// enough to fit one chunk per band exactly as cheap as it was before chunking.
			for base, reused := lo, false; base < hi; base, reused = base+cw, true {
				end := min(base+cw, hi)
				if reused {
					if p > 0 { // padded taps are never written; an unpadded row is filled whole
						clear(sc[:(end-base)*k])
					}
					// gemmF64Band ACCUMULATES into C, so a reused product window starts at zero.
					clear(pw[:(end-base)*f])
				}
				im2colFillBand(sc, xs, base, end, base, k, ho, wo, c, kh, kw, s, p, h, wd)
				gemmF64Band(sc, wt, pw, 0, end-base, k, f)
				convScatterBand(pw, os, bs, base, end, base, f, hw)
			}
		})
	case tensor.F32:
		xs, os := xc.Storage().F32(), out.Storage().F32()
		wtFill(wt, wcont.Storage().F32(), f, k)
		parallelWork(rows, work, func(lo, hi int) {
			sc, sp := convSlice(cols, &slot, cw, k, f)
			pw := prod[sp : sp+cw*f : sp+cw*f]
			// The pool hands out a ZEROED window, so the FIRST chunk needs no clearing and
			// only the chunks that reuse it do. That distinction is what keeps a conv small
			// enough to fit one chunk per band exactly as cheap as it was before chunking.
			for base, reused := lo, false; base < hi; base, reused = base+cw, true {
				end := min(base+cw, hi)
				if reused {
					if p > 0 { // padded taps are never written; an unpadded row is filled whole
						clear(sc[:(end-base)*k])
					}
					// gemmF64Band ACCUMULATES into C, so a reused product window starts at zero.
					clear(pw[:(end-base)*f])
				}
				im2colFillBand(sc, xs, base, end, base, k, ho, wo, c, kh, kw, s, p, h, wd)
				gemmF64Band(sc, wt, pw, 0, end-base, k, f)
				convScatterBand(pw, os, bs, base, end, base, f, hw)
			}
		})
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// convGeo bundles the conv2d geometry shared by the kernel paths.
type convGeo struct {
	n, c, h, wd, f, kh, kw, s, p, ho, wo, k, rows int
}

// conv2dF32Native is the gemm-routed f32 conv of the perf builds: the same
// fused ONE-parallelWork row-band pipeline as the f64 path below (im2col →
// GEMM band → scatter, §T463's pool-churn lesson), but with an f32 column
// matrix and the f32-native SIMD band kernel (NEON on arm64 / AVX on amd64 via
// gemmF32Rows). Numerics: the im2col fill and weight transpose are exact
// copies, so the only rounding change vs the default path is the gemm's f32
// accumulation (K·u_f32-bounded, ADR-0021 tolerance) and the bias add in f32 —
// both inside the gemmF32Tolerant parity budget.
func conv2dF32Native(ctx *backend.Context, xc, wcont, bias *tensor.Tensor, g convGeo) ([]*tensor.Tensor, error) {
	k, rows, f := g.k, g.rows, g.f
	wtP := getF32Raw(k * f) // fully overwritten by wtFill
	defer putF32(wtP)
	wt := *wtP

	xs, ws := xc.Storage().F32(), wcont.Storage().F32()
	wtFill(wt, ws, f, k)

	out := tensor.NewOn(ctx.Device(), tensor.F32, tensor.Shape{g.n, f, g.ho, g.wo})
	os := out.Storage().F32()
	var bs []float32
	if bias != nil {
		bs = make([]float32, f)
		for fi := range f {
			bs[fi] = float32(bias.AtF64(fi))
		}
	}
	hw := g.ho * g.wo
	work := k + k*f + f // per-row: im2col fill + GEMM row + scatter
	nw, cw := convSlots(rows, k, f, 4)
	colsP := getF32(nw * cw * k) // zeroed: padding taps stay 0
	defer putF32(colsP)
	prodP := getF32Raw(nw * cw * f) // store semantics: fully written by the gemm
	defer putF32(prodP)
	allCols, allProd := *colsP, *prodP
	var slot atomic.Int32
	parallelWork(rows, work, func(lo, hi int) {
		sl := int(slot.Add(1)-1) * cw
		cols := allCols[sl*k : sl*k+cw*k : sl*k+cw*k]
		prod := allProd[sl*f : sl*f+cw*f : sl*f+cw*f]
		for cbase, reused := lo, false; cbase < hi; cbase, reused = cbase+cw, true {
			cend := min(cbase+cw, hi)
			if reused && g.p > 0 { // the pool's window arrives zeroed; only a reused one needs clearing
				clear(cols[:(cend-cbase)*k])
			}
			im2colFillBand(cols, xs, cbase, cend, cbase, k, g.ho, g.wo, g.c, g.kh, g.kw, g.s, g.p, g.h, g.wd)
			gemmF32Rows(cols, wt, prod, 0, cend-cbase, k, f)
			for r := cbase; r < cend; r++ {
				ni, rem := r/hw, r%hw
				pr := prod[(r-cbase)*f : (r-cbase)*f+f : (r-cbase)*f+f]
				if bs != nil {
					for fi, v := range pr {
						os[(ni*f+fi)*hw+rem] = v + bs[fi]
					}
				} else {
					for fi, v := range pr {
						os[(ni*f+fi)*hw+rem] = v
					}
				}
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

// convSlots returns how many scratch slots the conv needs and how many output rows one
// slot holds. Each parallelWork band claims one slot and walks its rows a slot at a time.
//
// The slot is sized so its column matrix and products fit in a couple of hundred kilobytes.
// im2col replicates every input element kh*kw times, so with a whole-tensor buffer each
// column went out to DRAM and came straight back for the GEMM. Keeping the slot L2-resident
// removes that round trip.
//
// The row count is ALSO capped at one band's worth, which is what keeps the total scratch
// below the buffer it replaces instead of workers times a slot. Without the cap, small
// convolutions — where the old buffer already fit in cache and there was nothing to win —
// paid more memory than before to gain nothing: measured -22.7% on the largest torch shape
// but +14 to +33% on the small ones.
func convSlots(rows, k, f, elem int) (slots, rowsPerSlot int) {
	slots = max(runtime.GOMAXPROCS(0), 1)
	band := (rows + slots - 1) / slots
	target := 1
	if per := (k + f) * elem; per > 0 {
		target = max(1, min(convChunkBytes/per, 4096))
	}
	return slots, max(1, min(target, band))
}

// convChunkBytes targets the L2 residency of one slot's column matrix and products.
const convChunkBytes = 1 << 18

// convScatterBand writes prod rows [lo,hi) — prod[(n,oy,ox), f] — into
// out[n,f,ho,wo], adding the hoisted bias when present.
func convScatterBand[T normFloat](prod []float64, os []T, bs []float64, lo, hi, off, f, hw int) {
	for r := lo; r < hi; r++ {
		ni, rem := r/hw, r%hw
		pr := prod[(r-off)*f : (r-off)*f+f : (r-off)*f+f]
		if bs != nil {
			for fi, v := range pr {
				os[(ni*f+fi)*hw+rem] = T(v + bs[fi])
			}
		} else {
			for fi, v := range pr {
				os[(ni*f+fi)*hw+rem] = T(v)
			}
		}
	}
}

func init() {
	std.add(backend.OpConv2D, tensor.F32, conv2dKernel)
	std.add(backend.OpConv2D, tensor.F64, conv2dKernel)
}

// im2colFillBand materializes im2col rows [lo,hi) (§T342) from a concrete []T
// input slice into the GEMM scratch (f64 on the default path, f32 on the
// gemm-routed f32-native path — the copy is exact either way) — direct indexed
// reads instead of the old per-element get closure. Padding taps stay 0
// (adding 0·w is bit-safe).
func im2colFillBand[D, T normFloat](cols []D, xs []T, lo, hi, off, k, ho, wo, c, kh, kw, s, p, h, wd int) {
	for r := lo; r < hi; r++ {
		ni := r / (ho * wo)
		rem := r % (ho * wo)
		oy, ox := rem/wo, rem%wo
		base := (r - off) * k // off rebases the write onto a chunk-local matrix; r still addresses x
		// Along the kernel width the input x-coord ix = ox·s − p + kx steps by 1, so
		// the in-bounds kx taps form ONE contiguous input run [kxLo,kxHi). Hoist the
		// x-bounds test out of the inner loop (compute the window once per row) and
		// copy the run branch-free; padding taps outside the window stay pre-zeroed
		// (§T342 relies on cols being zeroed). Bit-identical values to the per-element
		// bounds-checked gather — im2col was 62% of the f32 Conv2D (the GEMM is only
		// ~30%), so killing the per-tap branch is the lever.
		ix0 := ox*s - p
		kxLo, kxHi := 0, kw
		if ix0 < 0 {
			kxLo = -ix0
		}
		if ix0+kw > wd {
			kxHi = wd - ix0
		}
		kk := 0
		for ci := 0; ci < c; ci++ {
			for ky := 0; ky < kh; ky++ {
				iy := oy*s + ky - p
				if iy >= 0 && iy < h && kxLo < kxHi {
					rowBase := ((ni*c+ci)*h+iy)*wd + ix0
					dst := cols[base+kk+kxLo : base+kk+kxHi]
					src := xs[rowBase+kxLo : rowBase+kxHi]
					for i := range dst {
						dst[i] = D(src[i])
					}
				}
				kk += kw
			}
		}
	}
}

// wtFill transposes the weights [F, C·KH·KW] into column order [C·KH·KW, F] over a
// concrete []T slice (devirtualized, no get closure). The destination is f64 on
// the default path and f32 on the gemm-routed f32-native path (exact copy).
func wtFill[D, T normFloat](wt []D, ws []T, f, k int) {
	for fi := 0; fi < f; fi++ {
		for kk := 0; kk < k; kk++ {
			wt[kk*f+fi] = D(ws[fi*k+kk])
		}
	}
}

// convSlice claims the next scratch slot for a parallelWork band and returns its column
// window plus the base offset of its product window. parallelWork calls its body once per
// band and never more often than GOMAXPROCS, so the counter cannot outrun the slots.
func convSlice(cols []float64, slot *atomic.Int32, cw, k, f int) ([]float64, int) {
	s := int(slot.Add(1)-1) * cw
	return cols[s*k : s*k+cw*k : s*k+cw*k], s * f
}
