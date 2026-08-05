package cpu

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaMaskedBackwardKernelCPU is the CPU backward of OpMHAMasked (Q,K,V,mask,dO → dQ,dK,dV,dMask).
// Until now OpMHAMaskedBackward had no CPU kernel and fell to backend/ref's per-element (AtF64/f64)
// kernel — 84ms at 512×8 vs the unmasked gemm-routed backward's ~8.7ms — so T5-family training
// (relative-position-bias attention) paid the serial ref. This routes the common self-attention case
// (sq==sk, F32, perf build) through the same f32-native SIMD band gemms as mhaBwdGemmF32, plus the
// additive mask in the softmax recompute and dMask = dScores (the gradient to the pre-softmax score,
// which the mask is added to). Cross-attention (sq≠sk), F64, and non-perf/short shapes fall to the
// exact ref kernel.
func mhaMaskedBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 5 {
		return nil, fmt.Errorf("cpu: mha_masked-backward wants (Q,K,V,mask,dO), got %d", len(in))
	}
	q, k, v, mask, g := in[0], in[1], in[2], in[3], in[4]
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	// Fall to the exact ref kernel for everything but the common self-attention F32 perf-build case.
	if !(f32NativeKernels && q.Dtype() == tensor.F32 && sq == sk && sq >= mhaGemmMinSeq &&
		heads > 0 && dm%heads == 0) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHAMaskedBackward, in, attrs)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHAMaskedBackward, in, attrs)
	}
	rep := heads / kvHeads
	scale := pa.Scale / math.Sqrt(float64(dk))
	perHead := mask.Ndim() == 3

	dQ := tensor.NewOn(ctx.Device(), tensor.F32, q.Shape())
	dK := tensor.NewOn(ctx.Device(), tensor.F32, k.Shape())
	dV := tensor.NewOn(ctx.Device(), tensor.F32, v.Shape())
	dMask := tensor.NewOn(ctx.Device(), tensor.F32, mask.Shape())
	geo := mhaGeo{sq: sq, sk: sk, dm: dm, dk: dk, kvDM: kvHeads * dk, heads: heads, rep: rep, scale: scale}
	mhaMaskedBwdGemmF32(
		q.Contiguous().Storage().F32(), k.Contiguous().Storage().F32(), v.Contiguous().Storage().F32(),
		mask.Contiguous().Storage().F32(), g.Contiguous().Storage().F32(),
		dQ.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), dMask.Storage().F32(), geo, perHead)
	return []*tensor.Tensor{dQ, dK, dV, dMask}, nil
}

// mhaMaskedBwdGemmF32 mirrors mhaBwdGemmF32 (pack Kᵀ/Vᵀ/K → per-(head,band) dQ + dK/dV partials →
// reduce) with the additive mask folded into the softmax recompute and a dMask output. sq==sk (=seq).
func mhaMaskedBwdGemmF32(q, k, v, mask, g, dQ, dK, dV, dMask []float32, geo mhaGeo, perHead bool) {
	seq, dk, kvDM := geo.sq, geo.dk, geo.kvDM
	rep, heads := geo.rep, geo.heads
	kvHeads := heads / rep
	packP := getF32Raw(kvHeads * 3 * dk * seq)
	defer putF32(packP)
	pack := *packP
	kvStride := 3 * dk * seq
	parallelWork(kvHeads*seq, 3*dk, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			kv, j := t/seq, t%seq
			base := j*kvDM + kv*dk
			kr := k[base : base+dk : base+dk]
			vr := v[base : base+dk : base+dk]
			kt := pack[kv*kvStride:]
			vt := pack[kv*kvStride+dk*seq:]
			for d := range dk {
				kt[d*seq+j] = kr[d]
				vt[d*seq+j] = vr[d]
			}
			copy(pack[kv*kvStride+2*dk*seq+j*dk:kv*kvStride+2*dk*seq+(j+1)*dk], kr)
		}
	})

	bands := (seq + mhaBwdBandRows - 1) / mhaBwdBandRows
	part := heads * bands * seq * dk
	partP := getF32Raw(2 * part)
	defer putF32(partP)
	pk, pv := (*partP)[:part], (*partP)[part:]
	// dMask partials: per (head), the band writes its rows' dScores. For a per-head mask that IS the
	// output layout [heads,seq,seq]; for a shared mask [seq,seq] the heads are summed in the reduce.
	dmP := getF32Raw(heads * seq * seq)
	defer putF32(dmP)
	dmPart := *dmP

	nTasks := heads * bands
	workers := max(runtime.GOMAXPROCS(0), 1)
	var nextT atomic.Int64
	parallelWork(workers, (nTasks*mhaBwdBandRows*seq*(5*dk+4)/2+workers-1)/workers, func(_, _ int) {
		for {
			t := int(nextT.Add(1)) - 1
			if t >= nTasks {
				return
			}
			h, b := t%heads, bands-1-t/heads
			i0 := b * mhaBwdBandRows
			iN := min(mhaBwdBandRows, seq-i0)
			slot := (h*bands + b) * seq * dk
			mhaMaskedBwdGemmBand(q, mask, g, dQ, pack, pk[slot:slot+seq*dk], pv[slot:slot+seq*dk],
				dmPart[h*seq*seq:(h+1)*seq*seq], geo, perHead, h, i0, iN)
		}
	})

	parallelWork(kvHeads*seq, rep*bands*dk, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			kv, j := t/seq, t%seq
			base := j*kvDM + kv*dk
			ko := dK[base : base+dk : base+dk]
			vo := dV[base : base+dk : base+dk]
			for d := range dk {
				var skk, sv float32
				for hb := kv * rep * bands; hb < (kv+1)*rep*bands; hb++ {
					off := hb*seq*dk + j*dk + d
					skk += pk[off]
					sv += pv[off]
				}
				ko[d] = skk
				vo[d] = sv
			}
		}
	})

	// dMask: per-head → the partial is already [heads,seq,seq]; shared → sum the heads into [seq,seq].
	if perHead {
		copy(dMask, dmPart[:heads*seq*seq])
	} else {
		parallelWork(seq, heads*seq, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				for j := range seq {
					var s float32
					for h := range heads {
						s += dmPart[h*seq*seq+i*seq+j]
					}
					dMask[i*seq+j] = s
				}
			}
		})
	}
}

// mhaMaskedBwdGemmBand is the masked twin of mhaBwdGemmBand: S=Q·Kᵀ → masked softmax → P; dA=dO·Vᵀ;
// dS=scale·P∘(dA−rowdot) over the FULL row (the mask, not a causal bound, decides the live span);
// dQ=dS·K; dK/dV partials dSᵀ·Q / Pᵀ·dO; and dS is written to the head's dMask slice.
func mhaMaskedBwdGemmBand(q, mask, g, dQ, pack, partK, partV, dmHead []float32, geo mhaGeo, perHead bool, h, i0, iN int) {
	seq, dk, dm := geo.sq, geo.dk, geo.dm
	kv := h / geo.rep
	kvStride := 3 * dk * seq
	kt := pack[kv*kvStride : kv*kvStride+dk*seq]
	vt := pack[kv*kvStride+dk*seq : kv*kvStride+2*dk*seq]
	kh := pack[kv*kvStride+2*dk*seq : (kv+1)*kvStride]
	qOff := h * dk

	qbP, gbP := getF32Raw(iN*dk), getF32Raw(iN*dk)
	sbP, daP := getF32Raw(iN*seq), getF32Raw(iN*seq)
	ptP, dqbP := getF32Raw(2*seq*iN), getF32Raw(iN*dk)
	defer putF32(qbP)
	defer putF32(gbP)
	defer putF32(sbP)
	defer putF32(daP)
	defer putF32(ptP)
	defer putF32(dqbP)
	qb, gb, sb, da, dqb := *qbP, *gbP, *sbP, *daP, *dqbP
	pt, dat := (*ptP)[:seq*iN], (*ptP)[seq*iN:]

	for r := range iN {
		copy(qb[r*dk:(r+1)*dk], q[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
		copy(gb[r*dk:(r+1)*dk], g[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
	}
	// Full sk (arbitrary mask): no causal column-skip.
	gemmF32RowsCols(qb, kt, sb, 0, iN, dk, seq, 0, seq)
	mhaMaskedSoftmaxBandVexpF32(sb, mask, geo, h, i0, iN, perHead) // scale + mask + max-shift softmax
	gemmF32RowsCols(gb, vt, da, 0, iN, dk, seq, 0, seq)
	// dS = scale·P∘(dA − rowdot) over the whole row; masked keys have P=0 → dS=0.
	scale := geo.scale
	for r := range iN {
		pr := sb[r*seq : (r+1)*seq : (r+1)*seq]
		dar := da[r*seq : (r+1)*seq : (r+1)*seq]
		dmr := dmHead[(i0+r)*seq : (i0+r)*seq+seq : (i0+r)*seq+seq]
		var dot float64
		for j := range seq {
			dot += float64(pr[j]) * float64(dar[j])
		}
		// dz = P∘(dA − rowdot) is the gradient to the SOFTMAX INPUT z = raw·scale + mask. The mask is
		// added to z directly, so dMask = dz; the raw-score gradient (for dQ/dK) is dS = scale·dz.
		for j := range seq {
			dz := float64(pr[j]) * (float64(dar[j]) - dot)
			dmr[j] = float32(dz)
			dar[j] = float32(scale * dz)
		}
	}
	gemmF32Rows(da, kh, dqb, 0, iN, seq, dk)
	for r := range iN {
		copy(dQ[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk], dqb[r*dk:(r+1)*dk])
	}
	for r := range iN {
		pr := sb[r*seq : (r+1)*seq : (r+1)*seq]
		dar := da[r*seq : (r+1)*seq : (r+1)*seq]
		for j := range seq {
			pt[j*iN+r] = pr[j]
			dat[j*iN+r] = dar[j]
		}
	}
	gemmF32Rows(pt, gb, partV, 0, seq, iN, dk)
	gemmF32Rows(dat, qb, partK, 0, seq, iN, dk)
}

func init() {
	std.add(backend.OpMHAMaskedBackward, tensor.F32, mhaMaskedBackwardKernelCPU)
}
