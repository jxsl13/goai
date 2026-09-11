package cpu

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

const (
	frozenMhaBwdBandRows = 128
	frozenMhaGemmMinSeq  = 16
)

// frozenMhaBwdGemmF32 is the pre-T-01M26VPHRVFAN banded F32 algorithm. It is
// deliberately serial at the task level: bands write disjoint dQ and partial
// slots, then the dK/dV fold uses the production order, so task scheduling is
// not part of the numeric result. The arithmetic primitives remain the real
// SIMD GEMMs and softmax used by production; this freezes the attention
// schedule/layout that the causal pruning candidate may change.
func frozenMhaBwdGemmF32(q, k, v, g, dQ, dK, dV []float32, geo mhaGeo) {
	seq, dk, kvDM := geo.sq, geo.dk, geo.kvDM
	rep, heads := geo.rep, geo.heads
	kvHeads := heads / rep
	pack := make([]float32, kvHeads*3*dk*seq)
	kvStride := 3 * dk * seq
	for kv := range kvHeads {
		for j := range seq {
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
	}

	bands := (seq + frozenMhaBwdBandRows - 1) / frozenMhaBwdBandRows
	part := heads * bands * seq * dk
	pk, pv := make([]float32, part), make([]float32, part)
	for h := range heads {
		for b := range bands {
			i0 := b * frozenMhaBwdBandRows
			iN := min(frozenMhaBwdBandRows, seq-i0)
			slot := (h*bands + b) * seq * dk
			frozenMhaBwdGemmBand(q, g, dQ, pack, pk[slot:slot+seq*dk], pv[slot:slot+seq*dk], geo, h, i0, iN)
		}
	}

	for kv := range kvHeads {
		for j := range seq {
			base := j*kvDM + kv*dk
			ko := dK[base : base+dk : base+dk]
			vo := dV[base : base+dk : base+dk]
			for d := range dk {
				var sk, sv float32
				for hb := kv * rep * bands; hb < (kv+1)*rep*bands; hb++ {
					off := hb*seq*dk + j*dk + d
					sk += pk[off]
					sv += pv[off]
				}
				ko[d], vo[d] = sk, sv
			}
		}
	}
}

func frozenMhaBwdGemmBand(q, g, dQ, pack, partK, partV []float32, geo mhaGeo, h, i0, iN int) {
	seq, dk, dm := geo.sq, geo.dk, geo.dm
	kv := h / geo.rep
	kvStride := 3 * dk * seq
	kt := pack[kv*kvStride : kv*kvStride+dk*seq]
	vt := pack[kv*kvStride+dk*seq : kv*kvStride+2*dk*seq]
	kh := pack[kv*kvStride+2*dk*seq : (kv+1)*kvStride]
	qOff := h * dk

	qb, gb := make([]float32, iN*dk), make([]float32, iN*dk)
	sb, da := make([]float32, iN*seq), make([]float32, iN*seq)
	pt, dat := make([]float32, seq*iN), make([]float32, seq*iN)
	dqb := make([]float32, iN*dk)
	for r := range iN {
		copy(qb[r*dk:(r+1)*dk], q[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
		copy(gb[r*dk:(r+1)*dk], g[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
	}
	jHi := seq
	if geo.causal {
		jHi = min((i0+iN+15)&^15, seq)
	}
	gemmF32RowsCols(qb, kt, sb, 0, iN, dk, seq, 0, jHi)
	mhaSoftmaxBandF32(sb, geo, h, i0, iN)
	gemmF32RowsCols(gb, vt, da, 0, iN, dk, seq, 0, jHi)
	for r := range iN {
		jmin, jmax := geo.bounds(i0 + r)
		pr := sb[r*seq : (r+1)*seq : (r+1)*seq]
		dar := da[r*seq : (r+1)*seq : (r+1)*seq]
		var dot float64
		for j := jmin; j < jmax; j++ {
			dot += float64(pr[j]) * float64(dar[j])
		}
		clear(dar[:jmin])
		for j := jmin; j < jmax; j++ {
			dar[j] = float32(geo.scale * float64(pr[j]) * (float64(dar[j]) - dot))
		}
		clear(dar[jmax:])
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

func frozenMhaBackwardProductionShape(in []*tensor.Tensor, attrs backend.AttnAttrs) []*tensor.Tensor {
	q, k, v, g := in[0], in[1], in[2], in[3]
	pa := attrs.WithDefaults()
	seq, dm := q.Shape()[0], q.Shape()[1]
	if pa.Batch > 1 {
		dQ := tensor.New(tensor.F32, q.Shape())
		dK := tensor.New(tensor.F32, k.Shape())
		dV := tensor.New(tensor.F32, v.Shape())
		qRows, kvRows := seq/pa.Batch, k.Shape()[0]/pa.Batch
		one := pa
		one.Batch = 1
		for b := range pa.Batch {
			qb, _ := q.Slice(0, b*qRows, (b+1)*qRows)
			kb, _ := k.Slice(0, b*kvRows, (b+1)*kvRows)
			vb, _ := v.Slice(0, b*kvRows, (b+1)*kvRows)
			gb, _ := g.Slice(0, b*qRows, (b+1)*qRows)
			part := frozenMhaBackwardProductionShape([]*tensor.Tensor{qb, kb, vb, gb}, one)
			for i, dst := range []*tensor.Tensor{dQ, dK, dV} {
				n := part[i].Numel()
				copy(dst.Storage().F32()[b*n:(b+1)*n], part[i].Storage().F32()[:n])
			}
		}
		return []*tensor.Tensor{dQ, dK, dV}
	}

	dk := dm / pa.Heads
	rep := pa.Heads / pa.KVHeads
	dQ := tensor.New(tensor.F32, q.Shape())
	dK := tensor.New(tensor.F32, k.Shape())
	dV := tensor.New(tensor.F32, v.Shape())
	geo := mhaGeo{
		sq: seq, sk: seq, dm: dm, dk: dk, kvDM: pa.KVHeads * dk,
		heads: pa.Heads, rep: rep, window: pa.Window, causal: pa.Causal,
		scale: pa.Scale / math.Sqrt(float64(dk)),
	}
	if pa.ALiBi {
		geo.slopes = backend.ALiBiSlopes(pa.Heads)
	}
	qc, kc, vc, gc := q.Contiguous(), k.Contiguous(), v.Contiguous(), g.Contiguous()
	if seq >= frozenMhaGemmMinSeq {
		frozenMhaBwdGemmF32(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), gc.Storage().F32(),
			dQ.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), geo)
	} else {
		mhaBwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), gc.Storage().F32(),
			dQ.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), geo)
	}
	return []*tensor.Tensor{dQ, dK, dV}
}

type frozenMhaFixtureKind uint8

const (
	frozenMhaRandom frozenMhaFixtureKind = iota
	frozenMhaSignedZeros
	frozenMhaUnderflowNegativeZero
	frozenMhaOmittedQGFinite
	frozenMhaOverflow
)

type frozenMhaCase struct {
	name                               string
	seq, heads, kvHeads, dk, batch     int
	causal, alibi, noncontiguous       bool
	window                             int
	scale                              float64
	kind                               frozenMhaFixtureKind
	nonfiniteInput, nonfiniteValueKind int // input 0..3; value 1=NaN, 2=+Inf, 3=-Inf
	poisonScratch                      bool
}

func frozenMhaTensor(tb testing.TB, rows, cols int, seed uint64, noncontiguous bool) *tensor.Tensor {
	tb.Helper()
	src := bench.RandF32(tensor.Shape{rows, cols}, seed)
	if !noncontiguous {
		return src
	}
	base := tensor.New(tensor.F32, tensor.Shape{cols, rows + 2})
	transposed, err := base.Transpose(0, 1)
	if err != nil {
		tb.Fatal(err)
	}
	view, err := transposed.Slice(0, 1, rows+1)
	if err != nil {
		tb.Fatal(err)
	}
	for i := range rows {
		for j := range cols {
			view.SetF64(src.AtF64(i, j), i, j)
		}
	}
	if view.Offset() == 0 || view.IsContiguous() {
		tb.Fatal("fixture did not produce an offset non-contiguous view")
	}
	return view
}

func frozenMhaSetFlat(t *tensor.Tensor, flat int, v float32) {
	cols := t.Shape()[1]
	t.SetF64(float64(v), flat/cols, flat%cols)
}

func frozenMhaInputs(tb testing.TB, c frozenMhaCase) []*tensor.Tensor {
	tb.Helper()
	qCols, kvCols := c.heads*c.dk, c.kvHeads*c.dk
	in := []*tensor.Tensor{
		frozenMhaTensor(tb, c.seq, qCols, 101, c.noncontiguous),
		frozenMhaTensor(tb, c.seq, kvCols, 102, c.noncontiguous),
		frozenMhaTensor(tb, c.seq, kvCols, 103, c.noncontiguous),
		frozenMhaTensor(tb, c.seq, qCols, 104, c.noncontiguous),
	}
	switch c.kind {
	case frozenMhaSignedZeros:
		for _, x := range in {
			for i := range x.Numel() {
				bits := uint32(0)
				if i%2 == 0 {
					bits = 1 << 31
				}
				frozenMhaSetFlat(x, i, math.Float32frombits(bits))
			}
		}
	case frozenMhaUnderflowNegativeZero:
		for _, x := range in {
			for i := range x.Numel() {
				frozenMhaSetFlat(x, i, 0)
			}
		}
		for i := range in[3].Numel() {
			frozenMhaSetFlat(in[3], i, 1)
		}
		// On the last causal row P is uniform. V's last row makes dS there
		// positive; multiplying it by K=-minsub rounds the dQ contribution
		// to -0 under the standard round-to-nearest Go FP environment.
		lastKVRow := (in[1].Shape()[0] - 1) * in[1].Shape()[1]
		frozenMhaSetFlat(in[1], lastKVRow, -math.SmallestNonzeroFloat32)
		frozenMhaSetFlat(in[2], lastKVRow, 1)
	case frozenMhaOmittedQGFinite:
		for _, x := range in {
			for i := range x.Numel() {
				frozenMhaSetFlat(x, i, 0)
			}
		}
		patterns := []float32{
			-3,
			math.Float32frombits(1 << 31),
			math.SmallestNonzeroFloat32,
			-math.SmallestNonzeroFloat32,
			math.MaxFloat32,
			-math.MaxFloat32,
		}
		for _, inputIndex := range []int{0, 3} {
			for i := range in[inputIndex].Numel() {
				frozenMhaSetFlat(in[inputIndex], i, patterns[i%len(patterns)])
			}
		}
	case frozenMhaOverflow:
		for inputIndex, x := range in {
			for i := range x.Numel() {
				sign := float32(1)
				if (i+inputIndex)%3 == 0 {
					sign = -1
				}
				frozenMhaSetFlat(x, i, sign*math.MaxFloat32)
			}
		}
	}
	if c.nonfiniteValueKind != 0 {
		var v float32
		switch c.nonfiniteValueKind {
		case 1:
			v = math.Float32frombits(0x7fc00001)
		case 2:
			v = float32(math.Inf(1))
		case 3:
			v = float32(math.Inf(-1))
		default:
			panic("invalid nonfinite fixture kind")
		}
		frozenMhaSetFlat(in[c.nonfiniteInput], 0, v)
	}
	return in
}

func frozenMhaLogicalBits(t *tensor.Tensor) []uint32 {
	bits := make([]uint32, t.Numel())
	cols := t.Shape()[1]
	for i := range bits {
		bits[i] = math.Float32bits(float32(t.AtF64(i/cols, i%cols)))
	}
	return bits
}

func frozenMhaAssertInputUnchanged(tb testing.TB, name string, before []uint32, after *tensor.Tensor) {
	tb.Helper()
	got := frozenMhaLogicalBits(after)
	if len(got) != len(before) {
		tb.Fatalf("%s input length changed: %d != %d", name, len(got), len(before))
	}
	for i := range before {
		if got[i] != before[i] {
			tb.Fatalf("%s input[%d] mutated: bits %08x != %08x", name, i, got[i], before[i])
		}
	}
}

type frozenMhaCompareCounts struct {
	finite, negativeZero, nonfinite int
}

func frozenMhaCompareGradient(tb testing.TB, name string, got, want *tensor.Tensor) frozenMhaCompareCounts {
	tb.Helper()
	if got.Dtype() != tensor.F32 || !got.Shape().Equal(want.Shape()) {
		tb.Fatalf("%s gradient has %v/%v, want %v/%v", name, got.Shape(), got.Dtype(), want.Shape(), want.Dtype())
	}
	gs, ws := got.Storage().F32(), want.Storage().F32()
	var counts frozenMhaCompareCounts
	for i, w := range ws {
		g := gs[i]
		switch {
		case math.IsNaN(float64(w)):
			counts.nonfinite++
			if !math.IsNaN(float64(g)) {
				tb.Fatalf("%s[%d]: got %v, want NaN class", name, i, g)
			}
		case math.IsInf(float64(w), 1):
			counts.nonfinite++
			if !math.IsInf(float64(g), 1) {
				tb.Fatalf("%s[%d]: got %v, want +Inf class", name, i, g)
			}
		case math.IsInf(float64(w), -1):
			counts.nonfinite++
			if !math.IsInf(float64(g), -1) {
				tb.Fatalf("%s[%d]: got %v, want -Inf class", name, i, g)
			}
		default:
			counts.finite++
			gb, wb := math.Float32bits(g), math.Float32bits(w)
			if wb == 1<<31 {
				counts.negativeZero++
			}
			if gb != wb {
				tb.Fatalf("%s[%d]: finite bits %08x != frozen %08x (%v != %v)", name, i, gb, wb, g, w)
			}
		}
	}
	return counts
}

func frozenMhaInstallPoisonedScratch(capacity int) (*atomic.Int64, func()) {
	var acquisitions atomic.Int64
	poisonedNew := func() any {
		acquisitions.Add(1)
		b := make([]float32, capacity)
		for i := range b {
			if i%2 == 0 {
				b[i] = math.Float32frombits(0x7fc0f00d)
			} else {
				b[i] = 12345.5
			}
		}
		b = b[:0]
		return &b
	}
	// Install a fresh pool rather than copying a used sync.Pool. New itself is
	// poisoned, so a GC that drops cached entries cannot silently weaken the
	// test by replacing stale raw storage with fresh zeroed storage.
	f32Scratch = sync.Pool{New: poisonedNew}
	return &acquisitions, func() {
		f32Scratch = sync.Pool{New: func() any { b := make([]float32, 0); return &b }}
	}
}

func TestMHABwdCausalFrozenCurrentExact(t *testing.T) {
	if !f32NativeKernels {
		t.Skip("frozen band oracle requires the SIMD-native F32 production route")
	}
	if mhaBwdBandRows != frozenMhaBwdBandRows || mhaGemmMinSeq != frozenMhaGemmMinSeq {
		t.Fatalf("production MHA backward schedule changed: band/minseq=%d/%d, frozen=%d/%d",
			mhaBwdBandRows, mhaGemmMinSeq, frozenMhaBwdBandRows, frozenMhaGemmMinSeq)
	}
	cpuBackend, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("registered CPU backend unavailable")
	}
	cases := []frozenMhaCase{
		{name: "seq15_scalar_boundary_d1", seq: 15, heads: 1, kvHeads: 1, dk: 1, causal: true},
		{name: "seq16_signed_zero_window1_d3", seq: 16, heads: 2, kvHeads: 2, dk: 3, batch: 1, causal: true, window: 1, kind: frozenMhaSignedZeros},
		{name: "seq16_negative_zero_underflow_d1", seq: 16, heads: 1, kvHeads: 1, dk: 1, causal: true, kind: frozenMhaUnderflowNegativeZero},
		{name: "seq127_gqa_d7_alibi", seq: 127, heads: 4, kvHeads: 2, dk: 7, causal: true, alibi: true},
		{name: "seq128_mha_d64", seq: 128, heads: 8, kvHeads: 8, dk: 64, causal: true},
		{name: "seq129_mqa_d17_window0_offset_strided_poison", seq: 129, heads: 4, kvHeads: 1, dk: 17, causal: true, noncontiguous: true, poisonScratch: true},
		{name: "seq129_mqa_d17_window17_control", seq: 129, heads: 4, kvHeads: 1, dk: 17, causal: true, window: 17},
		{name: "seq129_signed_zero_multiband_d17", seq: 129, heads: 1, kvHeads: 1, dk: 17, causal: true, kind: frozenMhaSignedZeros},
		{name: "seq129_negative_zero_underflow_multiband_d17", seq: 129, heads: 1, kvHeads: 1, dk: 17, causal: true, kind: frozenMhaUnderflowNegativeZero},
		{name: "seq129_omitted_qg_finite_adversarial_d17", seq: 129, heads: 1, kvHeads: 1, dk: 17, causal: true, kind: frozenMhaOmittedQGFinite},
		{name: "seq255_mha_d3_scale", seq: 255, heads: 3, kvHeads: 3, dk: 3, causal: true, scale: 1.25},
		{name: "seq256_batch0_d64", seq: 256, heads: 8, kvHeads: 8, dk: 64, batch: 0, causal: true},
		{name: "seq256_batch1_d64", seq: 256, heads: 8, kvHeads: 8, dk: 64, batch: 1, causal: true},
		{name: "seq256_batch2_boundary_d64", seq: 256, heads: 8, kvHeads: 8, dk: 64, batch: 2, causal: true},
		{name: "seq256_negative_zero_underflow_multiband_d64", seq: 256, heads: 1, kvHeads: 1, dk: 64, causal: true, kind: frozenMhaUnderflowNegativeZero},
		{name: "seq257_noncausal_d7_alibi_control", seq: 257, heads: 2, kvHeads: 2, dk: 7, causal: false, alibi: true},
		{name: "seq257_finite_overflow_multiband_d64", seq: 257, heads: 1, kvHeads: 1, dk: 64, causal: true, kind: frozenMhaOverflow},
		{name: "seq512_mha_d64", seq: 512, heads: 8, kvHeads: 8, dk: 64, causal: true},
		{name: "seq512_batch2_per_sequence256_d17", seq: 512, heads: 2, kvHeads: 1, dk: 17, batch: 2, causal: true},
		{name: "finite_overflow_reachability", seq: 16, heads: 1, kvHeads: 1, dk: 3, causal: true, kind: frozenMhaOverflow},
	}
	for _, dimensions := range []struct {
		name    string
		seq, dk int
	}{
		{name: "reachability", seq: 16, dk: 1},
		{name: "multiband", seq: 129, dk: 17},
	} {
		for input := range 4 {
			for valueKind, valueName := range []string{"NaN", "+Inf", "-Inf"} {
				cases = append(cases, frozenMhaCase{
					name: fmt.Sprintf("nonfinite_%s_%s_in_%c", dimensions.name, valueName, "QKVG"[input]),
					seq:  dimensions.seq, heads: 1, kvHeads: 1, dk: dimensions.dk, causal: true,
					nonfiniteInput: input, nonfiniteValueKind: valueKind + 1,
				})
			}
		}
	}

	var totals frozenMhaCompareCounts
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			inputs := frozenMhaInputs(t, c)
			before := make([][]uint32, len(inputs))
			for i, input := range inputs {
				before[i] = frozenMhaLogicalBits(input)
			}
			attrs := backend.AttnAttrs{
				Heads: c.heads, KVHeads: c.kvHeads, Batch: c.batch, Causal: c.causal,
				Scale: c.scale, ALiBi: c.alibi, Window: c.window,
			}
			want := frozenMhaBackwardProductionShape(inputs, attrs)

			var poisonAcquisitions *atomic.Int64
			if c.poisonScratch {
				oldProcs := runtime.GOMAXPROCS(1)
				var restoreScratch func()
				poisonAcquisitions, restoreScratch = frozenMhaInstallPoisonedScratch(150000)
				t.Cleanup(func() {
					restoreScratch()
					runtime.GOMAXPROCS(oldProcs)
				})
			}
			got, err := backend.Execute(backend.NewContext().WithBackend(cpuBackend), backend.OpMHABackward, inputs, attrs)
			if err != nil {
				t.Fatal(err)
			}
			if poisonAcquisitions != nil && poisonAcquisitions.Load() == 0 {
				t.Fatal("production did not acquire poisoned F32 scratch")
			}
			if len(got) != 3 || len(want) != 3 {
				t.Fatalf("gradient arity got/frozen = %d/%d, want 3/3", len(got), len(want))
			}
			for i, label := range []string{"dQ", "dK", "dV"} {
				counts := frozenMhaCompareGradient(t, label, got[i], want[i])
				totals.finite += counts.finite
				totals.negativeZero += counts.negativeZero
				totals.nonfinite += counts.nonfinite
			}
			for i, label := range []string{"Q", "K", "V", "G"} {
				frozenMhaAssertInputUnchanged(t, label, before[i], inputs[i])
			}
		})
	}
	if totals.finite == 0 || totals.negativeZero == 0 || totals.nonfinite == 0 {
		t.Fatalf("oracle coverage finite/negative-zero/nonfinite = %d/%d/%d, want each nonzero",
			totals.finite, totals.negativeZero, totals.nonfinite)
	}
}
