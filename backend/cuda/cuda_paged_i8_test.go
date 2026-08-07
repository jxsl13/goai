//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jxsl13/goai/backend/cuda"
)

// buildI8Paged sets up a paged int8 KV cache for `batch` seqs of `seqLen` tokens (seq s owns physical
// block range [s*maxBlocks, s*maxBlocks+maxBlocks)). Returns device buffers + the dequantized (int8·scale)
// host K/V for a reference. Q is random. Deterministic (fixed seed) for reproducibility.
func buildI8Paged(batch, seqLen, qHeads, kvHeads, hd, blockSize int, seed int64) (
	q *cuda.DeviceF32, k8, v8, ks, vs, bt, sl unsafe.Pointer, o *cuda.DeviceF32,
	maxBlocks int, kdeq, vdeq []float64, qh []float32, free func()) {
	rng := rand.New(rand.NewSource(seed))
	kvW := kvHeads * hd
	maxBlocks = (seqLen + blockSize - 1) / blockSize
	nPhys := batch * maxBlocks * blockSize
	k8h := make([]int8, nPhys*kvW)
	v8h := make([]int8, nPhys*kvW)
	ksh := make([]float32, nPhys*kvHeads)
	vsh := make([]float32, nPhys*kvHeads)
	kdeq = make([]float64, nPhys*kvW)
	vdeq = make([]float64, nPhys*kvW)
	// Fill each (seq, token, kvHead) with random f32, quantize per-head to int8 + scale.
	for s := 0; s < batch; s++ {
		for tok := 0; tok < seqLen; tok++ {
			phys := s*maxBlocks*blockSize + tok
			for h := 0; h < kvHeads; h++ {
				var kmax, vmax float64
				kr := make([]float64, hd)
				vr := make([]float64, hd)
				for d := 0; d < hd; d++ {
					kr[d] = rng.NormFloat64() * 0.5
					vr[d] = rng.NormFloat64() * 0.5
					kmax = math.Max(kmax, math.Abs(kr[d]))
					vmax = math.Max(vmax, math.Abs(vr[d]))
				}
				kscale := kmax / 127
				vscale := vmax / 127
				if kscale == 0 {
					kscale = 1
				}
				if vscale == 0 {
					vscale = 1
				}
				ksh[phys*kvHeads+h] = float32(kscale)
				vsh[phys*kvHeads+h] = float32(vscale)
				for d := 0; d < hd; d++ {
					qk := int8(math.Round(kr[d] / kscale))
					qv := int8(math.Round(vr[d] / vscale))
					k8h[phys*kvW+h*hd+d] = qk
					v8h[phys*kvW+h*hd+d] = qv
					kdeq[phys*kvW+h*hd+d] = float64(qk) * kscale
					vdeq[phys*kvW+h*hd+d] = float64(qv) * vscale
				}
			}
		}
	}
	qh = make([]float32, batch*qHeads*hd)
	for i := range qh {
		qh[i] = float32(rng.NormFloat64() * 0.5)
	}
	bth := make([]int32, batch*maxBlocks)
	for s := 0; s < batch; s++ {
		for b := 0; b < maxBlocks; b++ {
			bth[s*maxBlocks+b] = int32(s*maxBlocks + b)
		}
	}
	slh := make([]int32, batch)
	for s := range slh {
		slh[s] = int32(seqLen)
	}
	q, _ = cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qh)
	o, _ = cuda.NewDeviceF32(batch, qHeads*hd)
	k8 = cuda.UploadI8(k8h)
	v8 = cuda.UploadI8(v8h)
	dks, _ := cuda.NewDeviceF32(nPhys, kvHeads)
	dks.UploadF32(ksh)
	dvs, _ := cuda.NewDeviceF32(nPhys, kvHeads)
	dvs.UploadF32(vsh)
	ks, vs = dks.DevPtr(), dvs.DevPtr()
	bt = cuda.UploadI32(bth)
	sl = cuda.UploadI32(slh)
	free = func() {
		q.Free()
		o.Free()
		dks.Free()
		dvs.Free()
		cuda.FreeDev(k8)
		cuda.FreeDev(v8)
		cuda.FreeDev(bt)
		cuda.FreeDev(sl)
	}
	return
}

func TestPagedDecodeAttnI8MatchesRef(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	batch, seqLen := 3, 96
	qHeads, kvHeads, hd, blockSize := 32, 4, 64, 16
	group := qHeads / kvHeads
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q, k8, v8, ks, vs, bt, sl, o, maxBlocks, kdeq, vdeq, qh, free := buildI8Paged(batch, seqLen, qHeads, kvHeads, hd, blockSize, 7)
	defer free()
	if err := cuda.PagedDecodeAttnGQAI8(q, k8, v8, ks, vs, bt, sl, o, batch, qHeads, kvHeads, hd, blockSize, maxBlocks, scale); err != nil {
		t.Fatalf("PagedDecodeAttnGQAI8: %v", err)
	}
	got := make([]float32, batch*qHeads*hd)
	o.DownloadF32(got)
	// host reference over the SAME dequantized (int8·scale) K/V.
	kvW := kvHeads * hd
	var maxErr float64
	for s := 0; s < batch; s++ {
		for h := 0; h < qHeads; h++ {
			kvh := h / group
			scores := make([]float64, seqLen)
			mx := math.Inf(-1)
			for k := 0; k < seqLen; k++ {
				phys := s*maxBlocks*blockSize + k
				var dot float64
				for d := 0; d < hd; d++ {
					dot += float64(qh[s*qHeads*hd+h*hd+d]) * kdeq[phys*kvW+kvh*hd+d]
				}
				scores[k] = dot * float64(scale)
				mx = math.Max(mx, scores[k])
			}
			var sum float64
			for k := 0; k < seqLen; k++ {
				scores[k] = math.Exp(scores[k] - mx)
				sum += scores[k]
			}
			for d := 0; d < hd; d++ {
				var acc float64
				for k := 0; k < seqLen; k++ {
					phys := s*maxBlocks*blockSize + k
					acc += scores[k] * vdeq[phys*kvW+kvh*hd+d]
				}
				ref := acc / sum
				g := float64(got[s*qHeads*hd+h*hd+d])
				maxErr = math.Max(maxErr, math.Abs(ref-g))
			}
		}
	}
	t.Logf("paged int8 decode attn vs host-ref (same dequant): maxErr=%.4g", maxErr)
	if maxErr > 2e-3 {
		t.Fatalf("paged int8 attn maxErr %g too large", maxErr)
	}
}

// TestPagedAppendI8RoundTrip drives the FULL int8-primary serving path: append seqLen f32 K/V tokens per
// sequence THROUGH the quantizing append kernel into an int8 pool, run the int8 paged attention, and
// compare to a host f32 reference computed over the SAME per-(token,head) max/127 quantization the append
// applies. Validates the append kernel + the attention read together.
func TestPagedAppendI8RoundTrip(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	batch, seqLen := 2, 48
	qHeads, kvHeads, hd, blockSize := 32, 4, 64, 16
	group := qHeads / kvHeads
	kvW := kvHeads * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	maxBlocks := (seqLen + blockSize - 1) / blockSize
	nPhys := batch * maxBlocks * blockSize
	rng := rand.New(rand.NewSource(11))
	kv := make([][][]float32, batch)
	vv := make([][][]float32, batch)
	for s := 0; s < batch; s++ {
		kv[s] = make([][]float32, seqLen)
		vv[s] = make([][]float32, seqLen)
		for tk := 0; tk < seqLen; tk++ {
			kv[s][tk] = make([]float32, kvW)
			vv[s][tk] = make([]float32, kvW)
			for i := range kv[s][tk] {
				kv[s][tk][i] = float32(rng.NormFloat64() * 0.5)
				vv[s][tk][i] = float32(rng.NormFloat64() * 0.5)
			}
		}
	}
	k8 := cuda.UploadI8(make([]int8, nPhys*kvW))
	v8 := cuda.UploadI8(make([]int8, nPhys*kvW))
	dks, _ := cuda.NewDeviceF32(nPhys, kvHeads)
	dvs, _ := cuda.NewDeviceF32(nPhys, kvHeads)
	ksp, vsp := dks.DevPtr(), dvs.DevPtr()
	bth := make([]int32, batch*maxBlocks)
	for s := 0; s < batch; s++ {
		for b := 0; b < maxBlocks; b++ {
			bth[s*maxBlocks+b] = int32(s*maxBlocks + b)
		}
	}
	bt := cuda.UploadI32(bth)
	dk, _ := cuda.NewDeviceF32(batch, kvW)
	dv, _ := cuda.NewDeviceF32(batch, kvW)
	for tk := 0; tk < seqLen; tk++ {
		row := make([]float32, batch*kvW)
		rowv := make([]float32, batch*kvW)
		for s := 0; s < batch; s++ {
			copy(row[s*kvW:], kv[s][tk])
			copy(rowv[s*kvW:], vv[s][tk])
		}
		dk.UploadF32(row)
		dv.UploadF32(rowv)
		slh := make([]int32, batch)
		for s := range slh {
			slh[s] = int32(tk)
		}
		sl := cuda.UploadI32(slh)
		if err := cuda.PagedAppendBatchedI8(k8, v8, ksp, vsp, bt, sl, dk, dv, batch, kvW, kvHeads, hd, blockSize, maxBlocks); err != nil {
			t.Fatalf("append tk=%d: %v", tk, err)
		}
		cuda.FreeDev(sl)
	}
	qh := make([]float32, batch*qHeads*hd)
	for i := range qh {
		qh[i] = float32(rng.NormFloat64() * 0.5)
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qh)
	o, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	slh := make([]int32, batch)
	for s := range slh {
		slh[s] = int32(seqLen)
	}
	sl := cuda.UploadI32(slh)
	if err := cuda.PagedDecodeAttnGQAI8(q, k8, v8, ksp, vsp, bt, sl, o, batch, qHeads, kvHeads, hd, blockSize, maxBlocks, scale); err != nil {
		t.Fatalf("attn: %v", err)
	}
	got := make([]float32, batch*qHeads*hd)
	o.DownloadF32(got)
	dequant := func(row []float32, h int) []float64 {
		var mx float64
		for d := 0; d < hd; d++ {
			mx = math.Max(mx, math.Abs(float64(row[h*hd+d])))
		}
		sc := mx / 127
		if sc == 0 {
			sc = 1
		}
		out := make([]float64, hd)
		for d := 0; d < hd; d++ {
			out[d] = math.Round(float64(row[h*hd+d])/sc) * sc
		}
		return out
	}
	var maxErr float64
	for s := 0; s < batch; s++ {
		for h := 0; h < qHeads; h++ {
			kvh := h / group
			scores := make([]float64, seqLen)
			mmx := math.Inf(-1)
			for k := 0; k < seqLen; k++ {
				kd := dequant(kv[s][k], kvh)
				var dot float64
				for d := 0; d < hd; d++ {
					dot += float64(qh[s*qHeads*hd+h*hd+d]) * kd[d]
				}
				scores[k] = dot * float64(scale)
				mmx = math.Max(mmx, scores[k])
			}
			var sum float64
			for k := 0; k < seqLen; k++ {
				scores[k] = math.Exp(scores[k] - mmx)
				sum += scores[k]
			}
			for d := 0; d < hd; d++ {
				var acc float64
				for k := 0; k < seqLen; k++ {
					vd := dequant(vv[s][k], kvh)
					acc += scores[k] * vd[d]
				}
				maxErr = math.Max(maxErr, math.Abs(acc/sum-float64(got[s*qHeads*hd+h*hd+d])))
			}
		}
	}
	q.Free()
	o.Free()
	dk.Free()
	dv.Free()
	dks.Free()
	dvs.Free()
	cuda.FreeDev(k8)
	cuda.FreeDev(v8)
	cuda.FreeDev(bt)
	cuda.FreeDev(sl)
	t.Logf("paged append-i8 + attn round-trip vs ref: maxErr=%.4g", maxErr)
	if maxErr > 2e-3 {
		t.Fatalf("round-trip maxErr %g too large", maxErr)
	}
}

func benchPagedI8(b *testing.B, batch, seqLen int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	qHeads, kvHeads, hd, blockSize := 32, 4, 64, 16
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q, k8, v8, ks, vs, bt, sl, o, maxBlocks, _, _, _, free := buildI8Paged(batch, seqLen, qHeads, kvHeads, hd, blockSize, 1)
	defer free()
	if err := cuda.PagedDecodeAttnGQAI8(q, k8, v8, ks, vs, bt, sl, o, batch, qHeads, kvHeads, hd, blockSize, maxBlocks, scale); err != nil {
		b.Fatalf("warm: %v", err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cuda.PagedDecodeAttnGQAI8(q, k8, v8, ks, vs, bt, sl, o, batch, qHeads, kvHeads, hd, blockSize, maxBlocks, scale)
	}
	cuda.GraphSync()
}

func BenchmarkPagedDecodeAttnI8_b512_len128(b *testing.B) { benchPagedI8(b, 512, 128) }
func BenchmarkPagedDecodeAttnI8_b512_len512(b *testing.B) { benchPagedI8(b, 512, 512) }
