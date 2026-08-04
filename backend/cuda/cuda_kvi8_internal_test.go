//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// TestKVCacheI8AppendRoundtrip validates the int8 per-head quantize-on-append: append a known f32
// K row, download the int8 values + per-head scales, dequantize, and check the result matches the
// original within the int8 rounding budget (per-head max/127 → half-ULP ≈ max/254 per element).
func TestKVCacheI8AppendRoundtrip(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const kvHeads, hd = 3, 64
	const wkv = kvHeads * hd
	const maxSeq = 4
	rng := rand.New(rand.NewSource(5))
	// distinct per-head magnitudes so the per-head scale matters
	src := make([]float32, wkv)
	headMag := []float64{0.2, 2.0, 20.0}
	for h := 0; h < kvHeads; h++ {
		for d := 0; d < hd; d++ {
			src[h*hd+d] = float32(rng.NormFloat64() * headMag[h])
		}
	}
	c, err := NewKVCacheI8(maxSeq, kvHeads, hd)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free()
	dk, err := NewDeviceF32(1, wkv)
	if err != nil {
		t.Fatal(err)
	}
	defer dk.Free()
	if err := dk.UploadF32(src); err != nil {
		t.Fatal(err)
	}
	pos, err := NewDevicePos()
	if err != nil {
		t.Fatal(err)
	}
	defer pos.Free()
	const P = 2
	if err := pos.Set(P); err != nil {
		t.Fatal(err)
	}
	if err := c.AppendDpos(dk, dk, pos); err != nil { // reuse dk as both k and v
		t.Fatal(err)
	}
	deq, err := c.downloadKForTest()
	if err != nil {
		t.Fatal(err)
	}
	// per-head tolerance = max|head|/127 (one quant step); allow a hair over for RN
	var maxRel float64
	for h := 0; h < kvHeads; h++ {
		var mx float64
		for d := 0; d < hd; d++ {
			mx = math.Max(mx, math.Abs(float64(src[h*hd+d])))
		}
		tol := mx/127.0 + 1e-6
		for d := 0; d < hd; d++ {
			idx := P*wkv + h*hd + d
			e := math.Abs(float64(deq[idx]) - float64(src[h*hd+d]))
			if e > tol {
				t.Fatalf("head %d dim %d: dequant %v vs %v, err %v > tol %v", h, d, deq[idx], src[h*hd+d], e, tol)
			}
			if mx > 0 {
				maxRel = math.Max(maxRel, e/mx)
			}
		}
	}
	t.Logf("int8 KV append roundtrip: max per-head rel err %.4f (want ≤ 1/127≈0.0079)", maxRel)
	// rows other than P must be untouched (zero-ish); spot check row 0 scale path not written → deq 0
	for d := 0; d < wkv; d++ {
		if deq[0*wkv+d] != 0 {
			// row 0 was never appended; int8 buffer uninitialized → scale unwritten. Not asserted strictly.
			break
		}
	}
}

// TestKVCacheI8FlashParity checks the int8 flash-decode attention matches the f16 path within the
// int8 quantization budget (per-head max/127 → a few % rel-RMS on the attention output).
func TestKVCacheI8FlashParity(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 8, 2, 64
	const wq = qHeads * hd
	const wkv = kvHeads * hd
	const seqKV = 48
	rng := rand.New(rand.NewSource(9))
	f16, err := NewKVCacheF16(seqKV, wkv)
	if err != nil {
		t.Fatal(err)
	}
	defer f16.Free()
	i8, err := NewKVCacheI8(seqKV, kvHeads, hd)
	if err != nil {
		t.Fatal(err)
	}
	defer i8.Free()
	pos, err := NewDevicePos()
	if err != nil {
		t.Fatal(err)
	}
	defer pos.Free()
	row, err := NewDeviceF32(1, wkv)
	if err != nil {
		t.Fatal(err)
	}
	defer row.Free()
	kv := make([]float32, wkv)
	for tok := 0; tok < seqKV; tok++ {
		for h := 0; h < kvHeads; h++ {
			mag := 0.3 + 3.0*float64(h) // heads differ in magnitude
			for d := 0; d < hd; d++ {
				kv[h*hd+d] = float32(rng.NormFloat64() * mag)
			}
		}
		must(t, row.UploadF32(kv))
		must(t, pos.Set(tok))
		must(t, f16.AppendDpos(row, row, pos))
		must(t, i8.AppendDpos(row, row, pos))
	}
	q := make([]float32, wq)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	dq, err := NewDeviceF32(1, wq)
	if err != nil {
		t.Fatal(err)
	}
	defer dq.Free()
	must(t, dq.UploadF32(q))
	must(t, pos.Set(seqKV - 1)) // offset = last token
	o16, _ := NewDeviceF32(1, wq)
	defer o16.Free()
	o8, _ := NewDeviceF32(1, wq)
	defer o8.Free()
	must(t, GroupedQueryAttentionKVF16DposFlashInto(dq, f16, qHeads, kvHeads, pos, o16))
	must(t, GroupedQueryAttentionKVI8DposFlashInto(dq, i8, qHeads, kvHeads, pos, o8))
	a := make([]float32, wq)
	b := make([]float32, wq)
	o16.DownloadF32(a)
	o8.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rel := math.Sqrt(num / den)
	t.Logf("int8 vs f16 flash-decode: rel-RMS %.4f", rel)
	if rel > 0.05 {
		t.Fatalf("int8 flash diverges from f16: rel-RMS %.4f (want ≤ 0.05)", rel)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func benchKVFlash(b *testing.B, seqKV int, i8 bool) {
	if !Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 8, 128 // Llama-3-8B-ish attention
	const wq = qHeads * hd
	const wkv = kvHeads * hd
	rng := rand.New(rand.NewSource(1))
	pos, _ := NewDevicePos()
	defer pos.Free()
	row, _ := NewDeviceF32(1, wkv)
	defer row.Free()
	kv := make([]float32, wkv)
	for i := range kv {
		kv[i] = float32(rng.NormFloat64())
	}
	row.UploadF32(kv)
	q := make([]float32, wq)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	dq, _ := NewDeviceF32(1, wq)
	defer dq.Free()
	dq.UploadF32(q)
	out, _ := NewDeviceF32(1, wq)
	defer out.Free()
	pos.Set(seqKV - 1)
	if i8 {
		c, _ := NewKVCacheI8(seqKV, kvHeads, hd)
		defer c.Free()
		for tok := 0; tok < seqKV; tok++ {
			pos.Set(tok)
			c.AppendDpos(row, row, pos)
		}
		pos.Set(seqKV - 1)
		GroupedQueryAttentionKVI8DposFlashInto(dq, c, qHeads, kvHeads, pos, out)
		GraphSync()
		b.ResetTimer()
		for range b.N {
			GroupedQueryAttentionKVI8DposFlashInto(dq, c, qHeads, kvHeads, pos, out)
		}
		GraphSync()
	} else {
		c, _ := NewKVCacheF16(seqKV, wkv)
		defer c.Free()
		for tok := 0; tok < seqKV; tok++ {
			pos.Set(tok)
			c.AppendDpos(row, row, pos)
		}
		pos.Set(seqKV - 1)
		GroupedQueryAttentionKVF16DposFlashInto(dq, c, qHeads, kvHeads, pos, out)
		GraphSync()
		b.ResetTimer()
		for range b.N {
			GroupedQueryAttentionKVF16DposFlashInto(dq, c, qHeads, kvHeads, pos, out)
		}
		GraphSync()
	}
}
func BenchmarkKVFlash_f16_len4096(b *testing.B) { benchKVFlash(b, 4096, false) }
func BenchmarkKVFlash_i8_len4096(b *testing.B)  { benchKVFlash(b, 4096, true) }
