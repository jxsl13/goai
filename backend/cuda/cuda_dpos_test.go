//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// The device-position op twins (position read from a DevicePos int) must produce
// exactly the same result as the launch-param versions — the invariant that lets
// a captured decode graph replay across tokens by updating only the device int.
func TestCUDADevicePosOpsMatchParam(t *testing.T) {
	skipNoGPU(t)
	const (
		seq     = 1
		qHeads  = 8
		kvHeads = 2
		hd      = 8
		seqKV   = 10
	)
	wq, wkv := qHeads*hd, kvHeads*hd
	pos, err := cuda.NewDevicePos()
	must(t, err)
	defer pos.Free()

	// (a) RoPEDpos(pos=P) == RoPE(PosOffset=P).
	{
		const P = 7
		x := bench.RandF32(tensor.Shape{seq, wq}, 1)
		dParam, _ := cuda.UploadF32(x)
		dDev, _ := cuda.UploadF32(x)
		must(t, dParam.RoPE(backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: P}))
		must(t, pos.Set(P))
		must(t, dDev.RoPEDpos(backend.RoPEAttrs{Base: 10000, Heads: qHeads}, pos))
		hp, _ := dParam.ToHost()
		hd2, _ := dDev.ToHost()
		if m := maxAbsDiff(hp, hd2); m > 1e-6 {
			t.Fatalf("RoPEDpos != RoPE, maxAbs %.3e", m)
		}
		dParam.Free()
		dDev.Free()
		t.Log("RoPEDpos == RoPE (device position)")
	}

	// (b) GQA-KV-dpos(off) == GQA-KV (off = seqKV-seqQ).
	{
		q := bench.RandF32(tensor.Shape{seq, wq}, 2)
		k := bench.RandF32(tensor.Shape{seqKV, wkv}, 3)
		v := bench.RandF32(tensor.Shape{seqKV, wkv}, 4)
		dq, _ := cuda.UploadF32(q)
		dk, _ := cuda.UploadF32(k)
		dv, _ := cuda.UploadF32(v)
		param, err := cuda.GroupedQueryAttentionKV(dq, dk, dv, qHeads, kvHeads)
		must(t, err)
		must(t, pos.Set(seqKV-seq)) // offset = seqKV - seqQ
		dev, err := cuda.GroupedQueryAttentionKVDpos(dq, dk, dv, qHeads, kvHeads, pos)
		must(t, err)
		hp, _ := param.ToHost()
		hd2, _ := dev.ToHost()
		if m := maxAbsDiff(hp, hd2); m > 1e-6 {
			t.Fatalf("GQA-KV-dpos != GQA-KV, maxAbs %.3e", m)
		}
		dq.Free()
		dk.Free()
		dv.Free()
		param.Free()
		dev.Free()
		t.Log("GQA-KV-dpos == GQA-KV (device offset)")
	}

	// (c) AppendDpos writes each token's k/v at the device row *pos.
	{
		cache, err := cuda.NewKVCache(seqKV, wkv)
		must(t, err)
		defer cache.Free()
		rows := make([]*tensor.Tensor, 3)
		for p := 0; p < 3; p++ {
			k := bench.RandF32(tensor.Shape{1, wkv}, uint64(10+p))
			v := bench.RandF32(tensor.Shape{1, wkv}, uint64(20+p))
			rows[p] = k
			dk, _ := cuda.UploadF32(k)
			dv, _ := cuda.UploadF32(v)
			must(t, pos.Set(p))
			must(t, cache.AppendDpos(dk, dv, pos))
			dk.Free()
			dv.Free()
		}
		cache.SetLen(3)
		got, _ := cache.K().ToHost()
		for p := 0; p < 3; p++ {
			for j := 0; j < wkv; j++ {
				if a, b := got.AtF64(p, j), rows[p].AtF64(0, j); a != b {
					t.Fatalf("AppendDpos row %d col %d: cache %v != %v", p, j, a, b)
				}
			}
		}
		t.Log("AppendDpos writes rows at the device position")
	}
}

func maxAbsDiff(a, b *tensor.Tensor) float64 {
	var m float64
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if d := abs(a.AtF64(idx...) - b.AtF64(idx...)); d > m {
			m = d
		}
	}
	return m
}
