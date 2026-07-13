package nlp_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// randRow builds a random [dim] f32 tensor with values in [-scale,scale).
func randRow(rng *rand.Rand, dim int, scale float64) *tensor.Tensor {
	v := make([]float64, dim)
	for i := range v {
		v[i] = (rng.Float64()*2 - 1) * scale
	}
	return tensor.FromFloat64(tensor.Shape{dim}, v)
}

// §V15 round-trip + fuzz: appending rows and reading them back through the Q8_0 cache is
// bit-identical to a direct gguf.Quantize→gguf.Dequantize (the cache is a faithful pass-through),
// and every recovered element is within the Q8_0 per-block error bound amax/254 (+f16 slack) of
// the original. Fuzzed over many random widths (multiples of 32) and value scales.
func TestQuantKVCacheRoundTripFuzz(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0FFEE))
	for iter := 0; iter < 200; iter++ {
		dim := 32 * (1 + rng.Intn(8)) // 32..256
		scale := math.Pow(10, rng.Float64()*4-2)
		c, err := nlp.NewQuantKVCache(dim)
		if err != nil {
			t.Fatal(err)
		}
		nTok := 1 + rng.Intn(6)
		ks := make([]*tensor.Tensor, nTok)
		vs := make([]*tensor.Tensor, nTok)
		for i := range nTok {
			ks[i] = randRow(rng, dim, scale)
			vs[i] = randRow(rng, dim, scale)
			if err := c.Append(ks[i], vs[i]); err != nil {
				t.Fatal(err)
			}
		}
		K, err := c.Keys()
		if err != nil {
			t.Fatal(err)
		}
		V, err := c.Values()
		if err != nil {
			t.Fatal(err)
		}
		if K.Shape()[0] != nTok || K.Shape()[1] != dim {
			t.Fatalf("Keys shape %v, want [%d %d]", K.Shape(), nTok, dim)
		}
		for i := range nTok {
			checkRow(t, iter, "K", ks[i], K, i, dim)
			checkRow(t, iter, "V", vs[i], V, i, dim)
		}
	}
}

// checkRow asserts cache row r equals the direct gguf round-trip bit-for-bit and stays within the
// Q8_0 error bound of the original.
func checkRow(t *testing.T, iter int, which string, orig, cache *tensor.Tensor, r, dim int) {
	t.Helper()
	// bit-identical to a direct gguf round-trip
	qb, err := gguf.Quantize(orig, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	dq, err := gguf.Dequantize(qb, gguf.Q8_0, dim)
	if err != nil {
		t.Fatal(err)
	}
	amax := 0.0
	for j := range dim {
		if a := math.Abs(orig.AtF64(j)); a > amax {
			amax = a
		}
		if got, want := cache.AtF64(r, j), dq.AtF64(j); got != want {
			t.Fatalf("iter %d %s[%d,%d]: cache %.9g != direct gguf %.9g", iter, which, r, j, got, want)
		}
	}
	// within the Q8_0 per-block bound amax/254 plus f16-scale slack
	tol := amax*(1.0/254) + amax/2048 + 1e-30
	for j := range dim {
		if d := math.Abs(cache.AtF64(r, j) - orig.AtF64(j)); d > tol {
			t.Errorf("iter %d %s[%d,%d]: |Δ|=%.6g > tol %.6g (amax %.6g)", iter, which, r, j, d, tol, amax)
		}
	}
}

// §V16 tier-1 (parity): multi-head attention over the Q8_0 cache matches attention over the
// full-precision f32 K/V within the near-lossless Q8_0 tolerance — the whole point of the cache
// is that quantizing K/V does not perturb the attention output.
func TestQuantKVCacheAttentionParity(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const dim, heads, sq, nTok = 64, 4, 3, 8
	c, _ := nlp.NewQuantKVCache(dim)
	kf := tensor.New(tensor.F32, tensor.Shape{nTok, dim})
	vf := tensor.New(tensor.F32, tensor.Shape{nTok, dim})
	for i := range nTok {
		kr, vr := randRow(rng, dim, 1.0), randRow(rng, dim, 1.0)
		if err := c.Append(kr, vr); err != nil {
			t.Fatal(err)
		}
		for j := range dim {
			kf.SetF64(kr.AtF64(j), i, j)
			vf.SetF64(vr.AtF64(j), i, j)
		}
	}
	q := tensor.New(tensor.F32, tensor.Shape{sq, dim})
	for i := range sq {
		for j := range dim {
			q.SetF64(rng.NormFloat64()*0.5, i, j)
		}
	}
	ctx := backend.NewContext()
	attrs := backend.AttnAttrs{Heads: heads, Causal: false}
	full, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, kf, vf}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	Kq, _ := c.Keys()
	Vq, _ := c.Values()
	quant, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, Kq, Vq}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	var maxd float64
	for i := range sq {
		for j := range dim {
			if d := math.Abs(full[0].AtF64(i, j) - quant[0].AtF64(i, j)); d > maxd {
				maxd = d
			}
		}
	}
	if maxd > 2e-2 { // near-lossless: Q8_0 KV attention ≈ f32 attention (§R108)
		t.Errorf("Q8_0 KV attention max deviation %.4g exceeds 0.02", maxd)
	}
}

// Bytes reports the quantized footprint (34 B per 32-block, keys+values) and it is ~3.76× smaller
// than the equivalent f32 cache (2·t·dim·4 bytes).
func TestQuantKVCacheMemory(t *testing.T) {
	const dim, nTok = 128, 100
	c, _ := nlp.NewQuantKVCache(dim)
	rng := rand.New(rand.NewSource(1))
	for range nTok {
		if err := c.Append(randRow(rng, dim, 1), randRow(rng, dim, 1)); err != nil {
			t.Fatal(err)
		}
	}
	if c.Len() != nTok {
		t.Fatalf("Len %d, want %d", c.Len(), nTok)
	}
	wantBytes := nTok * (dim / 32) * 34 * 2
	if c.Bytes() != wantBytes {
		t.Errorf("Bytes %d, want %d", c.Bytes(), wantBytes)
	}
	f32Bytes := 2 * nTok * dim * 4
	if ratio := float64(f32Bytes) / float64(c.Bytes()); ratio < 3.7 || ratio > 3.8 {
		t.Errorf("memory reduction ratio %.3f, want ~3.76×", ratio)
	}
}

func TestQuantKVCacheErrors(t *testing.T) {
	if _, err := nlp.NewQuantKVCache(48); err == nil {
		t.Error("dim 48 (not a multiple of 32) must error")
	}
	c, _ := nlp.NewQuantKVCache(32)
	bad := tensor.New(tensor.F32, tensor.Shape{16})
	if err := c.Append(bad, bad); err == nil {
		t.Error("appending a 16-element row to a dim-32 cache must error")
	}
}

// A Q8_0 KV cache stores each token's key/value row in 8-bit blocks instead of f32, cutting
// long-context memory ~3.76×. Here two tokens of width 64 are cached: their quantized footprint
// is 2 tokens × (64/32 blocks) × 34 bytes × 2 (K+V) = 272 bytes, versus 2×2×64×4 = 1024 for f32.
func ExampleQuantKVCache() {
	c, _ := nlp.NewQuantKVCache(64)
	k := tensor.New(tensor.F32, tensor.Shape{64})
	v := tensor.New(tensor.F32, tensor.Shape{64})
	_ = c.Append(k, v)
	_ = c.Append(k, v)
	fmt.Printf("%d tokens, %d bytes\n", c.Len(), c.Bytes())
	// Output: 2 tokens, 272 bytes
}
