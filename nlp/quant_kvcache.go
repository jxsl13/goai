package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// QuantKVCache is a KV cache that stores each token's key and value row in 8-bit Q8_0 block
// format (§R108) instead of f32, cutting long-context KV memory ~3.76× (34 bytes per 32-element
// block = 1.0625 B/elem vs 4 B/elem) — the dominant memory cost of long-context inference. Each
// appended row is quantized independently along its contiguous dimension with the same per-block
// Q8_0 layout used for weights (§R94, already tier-2 verified): per 32 elements one f16 scale
// d=amax/127 and 32 int8 quants. Keys/Values dequantize the whole store back to f32 [t,dim]
// tensors for attention. 8-bit per-block symmetric quantization is the near-lossless tier for KV
// caching — the per-channel-Key/per-token-Value asymmetry (Hooper et al. 2024, KVQuant) only
// matters at 4/3/2-bit, not at 8-bit (§R108). The row width dim must be a multiple of 32.
type QuantKVCache struct {
	dim  int      // per-token row width (multiple of 32)
	keys [][]byte // one Q8_0-encoded row per cached token
	vals [][]byte
}

// NewQuantKVCache creates an empty Q8_0 KV cache for rows of width dim (must be a multiple of the
// 32-element Q8_0 block size).
func NewQuantKVCache(dim int) (*QuantKVCache, error) {
	if dim <= 0 || dim%32 != 0 {
		return nil, fmt.Errorf("nlp: QuantKVCache dim %d must be a positive multiple of 32", dim)
	}
	return &QuantKVCache{dim: dim}, nil
}

// Len returns the number of cached tokens.
func (c *QuantKVCache) Len() int { return len(c.keys) }

// Bytes returns the total quantized storage (keys + values) in bytes — the memory footprint that
// replaces 2·t·dim·4 bytes of f32 cache.
func (c *QuantKVCache) Bytes() int {
	blocks := c.dim / 32
	return len(c.keys) * blocks * 34 * 2 // 34 B/block, keys and values
}

// Append quantizes one token's key and value rows (each [dim] or [1,dim]) to Q8_0 and stores
// them. k and v are consumed as flat dim-vectors.
func (c *QuantKVCache) Append(k, v *tensor.Tensor) error {
	if k.Numel() != c.dim || v.Numel() != c.dim {
		return fmt.Errorf("nlp: QuantKVCache append expects %d-element k,v, got %d and %d",
			c.dim, k.Numel(), v.Numel())
	}
	kb, err := gguf.Quantize(k, gguf.Q8_0)
	if err != nil {
		return err
	}
	vb, err := gguf.Quantize(v, gguf.Q8_0)
	if err != nil {
		return err
	}
	c.keys = append(c.keys, kb)
	c.vals = append(c.vals, vb)
	return nil
}

// Keys dequantizes the cached keys to a fresh f32 tensor [t,dim] (t = Len). Values is its twin.
func (c *QuantKVCache) Keys() (*tensor.Tensor, error) { return c.stack(c.keys) }

// Values dequantizes the cached values to a fresh f32 tensor [t,dim].
func (c *QuantKVCache) Values() (*tensor.Tensor, error) { return c.stack(c.vals) }

// stack dequantizes every stored row and packs them into a [t,dim] f32 tensor.
//
//perfscan:ignore PS6004 not a perf finding: unverified-invariant/correctness check
func (c *QuantKVCache) stack(rows [][]byte) (*tensor.Tensor, error) {
	t := len(rows)
	out := tensor.New(tensor.F32, tensor.Shape{t, c.dim})
	dst := out.Storage().F32()
	for r, row := range rows {
		dq, err := gguf.Dequantize(row, gguf.Q8_0, c.dim)
		if err != nil {
			return nil, err
		}
		// pack the dequantized row directly into out[r,:] — typed, no per-element
		// AtF64/SetF64 dispatch (§base-perf; this runs per attention over the cache).
		o := dst[r*c.dim : r*c.dim+c.dim]
		dqc := dq.Contiguous()
		switch dqc.Dtype() {
		case tensor.F64:
			s := dqc.Storage().F64()
			for j := 0; j < c.dim; j++ {
				o[j] = float32(s[j])
			}
		case tensor.F32:
			copy(o, dqc.Storage().F32())
		default:
			for j := 0; j < c.dim; j++ {
				o[j] = float32(dqc.AtF64(j))
			}
		}
	}
	return out, nil
}
