//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
)

// cuda adapter: thin assertions from the backend-agnostic buffer/recorder interfaces to the
// concrete cuda.DeviceF32/cuda.Recorder, the same shape as the metal (§T408) and vulkan (§T409)
// adapters. The one difference is submit semantics: the CUDA recorder enqueues eagerly on the work
// stream (Commit is a no-op, Wait syncs), so asyncEncode stays false — a second recorder must not
// pre-encode onto the shared stream while one is in flight.

type cBuf struct{ *cuda.DeviceF32 }

type cRec struct{ r *cuda.Recorder }

func (c cRec) RMSNorm(x, g, o buffer, rows, dim int, eps float32) error {
	return c.r.RMSNorm(cb(x), cb(g), cb(o), rows, dim, eps)
}
func (c cRec) LayerNorm(x, g, b, o buffer, rows, dim int, eps float32) error {
	return c.r.LayerNorm(cb(x), cb(g), cb(b), cb(o), rows, dim, eps)
}
func (c cRec) AddBias(x, b, o buffer, rows, n int) error {
	return c.r.AddBias(cb(x), cb(b), cb(o), rows, n)
}
func (c cRec) MatMul(a, b, cc buffer, m, k, n int) error {
	return c.r.MatMul(cb(a), cb(b), cb(cc), m, k, n)
}
func (c cRec) MatMulAcc(a, b, cc buffer, m, k, n int) error {
	return c.r.MatMulAcc(cb(a), cb(b), cb(cc), m, k, n)
}
func (c cRec) RoPE(q, inv, o buffer, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return c.r.RoPE(cb(q), cb(inv), cb(o), seq, width, heads, hd, half, pos, posDiv)
}
func (c cRec) RoPEAt(q, inv, o buffer, off, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return c.r.RoPEAt(cb(q), cb(inv), cb(o), off, seq, width, heads, hd, half, pos, posDiv)
}
func (c cRec) RoPEPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, half, pos int, posDiv float32) error {
	return c.r.RoPEPair(cb(qkv), cb(inv), seq, stride, headsQ, offQ, headsK, offK, hd, half, pos, posDiv)
}
func (c cRec) Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error {
	return c.r.Blit(cb(src), srcOff, cb(dst), dstOff, n)
}
func (c cRec) Copy2D(src buffer, srcOff, srcStride int, dst buffer, dstOff, dstStride, rows, rowFloats int) error {
	return c.r.Copy2D(cb(src), srcOff, srcStride, cb(dst), dstOff, dstStride, rows, rowFloats)
}
func (c cRec) MHA(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return c.r.MHA(cb(q), cb(k), cb(v), cb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (c cRec) Unary(x, o buffer, op int) error { return c.r.Unary(cb(x), cb(o), op) }
func (c cRec) Binary(a, b, o buffer, op int) error {
	return c.r.Binary(cb(a), cb(b), cb(o), op)
}
func (c cRec) QMatMulResident(x buffer, w qweight, o buffer, m int) error {
	// Quantized (Q8) adapter is a follow-up: it needs a cuda qweight type with Close()
	// plus cuda.Recorder.QMatMulResident. The f32 NewCUDA path never records this.
	return fmt.Errorf("llamagpu: CUDA QMatMulResident not yet wired (NewCUDA is f32-only)")
}
func (c cRec) Commit() error { return c.r.Commit() }
func (c cRec) Wait() error   { return c.r.Wait() }
func (c cRec) Finish() error { return c.r.Finish() }
func (c cRec) Free()         { c.r.Free() }

func cb(b buffer) *cuda.DeviceF32 { return b.(cBuf).DeviceF32 }

// NewCUDA uploads an nlp.Llama's weights into CUDA device buffers and prepares a KV cache up to
// m.Config.Ctx tokens — the NVIDIA variant of New/NewVulkan. It reuses the backend-agnostic batched
// Decoder core (§T409) via the cuda.Recorder, so Generate/Step/StepN and every sampler work
// unchanged. f32 path (quantized NewQuantCUDA is a follow-up: it needs cuda.Recorder.QMatMulResident).
func NewCUDA(m *nlp.Llama) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false, // one shared work stream — pre-encoding a second recorder would interleave
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}
