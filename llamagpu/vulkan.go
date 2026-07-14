//go:build vulkan && cgo

package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/nlp"
)

// vulkan adapter (§T409): thin assertions from the backend-agnostic buffer/recorder interfaces to
// the concrete vulkan.DeviceBuffer/vulkan.Recorder — the same API as metal by construction (§T408).

type vBuf struct{ *vulkan.DeviceBuffer }

type vRec struct{ r *vulkan.Recorder }

func (v vRec) RMSNorm(x, g, o buffer, rows, dim int, eps float32) error {
	return v.r.RMSNorm(vb(x), vb(g), vb(o), rows, dim, eps)
}
func (v vRec) LayerNorm(x, g, b, o buffer, rows, dim int, eps float32) error {
	return v.r.LayerNorm(vb(x), vb(g), vb(b), vb(o), rows, dim, eps)
}
func (v vRec) AddBias(x, b, o buffer, rows, n int) error {
	return v.r.AddBias(vb(x), vb(b), vb(o), rows, n)
}
func (v vRec) MatMul(a, b, c buffer, m, k, n int) error {
	return v.r.MatMul(vb(a), vb(b), vb(c), m, k, n)
}
func (v vRec) RoPE(q, inv, o buffer, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return v.r.RoPE(vb(q), vb(inv), vb(o), seq, width, heads, hd, half, pos, posDiv)
}
func (v vRec) RoPEAt(q, inv, o buffer, off, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return v.r.RoPEAt(vb(q), vb(inv), vb(o), off, seq, width, heads, hd, half, pos, posDiv)
}
func (v vRec) Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error {
	return v.r.Blit(vb(src), srcOff, vb(dst), dstOff, n)
}
func (v vRec) Copy2D(src buffer, srcOff, srcStride int, dst buffer, dstOff, dstStride, rows, rowFloats int) error {
	return v.r.Copy2D(vb(src), srcOff, srcStride, vb(dst), dstOff, dstStride, rows, rowFloats)
}
func (v vRec) MHA(q, k, va, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return v.r.MHA(vb(q), vb(k), vb(va), vb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (v vRec) Unary(x, o buffer, op int) error { return v.r.Unary(vb(x), vb(o), op) }
func (v vRec) Binary(a, b, o buffer, op int) error {
	return v.r.Binary(vb(a), vb(b), vb(o), op)
}
func (v vRec) QMatMulResident(x buffer, w qweight, o buffer, m int) error {
	return v.r.QMatMulResident(vb(x), w.(*vulkan.ResidentQWeight), vb(o), m)
}
func (v vRec) Finish() error { return v.r.Finish() }
func (v vRec) Free()         { v.r.Free() }

func vb(b buffer) *vulkan.DeviceBuffer { return b.(vBuf).DeviceBuffer }

// NewVulkan uploads m's weights into vulkan device buffers and prepares a KV cache up to
// m.Config.Ctx tokens — the vulkan variant of the batched decode (decode is dispatch-bound on
// vulkan too; §T390 measured 6.15× batched-vs-per-op at the recorder level).
func NewVulkan(m *nlp.Llama) (*Decoder, error) {
	if !vulkan.Available() {
		return nil, fmt.Errorf("llamagpu: no vulkan GPU")
	}
	return newDecoder(m, backendOps{
		name: string(backend.Vulkan),
		newBuffer: func(data []float32) (buffer, error) {
			b, err := vulkan.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return vBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := vulkan.NewRecorder()
			if err != nil {
				return nil, err
			}
			return vRec{r}, nil
		},
		uploadQWeight: vulkanUploadQWeight,
	})
}

// NewGPTVulkan uploads an nlp.GPT's weights into vulkan device buffers for batched decoding —
// the vulkan variant of NewGPT (§T422).
func NewGPTVulkan(m *nlp.GPT) (*GPTDecoder, error) {
	if !vulkan.Available() {
		return nil, fmt.Errorf("llamagpu: no vulkan GPU")
	}
	return newGPTDecoder(m, backendOps{
		name: string(backend.Vulkan),
		newBuffer: func(data []float32) (buffer, error) {
			b, err := vulkan.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return vBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := vulkan.NewRecorder()
			if err != nil {
				return nil, err
			}
			return vRec{r}, nil
		},
	})
}

func vulkanUploadQWeight(weight []byte, qt uint32, n, k int) (qweight, error) {
	rw, err := vulkan.Backend{}.UploadQuant(weight, qt, n, k)
	if err != nil {
		return nil, err
	}
	return rw.(*vulkan.ResidentQWeight), nil
}

// NewQuantVulkan uploads a quantized Llama with every projection as a device-resident quantized
// weight — the vulkan variant of NewQuant (§T415).
func NewQuantVulkan(m *nlp.QuantLlama) (*Decoder, error) {
	if !vulkan.Available() {
		return nil, fmt.Errorf("llamagpu: no vulkan GPU")
	}
	return newQuantDecoder(m, backendOps{
		name: string(backend.Vulkan),
		newBuffer: func(data []float32) (buffer, error) {
			b, err := vulkan.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return vBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := vulkan.NewRecorder()
			if err != nil {
				return nil, err
			}
			return vRec{r}, nil
		},
		uploadQWeight: vulkanUploadQWeight,
	})
}
