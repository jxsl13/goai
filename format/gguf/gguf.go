// Package gguf reads the GGUF model format (ggml/llama.cpp, §T22, §R7):
// header (magic "GGUF", version 3), metadata KVs, tensor infos, aligned data
// section. Quantized tensors are dequantized to F32 on load: Q8_0 (f16 scale +
// 32×int8 per block) and Q4_0 (f16 scale + 32 4-bit values, offset −8), plus
// F16→F32 and raw F32/F64. GGUF stores dims innermost-first; shapes are
// reversed into our row-major convention on read.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/jxsl13/goai/tensor"
)

const magic = 0x46554747 // "GGUF" little-endian

// metadata value types (spec order).
const (
	vtU8 = iota
	vtI8
	vtU16
	vtI16
	vtU32
	vtI32
	vtF32
	vtBool
	vtString
	vtArray
	vtU64
	vtI64
	vtF64
)

// ggml tensor types we support.
const (
	tF32  = 0
	tF16  = 1
	tQ4_0 = 2
	tQ8_0 = 8
)

// File is a parsed GGUF file: metadata and dequantized tensors.
type File struct {
	Version  uint32
	Metadata map[string]any
	Tensors  map[string]*tensor.Tensor
}

type tensorInfo struct {
	name   string
	shape  tensor.Shape // row-major (reversed from file order)
	ggType uint32
	offset uint64
}

type reader struct {
	r io.Reader
	n int64 // bytes consumed
}

func (rd *reader) read(p []byte) error {
	m, err := io.ReadFull(rd.r, p)
	rd.n += int64(m)
	return err
}

func (rd *reader) u32() (uint32, error) {
	var b [4]byte
	if err := rd.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (rd *reader) u64() (uint64, error) {
	var b [8]byte
	if err := rd.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (rd *reader) str() (string, error) {
	n, err := rd.u64()
	if err != nil {
		return "", err
	}
	if n > 1<<24 {
		return "", fmt.Errorf("gguf: unreasonable string length %d", n)
	}
	b := make([]byte, n)
	if err := rd.read(b); err != nil {
		return "", err
	}
	return string(b), nil
}

// value reads one metadata value of type vt. depth bounds array nesting: GGUF
// arrays may themselves hold arrays (element type ARRAY), so a hostile file
// could nest deeply and blow the stack — depth caps that (§V15).
func (rd *reader) value(vt uint32) (any, error) { return rd.valueDepth(vt, 0) }

func (rd *reader) valueDepth(vt uint32, depth int) (any, error) {
	if depth > 64 {
		return nil, fmt.Errorf("gguf: metadata array nesting exceeds cap")
	}
	switch vt {
	case vtU8, vtI8, vtBool:
		var b [1]byte
		if err := rd.read(b[:]); err != nil {
			return nil, err
		}
		switch vt {
		case vtBool:
			return b[0] != 0, nil
		case vtI8:
			return int8(b[0]), nil
		}
		return b[0], nil
	case vtU16, vtI16:
		var b [2]byte
		if err := rd.read(b[:]); err != nil {
			return nil, err
		}
		v := binary.LittleEndian.Uint16(b[:])
		if vt == vtI16 {
			return int16(v), nil
		}
		return v, nil
	case vtU32:
		return rd.u32()
	case vtI32:
		v, err := rd.u32()
		return int32(v), err
	case vtF32:
		v, err := rd.u32()
		return math.Float32frombits(v), err
	case vtU64:
		return rd.u64()
	case vtI64:
		v, err := rd.u64()
		return int64(v), err
	case vtF64:
		v, err := rd.u64()
		return math.Float64frombits(v), err
	case vtString:
		return rd.str()
	case vtArray:
		et, err := rd.u32()
		if err != nil {
			return nil, err
		}
		n, err := rd.u64()
		if err != nil {
			return nil, err
		}
		if n > 1<<24 {
			return nil, fmt.Errorf("gguf: unreasonable array length %d", n)
		}
		arr := make([]any, 0, min(n, 1024)) // grow, don't pre-alloc n·16B from a claim
		for range n {
			v, err := rd.valueDepth(et, depth+1)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("gguf: unknown metadata type %d", vt)
	}
}

// Read parses a GGUF stream.
func Read(r io.Reader) (*File, error) {
	rd := &reader{r: r}
	m, err := rd.u32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if m != magic {
		return nil, fmt.Errorf("gguf: bad magic %#x", m)
	}
	version, err := rd.u32()
	if err != nil {
		return nil, err
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}
	nTensors, err := rd.u64()
	if err != nil {
		return nil, err
	}
	nKV, err := rd.u64()
	if err != nil {
		return nil, err
	}
	if nTensors > 1<<20 || nKV > 1<<20 {
		return nil, fmt.Errorf("gguf: unreasonable counts (%d tensors, %d kvs)", nTensors, nKV)
	}

	meta := make(map[string]any, nKV)
	for range nKV {
		key, err := rd.str()
		if err != nil {
			return nil, err
		}
		vt, err := rd.u32()
		if err != nil {
			return nil, err
		}
		if meta[key], err = rd.value(vt); err != nil {
			return nil, fmt.Errorf("gguf: metadata %q: %w", key, err)
		}
	}

	infos := make([]tensorInfo, nTensors)
	for i := range infos {
		name, err := rd.str()
		if err != nil {
			return nil, err
		}
		nd, err := rd.u32()
		if err != nil {
			return nil, err
		}
		if nd > 8 {
			return nil, fmt.Errorf("gguf: tensor %q has %d dims", name, nd)
		}
		dims := make([]uint64, nd)
		for d := range dims {
			if dims[d], err = rd.u64(); err != nil {
				return nil, err
			}
		}
		// Validate BEFORE building the tensor: hostile dims must error, never
		// panic or wrap (§V15, B28). Cap each dim and the running product so
		// int overflow (2^63 → negative, n·size wrapping to 0) is impossible.
		const maxDim = 1 << 40
		numel := uint64(1)
		for _, dv := range dims {
			if dv > maxDim {
				return nil, fmt.Errorf("gguf: tensor %q dim %d exceeds cap", name, dv)
			}
			numel *= dv
			if numel > maxDim {
				return nil, fmt.Errorf("gguf: tensor %q element count exceeds cap", name)
			}
		}
		// file order is innermost-first → reverse into row-major
		shape := make(tensor.Shape, nd)
		for d := range dims {
			shape[int(nd)-1-d] = int(dims[d])
		}
		ggType, err := rd.u32()
		if err != nil {
			return nil, err
		}
		offset, err := rd.u64()
		if err != nil {
			return nil, err
		}
		infos[i] = tensorInfo{name: name, shape: shape, ggType: ggType, offset: offset}
	}

	// data section starts at the next alignment boundary
	align := int64(32)
	if a, ok := meta["general.alignment"].(uint32); ok && a > 0 {
		align = int64(a)
	}
	if pad := (align - rd.n%align) % align; pad > 0 {
		if err := rd.read(make([]byte, pad)); err != nil {
			return nil, fmt.Errorf("gguf: skip padding: %w", err)
		}
	}
	data, err := io.ReadAll(rd.r)
	if err != nil {
		return nil, fmt.Errorf("gguf: read data: %w", err)
	}

	out := &File{Version: version, Metadata: meta, Tensors: make(map[string]*tensor.Tensor, nTensors)}
	for _, ti := range infos {
		t, err := decodeTensor(ti, data)
		if err != nil {
			return nil, err
		}
		out.Tensors[ti.name] = t
	}
	return out, nil
}

// ReadFile parses a .gguf file.
func ReadFile(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

// decodeTensor extracts and (if needed) dequantizes one tensor from data.
func decodeTensor(ti tensorInfo, data []byte) (*tensor.Tensor, error) {
	n := ti.shape.Numel()
	need, err := byteSize(ti.ggType, n)
	if err != nil {
		return nil, fmt.Errorf("gguf: tensor %q: %w", ti.name, err)
	}
	if ti.offset+uint64(need) > uint64(len(data)) {
		return nil, fmt.Errorf("gguf: tensor %q data [%d,%d) beyond section %d",
			ti.name, ti.offset, ti.offset+uint64(need), len(data))
	}
	raw := data[ti.offset : ti.offset+uint64(need)]

	switch ti.ggType {
	case tF32:
		t := tensor.New(tensor.F32, ti.shape)
		dst := t.Storage().F32()
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return t, nil
	case tF16:
		t := tensor.New(tensor.F32, ti.shape)
		dst := t.Storage().F32()
		for i := range dst {
			dst[i] = f16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return t, nil
	case tQ8_0:
		return dequantQ8_0(ti.shape, raw)
	case tQ4_0:
		return dequantQ4_0(ti.shape, raw)
	default:
		return nil, fmt.Errorf("gguf: tensor %q: unsupported ggml type %d", ti.name, ti.ggType)
	}
}

const blockElems = 32

func byteSize(ggType uint32, n int) (int, error) {
	switch ggType {
	case tF32:
		return n * 4, nil
	case tF16:
		return n * 2, nil
	case tQ8_0:
		if n%blockElems != 0 {
			return 0, fmt.Errorf("Q8_0 numel %d not multiple of %d", n, blockElems)
		}
		return n / blockElems * 34, nil // f16 scale + 32 int8
	case tQ4_0:
		if n%blockElems != 0 {
			return 0, fmt.Errorf("Q4_0 numel %d not multiple of %d", n, blockElems)
		}
		return n / blockElems * 18, nil // f16 scale + 16 nibble-bytes
	default:
		return 0, fmt.Errorf("unsupported ggml type %d", ggType)
	}
}

// dequantQ8_0: per 32-block, x[i] = d · q[i] with d f16 and q int8.
func dequantQ8_0(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dst := t.Storage().F32()
	for b := 0; b*blockElems < len(dst); b++ {
		blk := raw[b*34:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		for i := range blockElems {
			dst[b*blockElems+i] = d * float32(int8(blk[2+i]))
		}
	}
	return t, nil
}

// dequantQ4_0: per 32-block, 16 bytes hold 32 nibbles; x = d·(nibble−8). The
// ggml layout pairs element i with i+16: low nibble → i, high nibble → i+16.
func dequantQ4_0(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dst := t.Storage().F32()
	for b := 0; b*blockElems < len(dst); b++ {
		blk := raw[b*18:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		for i := range 16 {
			q := blk[2+i]
			dst[b*blockElems+i] = d * float32(int(q&0x0F)-8)
			dst[b*blockElems+i+16] = d * float32(int(q>>4)-8)
		}
	}
	return t, nil
}

// f16ToF32 converts an IEEE-754 binary16 to float32 (handles subnormals, inf,
// NaN).
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h) & 0x3FF
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign) // ±0
		}
		// subnormal: normalize
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3FF
		return math.Float32frombits(sign | e<<23 | frac<<13)
	case 0x1F:
		return math.Float32frombits(sign | 0xFF<<23 | frac<<13) // inf/NaN
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
	}
}
