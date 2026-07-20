// Package gguf reads and writes the GGUF model format (ggml/llama.cpp, §T22, §T107,
// §R7): header (magic "GGUF", version 3), metadata KVs, tensor infos, aligned data
// section. Quantized tensors are dequantized to F32 on load: Q8_0 (f16 scale +
// 32×int8 per block) and Q4_0 (f16 scale + 32 4-bit values, offset −8), plus
// F16→F32 and raw F32/F64. GGUF stores dims innermost-first; shapes are
// reversed into our row-major convention on read. Write serializes a File back
// (tensors as F32), so Read→Write→Read round-trips exactly (§V15). Quantize/Dequantize
// (§T122) also encode f32↔Q8_0/Q4_0 directly, so quantized weights can be produced,
// not just read.
//
// Further reading: the GGUF specification (ggml-org/ggml docs) and the llama.cpp quantization source — the defining references for this format (file formats have no paper, SPEC V16).
//
// In plain terms: GGUF is the file format of the llama.cpp world — weights, tokenizer and metadata in one file, usually with the weights compressed to a few bits each (quantized) so large models fit in ordinary memory.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"

	"github.com/jxsl13/goai/tensor"
)

// nativeLittleEndian reports whether the host stores integers little-endian. GGUF's
// data section is little-endian on disk, so on such hosts (every platform GoAI targets)
// the raw F32 bytes already match the in-memory layout of an F32 tensor and decoding is
// one bulk copy rather than a per-element Float32frombits loop (§T907). Big-endian hosts
// fall back to the element-wise path, so the result is identical either way.
var nativeLittleEndian = func() bool {
	var x uint16 = 1
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// rawCopyLE bulk-copies verbatim little-endian source bytes into the backing store of a
// numeric destination slice, collapsing a per-element decode into one memmove. elemSize
// is the byte width of one T. Returns false on a big-endian host or an empty slice,
// signalling the caller to fall back to the element-wise decode.
func rawCopyLE[T any](dst []T, src []byte, elemSize int) bool {
	if !nativeLittleEndian || len(dst) == 0 {
		return false
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize), src)
	return true
}

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
	tQ2_K = 10 // k-quant super-block, asymmetric affine 2-bit (§R104)
	tQ3_K = 11 // k-quant super-block, symmetric 3-bit + high-bit plane (§R103)
	tQ4_K = 12 // k-quant super-block, asymmetric affine (§R100)
	tQ5_K = 13 // k-quant super-block, asymmetric affine + high-bit plane (§R102)
	tQ6_K = 14 // k-quant super-block (§R99)
)

// File is a parsed GGUF file: metadata and dequantized tensors.
type File struct {
	Version  uint32                    // GGUF format version
	Metadata map[string]any            // key→value header metadata
	Tensors  map[string]*tensor.Tensor // tensor name→info map
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

// parsed is the raw result of reading a GGUF header — metadata + tensor infos + the (still
// encoded) data section — shared by Read (which dequantizes each tensor) and ReadRaw (which
// keeps each tensor in its quantized byte form).
type parsed struct {
	version uint32
	meta    map[string]any
	infos   []tensorInfo
	data    []byte
}

// parse reads the GGUF header, metadata KVs, tensor infos and the aligned data section.
func parse(r io.Reader) (*parsed, error) {
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
		// The product guard divides BEFORE multiplying: a post-multiply check
		// itself wraps uint64 (dims 2^40 × 2^40 = 2^80 ≡ 0) and would let a
		// hostile header pass the cap with a shape whose Numel lies.
		const maxDim = 1 << 40
		numel := uint64(1)
		for _, dv := range dims {
			if dv > maxDim {
				return nil, fmt.Errorf("gguf: tensor %q dim %d exceeds cap", name, dv)
			}
			if dv != 0 && numel > maxDim/dv {
				return nil, fmt.Errorf("gguf: tensor %q element count exceeds cap", name)
			}
			numel *= dv
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

	return &parsed{version: version, meta: meta, infos: infos, data: data}, nil
}

// Read parses a GGUF stream, dequantizing every tensor to an F32 tensor (File.Tensors).
func Read(r io.Reader) (*File, error) {
	p, err := parse(r)
	if err != nil {
		return nil, err
	}
	out := &File{Version: p.version, Metadata: p.meta, Tensors: make(map[string]*tensor.Tensor, len(p.infos))}
	for _, ti := range p.infos {
		t, err := decodeTensor(ti, p.data)
		if err != nil {
			return nil, err
		}
		out.Tensors[ti.name] = t
	}
	return out, nil
}

// QuantTensor is a tensor kept in its ggml block-quantized (or F32/F16) byte form as read from a
// GGUF file — the memory-efficient representation for running a model quantized, instead of a
// dequantized float tensor (§T150). Dequantize expands it on demand.
type QuantTensor struct {
	Data   []byte       // raw bytes in the ggml block layout for GGType
	GGType uint32       // ggml tensor type code (0=F32, 1=F16, 8=Q8_0, 12=Q4_K, …)
	Shape  tensor.Shape // logical shape (row-major)
}

// Dequantize decodes the tensor to an F32 tensor — the same result Read produces for it.
func (q QuantTensor) Dequantize() (*tensor.Tensor, error) {
	return decodeTensor(tensorInfo{shape: q.Shape, ggType: q.GGType, offset: 0}, q.Data)
}

// RawFile is a GGUF parsed with each tensor kept in its quantized byte form (QuantTensor) rather
// than dequantized — for building models that run quantized (§T150). Metadata is read as in Read.
type RawFile struct {
	Version  uint32                 // GGUF version
	Metadata map[string]any         // metadata KVs (same as Read)
	Tensors  map[string]QuantTensor // name → still-quantized tensor
}

// ReadRaw parses a GGUF stream KEEPING each tensor quantized (QuantTensor) — the inverse of the
// eager dequantization Read does, so a quantized model can be loaded without ever materializing
// full-precision weights.
func ReadRaw(r io.Reader) (*RawFile, error) {
	p, err := parse(r)
	if err != nil {
		return nil, err
	}
	out := &RawFile{Version: p.version, Metadata: p.meta, Tensors: make(map[string]QuantTensor, len(p.infos))}
	for _, ti := range p.infos {
		need, err := byteSize(ti.ggType, ti.shape.Numel())
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %q: %w", ti.name, err)
		}
		// subtraction form: offset+need would overflow uint64 for hostile offsets (§B47)
		if ti.offset > uint64(len(p.data)) || uint64(need) > uint64(len(p.data))-ti.offset {
			return nil, fmt.Errorf("gguf: tensor %q data [%d,+%d) beyond section %d",
				ti.name, ti.offset, need, len(p.data))
		}
		raw := make([]byte, need)
		copy(raw, p.data[ti.offset:ti.offset+uint64(need)])
		out.Tensors[ti.name] = QuantTensor{Data: raw, GGType: ti.ggType, Shape: ti.shape}
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
	// subtraction form: offset+need would overflow uint64 for hostile offsets (§B47)
	if ti.offset > uint64(len(data)) || uint64(need) > uint64(len(data))-ti.offset {
		return nil, fmt.Errorf("gguf: tensor %q data [%d,+%d) beyond section %d",
			ti.name, ti.offset, need, len(data))
	}
	raw := data[ti.offset : ti.offset+uint64(need)]

	switch ti.ggType {
	case tF32:
		t := tensor.New(tensor.F32, ti.shape)
		dst := t.Storage().F32()
		// On a little-endian host the raw section IS the F32 bit layout — one memmove
		// (§T907: was a per-element Float32frombits loop at 2.2 GB/s vs gguf-py's 12.2).
		if !rawCopyLE(dst, raw, 4) {
			for i := range dst {
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			}
		}
		return t, nil
	case tF16:
		t := tensor.New(tensor.F32, ti.shape)
		dst := t.Storage().F32()
		// Bulk-copy the LE F16 halfwords into an aligned buffer (one memmove), then table-
		// convert sequentially — avoids the per-element binary.LittleEndian.Uint16 reslice.
		// f16→f32 itself is a table lookup, so the read was the remaining per-element cost.
		if src := make([]uint16, len(dst)); rawCopyLE(src, raw, 2) {
			for i, h := range src {
				dst[i] = f16ToF32(h)
			}
		} else {
			for i := range dst {
				dst[i] = f16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
			}
		}
		return t, nil
	case tQ8_0:
		return dequantQ8_0(ti.shape, raw)
	case tQ4_0:
		return dequantQ4_0(ti.shape, raw)
	case tQ2_K:
		return dequantQ2_K(ti.shape, raw)
	case tQ3_K:
		return dequantQ3_K(ti.shape, raw)
	case tQ4_K:
		return dequantQ4_K(ti.shape, raw)
	case tQ5_K:
		return dequantQ5_K(ti.shape, raw)
	case tQ6_K:
		return dequantQ6_K(ti.shape, raw)
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
	case tQ2_K:
		if n%qkK != 0 {
			return 0, fmt.Errorf("Q2_K numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * q2kBlockSize, nil // 256-element super-block
	case tIQ2_XXS:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ2_XXS numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq2xxsBlockSize, nil // §T554 i-quant super-block
	case tIQ2_XS:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ2_XS numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq2xsBlockSize, nil // §T554 i-quant super-block
	case tIQ3_XXS:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ3_XXS numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq3xxsBlockSize, nil // §T554 i-quant super-block
	case tIQ4_NL:
		if n%blockElems != 0 {
			return 0, fmt.Errorf("IQ4_NL numel %d not multiple of %d", n, blockElems)
		}
		return n / blockElems * iq4nlBlockSize, nil // §T554 i-quant block
	case tIQ4_XS:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ4_XS numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq4xsBlockSize, nil // §T554 i-quant super-block
	case tIQ3_S:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ3_S numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq3sBlockSize, nil // §T554 i-quant super-block
	case tIQ2_S:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ2_S numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq2sBlockSize, nil // §T554 i-quant super-block
	case tIQ1_S:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ1_S numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq1sBlockSize, nil // §T554 i-quant super-block
	case tIQ1_M:
		if n%qkK != 0 {
			return 0, fmt.Errorf("IQ1_M numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * iq1mBlockSize, nil // §T554 i-quant super-block
	case tMXFP4:
		if n%blockElems != 0 {
			return 0, fmt.Errorf("MXFP4 numel %d not multiple of %d", n, blockElems)
		}
		return n / blockElems * mxfp4BlockSize, nil // §T555 microscaling block
	case tQ3_K:
		if n%qkK != 0 {
			return 0, fmt.Errorf("Q3_K numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * q3kBlockSize, nil // 256-element super-block
	case tQ4_K:
		if n%qkK != 0 {
			return 0, fmt.Errorf("Q4_K numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * q4kBlockSize, nil // 256-element super-block
	case tQ5_K:
		if n%qkK != 0 {
			return 0, fmt.Errorf("Q5_K numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * q5kBlockSize, nil // 256-element super-block
	case tQ6_K:
		if n%qkK != 0 {
			return 0, fmt.Errorf("Q6_K numel %d not multiple of %d", n, qkK)
		}
		return n / qkK * q6kBlockSize, nil // 256-element super-block
	default:
		return 0, fmt.Errorf("unsupported ggml type %d", ggType)
	}
}

// dequantQ8_0: per 32-block, x[i] = d · q[i] with d f16 and q int8.
// Fixed-length block subslices let the compiler drop the per-element bounds
// checks in the inner loop (docs/perf-notes-lowlevel.md).
func dequantQ8_0(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dst := t.Storage().F32()
	for b := 0; b*blockElems < len(dst); b++ {
		blk := raw[b*34 : b*34+34]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		y, q := dst[b*blockElems:b*blockElems+blockElems], blk[2:34]
		for i := range y {
			y[i] = d * float32(int8(q[i]))
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
		blk := raw[b*18 : b*18+18]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		y, qs := dst[b*blockElems:b*blockElems+blockElems], blk[2:18]
		for i, q := range qs {
			y[i] = d * float32(int(q&0x0F)-8)
			y[i+16] = d * float32(int(q>>4)-8)
		}
	}
	return t, nil
}

// f16Table caches all 65536 f16→f32 conversions (256 KiB), built at init from
// the reference converter below so the two cannot drift. The F16 tensor decode
// loop and every per-block scale load go through the table — a single L1/L2
// load instead of the branchy bit manipulation (~1.6× on the F16 decode path,
// docs/perf-notes-lowlevel.md; same approach as ggml's ggml_table_f32_f16).
var f16Table [1 << 16]float32

func init() {
	for i := range f16Table {
		f16Table[i] = f16ToF32bits(uint16(i))
	}
}

// f16ToF32 converts an IEEE-754 binary16 to float32 via the lookup table.
func f16ToF32(h uint16) float32 { return f16Table[h] }

// f16ToF32bits converts an IEEE-754 binary16 to float32 (handles subnormals,
// inf, NaN). Reference implementation; runtime conversions use f16Table.
func f16ToF32bits(h uint16) float32 {
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
