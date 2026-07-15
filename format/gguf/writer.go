package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// Write serializes f to w as GGUF (the ggml/llama.cpp format, §R7): magic, version,
// counts, metadata KVs, tensor infos, then the alignment-padded data section. It is
// the inverse of Read and round-trips exactly (§V15): metadata values keep their GGUF
// types, tensor shapes are reversed back to the file's innermost-first order, and each
// tensor's data is placed at its declared offset on a general.alignment boundary
// (default 32). Tensors are written as F32 — the lossless in-memory form Read produces
// (it dequantizes on load), so a Read→Write→Read cycle reproduces the tensor bytes.
// Metadata keys and tensor names are emitted in sorted order for a deterministic file.
func Write(w io.Writer, f *File) error { return WriteQuantized(w, f, nil) }

// WriteQuantized is Write with per-tensor quantization: a tensor named in quant is
// encoded in that block-quantized format (Q8_0/Q4_0, §T122) instead of F32, producing a
// real quantized model (~4×/7× smaller). Un-named tensors stay F32. On read the
// quantized tensors are dequantized, so the round-trip is exact for F32 tensors and
// within the quantization error for quantized ones (§V15).
func WriteQuantized(w io.Writer, f *File, quant map[string]QuantType) error {
	bw := bufio.NewWriter(w)
	wr := &writer{w: bw}

	version := f.Version
	if version < 2 || version > 3 {
		version = 3
	}
	align := int64(32)
	if a, ok := f.Metadata["general.alignment"].(uint32); ok && a > 0 {
		align = int64(a)
	}

	// stable order
	metaKeys := make([]string, 0, len(f.Metadata))
	for k := range f.Metadata {
		metaKeys = append(metaKeys, k)
	}
	sort.Strings(metaKeys)
	tenNames := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		tenNames = append(tenNames, k)
	}
	sort.Strings(tenNames)

	// per-tensor ggml type + byte length (quantized tensors are encoded once here).
	ggTypes := make([]uint32, len(tenNames))
	lens := make([]uint64, len(tenNames))
	qbytes := make([][]byte, len(tenNames)) // nil = write F32 lazily
	for i, name := range tenNames {
		t := f.Tensors[name]
		if qt, ok := quant[name]; ok {
			b, err := Quantize(t, qt)
			if err != nil {
				return fmt.Errorf("gguf: quantize %q: %w", name, err)
			}
			ggTypes[i], qbytes[i], lens[i] = uint32(qt), b, uint64(len(b))
		} else {
			ggTypes[i], lens[i] = tF32, uint64(t.Numel()*4)
		}
	}

	// precompute each tensor's aligned offset within the data section
	offsets := make([]uint64, len(tenNames))
	var off uint64
	for i := range tenNames {
		offsets[i] = off
		off = alignUp(off+lens[i], uint64(align))
	}

	wr.u32(magic)
	wr.u32(version)
	wr.u64(uint64(len(tenNames)))
	wr.u64(uint64(len(metaKeys)))
	for _, k := range metaKeys {
		wr.str(k)
		if err := wr.kv(f.Metadata[k]); err != nil {
			return fmt.Errorf("gguf: write metadata %q: %w", k, err)
		}
	}
	for i, name := range tenNames {
		sh := f.Tensors[name].Shape()
		wr.str(name)
		wr.u32(uint32(len(sh)))
		for d := len(sh) - 1; d >= 0; d-- { // row-major → innermost-first
			wr.u64(uint64(sh[d]))
		}
		wr.u32(ggTypes[i])
		wr.u64(offsets[i])
	}
	if wr.err != nil {
		return wr.err
	}

	// pad header → data section to the alignment boundary
	if pad := (align - wr.n%align) % align; pad > 0 {
		wr.pad(int(pad))
	}
	// data section: each tensor at its offset, padded to alignment between tensors
	var written uint64
	for i, name := range tenNames {
		if pad := offsets[i] - written; pad > 0 {
			wr.pad(int(pad))
			written += pad
		}
		if qbytes[i] != nil {
			wr.write(qbytes[i])
		} else {
			wr.f32Data(f.Tensors[name])
		}
		written += lens[i]
	}
	if wr.err != nil {
		return wr.err
	}
	return bw.Flush()
}

// WriteFile writes f to a .gguf file at path.
func WriteFile(path string, f *File) error {
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Write(out, f); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func alignUp(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) / a * a
}

type writer struct {
	w   io.Writer
	n   int64 // bytes written
	err error
}

func (wr *writer) write(p []byte) {
	if wr.err != nil {
		return
	}
	m, err := wr.w.Write(p)
	wr.n += int64(m)
	wr.err = err
}

func (wr *writer) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	wr.write(b[:])
}

func (wr *writer) u64(v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	wr.write(b[:])
}

func (wr *writer) str(s string) {
	wr.u64(uint64(len(s)))
	wr.write([]byte(s))
}

func (wr *writer) pad(n int) { wr.write(make([]byte, n)) }

func (wr *writer) f32Data(t *tensor.Tensor) {
	if wr.err != nil {
		return
	}
	buf := make([]byte, t.Numel()*4)
	switch c := t.Contiguous(); c.Dtype() {
	case tensor.F32:
		for i, v := range c.Storage().F32() {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
	case tensor.F64: // bulk convert (avoids the per-element AtF64/Unravel dispatch)
		for i, v := range c.Storage().F64() {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
		}
	default: // best-effort convert any other dtype to F32
		for i := range t.Numel() {
			v := float32(t.AtF64(tensor.Unravel(i, t.Shape())...))
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
	}
	wr.write(buf)
}

// kv writes a top-level metadata value: its GGUF type tag then its body.
func (wr *writer) kv(v any) error {
	vt, err := ggufType(v)
	if err != nil {
		return err
	}
	wr.u32(vt)
	return wr.body(vt, v)
}

// body writes a value's bytes for the given GGUF type (no type tag).
func (wr *writer) body(vt uint32, v any) error {
	switch vt {
	case vtU8:
		wr.write([]byte{v.(uint8)})
	case vtI8:
		wr.write([]byte{byte(v.(int8))})
	case vtBool:
		var b byte
		if v.(bool) {
			b = 1
		}
		wr.write([]byte{b})
	case vtU16:
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], v.(uint16))
		wr.write(b[:])
	case vtI16:
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(v.(int16)))
		wr.write(b[:])
	case vtU32:
		wr.u32(v.(uint32))
	case vtI32:
		wr.u32(uint32(v.(int32)))
	case vtF32:
		wr.u32(math.Float32bits(v.(float32)))
	case vtU64:
		wr.u64(v.(uint64))
	case vtI64:
		wr.u64(uint64(v.(int64)))
	case vtF64:
		wr.u64(math.Float64bits(v.(float64)))
	case vtString:
		wr.str(v.(string))
	case vtArray:
		arr := v.([]any)
		et := uint32(vtI32) // element type of an empty array is unobservable on read-back
		if len(arr) > 0 {
			t, err := ggufType(arr[0])
			if err != nil {
				return err
			}
			et = t
		}
		wr.u32(et)
		wr.u64(uint64(len(arr)))
		for _, e := range arr {
			t, err := ggufType(e)
			if err != nil {
				return err
			}
			if t != et {
				return fmt.Errorf("gguf: heterogeneous array (%d vs %d)", t, et)
			}
			if err := wr.body(et, e); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("gguf: cannot write metadata type %d", vt)
	}
	return wr.err
}

// ggufType maps a Go value produced by Read back to its GGUF metadata type tag.
func ggufType(v any) (uint32, error) {
	switch v.(type) {
	case uint8:
		return vtU8, nil
	case int8:
		return vtI8, nil
	case uint16:
		return vtU16, nil
	case int16:
		return vtI16, nil
	case uint32:
		return vtU32, nil
	case int32:
		return vtI32, nil
	case float32:
		return vtF32, nil
	case bool:
		return vtBool, nil
	case string:
		return vtString, nil
	case uint64:
		return vtU64, nil
	case int64:
		return vtI64, nil
	case float64:
		return vtF64, nil
	case []any:
		return vtArray, nil
	default:
		return 0, fmt.Errorf("gguf: unsupported metadata value type %T", v)
	}
}
