// Package safetensors reads and writes the HuggingFace safetensors format
// (§T19, §R7): u64-LE header length, JSON header mapping tensor names to
// {dtype, shape, data_offsets}, then the raw little-endian C-order data
// section. Offsets are validated strictly — they must tile the data section
// exactly with no gaps or overlaps, mirroring the reference implementation.
//
// Writing supports F32, F64, F16, BF16. The 16-bit floats are stored verbatim
// as their raw little-endian uint16 bits, so they round-trip bit-exactly with
// no widening. Writing is deterministic: names are emitted in sorted order
// with contiguous offsets.
//
// Loading additionally accepts every remaining dtype of the official spec and
// WIDENS it (§T577), because the tensor layer has no integer/FP8 storage:
// F8_E4M3 and F8_E5M2 (DeepSeek-V3-class FP8 checkpoints) decode exactly to
// F32 — see fp8.go; BOOL (0→0, nonzero→1), I8, U8, I16 and U16 widen exactly
// to F32; I32 and U32 widen exactly to F64; I64 and U64 widen to F64, which is
// exact up to 2^53 (beyond that the nearest representable is taken — token
// ids, masks and index tensors are far below this).
//
// Further reading: the official safetensors format specification (huggingface/safetensors) — the defining reference for this format (file formats have no paper, SPEC V16).
//
// In plain terms: safetensors is the HuggingFace standard for storing model weights — a simple, safe layout (a JSON table of contents plus raw numbers) that cannot execute code when loaded, unlike Python pickles.
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// maxHeaderSize mirrors the reference implementation's 100 MB header cap.
const maxHeaderSize = 100 * 1024 * 1024

type entry struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

func dtypeName(d tensor.Dtype) (string, error) {
	switch d {
	case tensor.F32:
		return "F32", nil
	case tensor.F64:
		return "F64", nil
	case tensor.F16:
		return "F16", nil
	case tensor.BF16:
		return "BF16", nil
	default:
		return "", fmt.Errorf("safetensors: unsupported dtype %v", d)
	}
}

// fileDtype describes how one on-disk dtype loads: its per-element byte size
// in the data section and the tensor dtype it widens into (§T577).
type fileDtype struct {
	size int
	out  tensor.Dtype
}

func dtypeOf(name string) (fileDtype, error) {
	switch name {
	case "F32":
		return fileDtype{4, tensor.F32}, nil
	case "F64":
		return fileDtype{8, tensor.F64}, nil
	case "F16":
		return fileDtype{2, tensor.F16}, nil
	case "BF16":
		return fileDtype{2, tensor.BF16}, nil
	case "F8_E4M3", "F8_E5M2", "BOOL", "I8", "U8":
		return fileDtype{1, tensor.F32}, nil
	case "I16", "U16":
		return fileDtype{2, tensor.F32}, nil
	case "I32", "U32":
		return fileDtype{4, tensor.F64}, nil
	case "I64", "U64":
		return fileDtype{8, tensor.F64}, nil
	default:
		return fileDtype{}, fmt.Errorf("safetensors: unsupported dtype %q", name)
	}
}

// Save writes tensors (and optional string metadata) to w. Deterministic:
// sorted names, contiguous offsets, header padded to 8 bytes with spaces.
func Save(w io.Writer, tensors map[string]*tensor.Tensor, meta map[string]string) error {
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	header := make(map[string]any, len(tensors)+1)
	if len(meta) > 0 {
		header["__metadata__"] = meta
	}
	offset := 0
	var data bytes.Buffer
	for _, n := range names {
		t := tensors[n].Contiguous()
		dn, err := dtypeName(t.Dtype())
		if err != nil {
			return fmt.Errorf("%w (tensor %q)", err, n)
		}
		size := t.Numel() * t.Dtype().Size()
		header[n] = entry{Dtype: dn, Shape: append([]int{}, t.Shape()...), DataOffsets: [2]int{offset, offset + size}}
		switch t.Dtype() {
		case tensor.F32:
			for _, v := range t.Storage().F32() {
				var b [4]byte
				binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
				data.Write(b[:])
			}
		case tensor.F64:
			for _, v := range t.Storage().F64() {
				var b [8]byte
				binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
				data.Write(b[:])
			}
		case tensor.F16, tensor.BF16:
			// 16-bit floats are kept as their raw uint16 bits; write them verbatim.
			for _, bits := range t.Storage().U16() {
				var b [2]byte
				binary.LittleEndian.PutUint16(b[:], bits)
				data.Write(b[:])
			}
		}
		offset += size
	}

	hj, err := json.Marshal(header) // Go maps marshal key-sorted → deterministic
	if err != nil {
		return fmt.Errorf("safetensors: marshal header: %w", err)
	}
	if pad := (8 - (len(hj) % 8)) % 8; pad > 0 {
		hj = append(hj, bytes.Repeat([]byte{' '}, pad)...)
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(hj))); err != nil {
		return err
	}
	if _, err := w.Write(hj); err != nil {
		return err
	}
	_, err = w.Write(data.Bytes())
	return err
}

// SaveFile writes tensors to path.
func SaveFile(path string, tensors map[string]*tensor.Tensor, meta map[string]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Save(f, tensors, meta); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Load reads all tensors and metadata from r.
func Load(r io.Reader) (map[string]*tensor.Tensor, map[string]string, error) {
	var hlen uint64
	if err := binary.Read(r, binary.LittleEndian, &hlen); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	if hlen > maxHeaderSize {
		return nil, nil, fmt.Errorf("safetensors: header size %d exceeds cap %d", hlen, maxHeaderSize)
	}
	hbuf := make([]byte, hlen)
	if _, err := io.ReadFull(r, hbuf); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hbuf, &raw); err != nil {
		return nil, nil, fmt.Errorf("safetensors: parse header: %w", err)
	}
	var meta map[string]string
	if m, ok := raw["__metadata__"]; ok {
		if err := json.Unmarshal(m, &meta); err != nil {
			return nil, nil, fmt.Errorf("safetensors: parse __metadata__: %w", err)
		}
		delete(raw, "__metadata__")
	}

	data, err := io.ReadAll(r)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: read data: %w", err)
	}

	// Parse entries and validate strictly: sizes match shape·dtype, and the
	// offsets tile the data section exactly (no gaps, no overlaps).
	type namedEntry struct {
		name string
		e    entry
	}
	entries := make([]namedEntry, 0, len(raw))
	for name, msg := range raw {
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		entries = append(entries, namedEntry{name, e})
	}
	// Sort by (begin, end): zero-size tensors share their begin with the next
	// tensor's, so the end tie-break keeps them ahead of it (§B21).
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i].e.DataOffsets, entries[j].e.DataOffsets
		if a[0] != b[0] {
			return a[0] < b[0]
		}
		return a[1] < b[1]
	})

	out := make(map[string]*tensor.Tensor, len(entries))
	cursor := 0
	for _, ne := range entries {
		e := ne.e
		dt, err := dtypeOf(e.Dtype)
		if err != nil {
			return nil, nil, fmt.Errorf("%w (tensor %q)", err, ne.name)
		}
		shape := tensor.Shape(e.Shape)
		if !shape.IsValid() {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: invalid shape %v", ne.name, e.Shape)
		}
		// cap dims/product so Numel cannot wrap and lie to the size check (§V15)
		const maxDim = 1 << 40
		numel := uint64(1)
		for _, dv := range shape {
			if uint64(dv) > maxDim {
				return nil, nil, fmt.Errorf("safetensors: tensor %q dim %d exceeds cap", ne.name, dv)
			}
			numel *= uint64(dv)
			if numel > maxDim {
				return nil, nil, fmt.Errorf("safetensors: tensor %q element count exceeds cap", ne.name)
			}
		}
		begin, end := e.DataOffsets[0], e.DataOffsets[1]
		want := shape.Numel() * dt.size
		if begin != cursor {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: offset %d leaves gap/overlap at %d", ne.name, begin, cursor)
		}
		if end-begin != want {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %d bytes ≠ shape %v × %s", ne.name, end-begin, shape, e.Dtype)
		}
		if end > len(data) {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: offsets [%d,%d) beyond data %d", ne.name, begin, end, len(data))
		}
		t := tensor.New(dt.out, shape)
		switch e.Dtype {
		case "F32":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[begin+i*4:]))
			}
		case "F64":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[begin+i*8:]))
			}
		case "F16", "BF16":
			dst := t.Storage().U16() // raw 16-bit bits, verbatim
			for i := range dst {
				dst[i] = binary.LittleEndian.Uint16(data[begin+i*2:])
			}
		case "F8_E4M3": // §T577: FP8 and integer dtypes widen on load
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = f8e4m3ToF32(data[begin+i])
			}
		case "F8_E5M2":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = f8e5m2ToF32(data[begin+i])
			}
		case "BOOL":
			dst := t.Storage().F32()
			for i := range dst {
				if data[begin+i] != 0 {
					dst[i] = 1
				}
			}
		case "I8":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = float32(int8(data[begin+i]))
			}
		case "U8":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = float32(data[begin+i])
			}
		case "I16":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = float32(int16(binary.LittleEndian.Uint16(data[begin+i*2:])))
			}
		case "U16":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = float32(binary.LittleEndian.Uint16(data[begin+i*2:]))
			}
		case "I32":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(int32(binary.LittleEndian.Uint32(data[begin+i*4:])))
			}
		case "U32":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(binary.LittleEndian.Uint32(data[begin+i*4:]))
			}
		case "I64":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(int64(binary.LittleEndian.Uint64(data[begin+i*8:])))
			}
		case "U64":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(binary.LittleEndian.Uint64(data[begin+i*8:]))
			}
		}
		out[ne.name] = t
		cursor = end
	}
	if cursor != len(data) {
		return nil, nil, fmt.Errorf("safetensors: %d trailing data bytes not covered by offsets", len(data)-cursor)
	}
	return out, meta, nil
}

// LoadFile reads a .safetensors file.
func LoadFile(path string) (map[string]*tensor.Tensor, map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	return Load(f)
}
