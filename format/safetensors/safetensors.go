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
	"unsafe"

	"github.com/jxsl13/goai/tensor"
)

// nativeLittleEndian reports whether the host stores multi-byte scalars
// little-endian. The safetensors data section is little-endian on disk, so on
// such hosts (every platform GoAI targets) the raw bytes already match the
// in-memory layout of an F32/F64/F16/BF16 tensor and loading a verbatim-bit
// dtype is a single bulk copy rather than a per-element decode. Big-endian
// hosts fall back to the element-wise path, so the result is identical either way.
var nativeLittleEndian = func() bool {
	var x uint16 = 1
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// rawCopyLE bulk-copies verbatim little-endian source bytes into the backing
// store of a numeric destination slice, collapsing a per-element decode loop
// into one memmove. elemSize is the byte width of one T. It is a no-op that
// returns false on a big-endian host or an empty slice, signalling the caller
// to fall back to the element-wise decode (correctness on any byte order).
func rawCopyLE[T any](dst []T, src []byte, elemSize int) bool {
	if !nativeLittleEndian || len(dst) == 0 {
		return false
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize), src)
	return true
}

// rawStoreLE is the write-side mirror of rawCopyLE: it bulk-copies the backing
// bytes of a verbatim-bit numeric slice into a little-endian byte buffer,
// collapsing a per-element PutUintN loop into one memmove. It returns false on
// a big-endian host or an empty slice so the caller falls back to element-wise
// encoding (identical bytes on any host).
func rawStoreLE[T any](dst []byte, src []T, elemSize int) bool {
	if !nativeLittleEndian || len(src) == 0 {
		return false
	}
	copy(dst, unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*elemSize))
	return true
}

// verbatimStorageBytes returns the raw backing bytes of t's storage for the
// verbatim-bit dtypes (F32/F64/F16/BF16), whose on-disk little-endian layout
// equals their in-memory layout, so the data section streams straight in with
// no decode. It returns nil for the widening dtypes (FP8/int/bool), which must
// decode element-wise, and a non-nil empty slice for a zero-element tensor. The
// returned slice aliases the tensor's storage, so it is only safe to fill while
// t is being constructed (as Load does).
func verbatimStorageBytes(t *tensor.Tensor, dtype string) []byte {
	switch dtype {
	case "F32":
		s := t.Storage().F32()
		if len(s) == 0 {
			return []byte{}
		}
		return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
	case "F64":
		s := t.Storage().F64()
		if len(s) == 0 {
			return []byte{}
		}
		return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*8)
	case "F16", "BF16":
		s := t.Storage().U16()
		if len(s) == 0 {
			return []byte{}
		}
		return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*2)
	}
	return nil
}

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

	// Pass 1: contiguize once, compute offsets, build the header. Pass 2 then
	// streams each tensor's bytes through a fixed scratch chunk directly to w —
	// no full-size intermediate buffer and no per-element bytes.Buffer.Write
	// call (the old inner loop paid a Write() call per 2–8 byte element;
	// bulk-encoding is ~2.5× faster, docs/perf-notes-lowlevel.md).
	header := make(map[string]any, len(tensors)+1)
	if len(meta) > 0 {
		header["__metadata__"] = meta
	}
	contig := make([]*tensor.Tensor, len(names))
	offset, maxSize := 0, 0
	for i, n := range names {
		t := tensors[n].Contiguous()
		contig[i] = t
		dn, err := dtypeName(t.Dtype())
		if err != nil {
			return fmt.Errorf("%w (tensor %q)", err, n)
		}
		size := t.Numel() * t.Dtype().Size()
		header[n] = entry{Dtype: dn, Shape: append([]int{}, t.Shape()...), DataOffsets: [2]int{offset, offset + size}}
		offset += size
		maxSize = max(maxSize, size)
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

	const chunkBytes = 256 << 10
	var b []byte
	if maxSize > 0 {
		b = make([]byte, min(maxSize, chunkBytes))
	}
	for _, t := range contig {
		switch t.Dtype() {
		case tensor.F32:
			src := t.Storage().F32()
			for len(src) > 0 {
				m := min(len(src), len(b)/4)
				c := b[:4*m]
				if !rawStoreLE(c, src[:m], 4) {
					for i, v := range src[:m] {
						binary.LittleEndian.PutUint32(c[i*4:], math.Float32bits(v))
					}
				}
				if _, err := w.Write(c); err != nil {
					return err
				}
				src = src[m:]
			}
		case tensor.F64:
			src := t.Storage().F64()
			for len(src) > 0 {
				m := min(len(src), len(b)/8)
				c := b[:8*m]
				if !rawStoreLE(c, src[:m], 8) {
					for i, v := range src[:m] {
						binary.LittleEndian.PutUint64(c[i*8:], math.Float64bits(v))
					}
				}
				if _, err := w.Write(c); err != nil {
					return err
				}
				src = src[m:]
			}
		case tensor.F16, tensor.BF16:
			// 16-bit floats are kept as their raw uint16 bits; write them verbatim.
			src := t.Storage().U16()
			for len(src) > 0 {
				m := min(len(src), len(b)/2)
				c := b[:2*m]
				if !rawStoreLE(c, src[:m], 2) {
					for i, bits := range src[:m] {
						binary.LittleEndian.PutUint16(c[i*2:], bits)
					}
				}
				if _, err := w.Write(c); err != nil {
					return err
				}
				src = src[m:]
			}
		}
	}
	return nil
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
// Load reads safetensors from a stream. The stream length is unknown, so a
// hostile header claiming huge data_offsets is bounded only by the per-dim cap;
// prefer [LoadFile], which cross-checks the declared offsets against the real
// file size and refuses to allocate a tensor the file cannot contain.
func Load(r io.Reader) (map[string]*tensor.Tensor, map[string]string, error) {
	return loadFrom(r, -1)
}

// loadFrom is the core reader. fileSize is the underlying file's total size in
// bytes, or -1 for a stream of unknown length. When known, a tensor's declared
// data-offset end may not exceed the file's payload capacity, so an 80-byte file
// claiming a 1 GiB tensor errors before tensor.New rather than allocating it from
// untrusted input (§B-DoS).
func loadFrom(r io.Reader, fileSize int64) (map[string]*tensor.Tensor, map[string]string, error) {
	var hlen uint64
	if err := binary.Read(r, binary.LittleEndian, &hlen); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	if hlen > maxHeaderSize {
		return nil, nil, fmt.Errorf("safetensors: header size %d exceeds cap %d", hlen, maxHeaderSize)
	}
	// Payload begins after the 8-byte length field and the header. When the file
	// size is known, no tensor's declared end offset may reach past it.
	availPayload := int64(-1)
	if fileSize >= 0 {
		availPayload = fileSize - 8 - int64(hlen)
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

	// Parse entries and validate strictly: sizes match shape·dtype, and the
	// offsets tile the data section exactly (no gaps, no overlaps). The data
	// section is then STREAMED — each tensor is read straight from r into its
	// storage in offset order, so a verbatim-bit tensor never lands in an
	// intermediate buffer (no io.ReadAll copy; §T723). Validation is done from
	// the header offsets alone (below), so it does not need the data materialized.
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
	var buf []byte // reused scratch for the decode fallback (widening / big-endian)
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
		// cap dims/product so Numel cannot wrap and lie to the size check (§V15).
		// The product guard divides BEFORE multiplying: the old post-multiply
		// check itself wrapped uint64 (shape [2^40, 2^40] → 2^80 ≡ 0), letting a
		// hostile header pass the cap and return a tensor whose shape claims
		// 2^80 elements over empty storage.
		const maxDim = 1 << 40
		numel := uint64(1)
		for _, dv := range shape {
			if uint64(dv) > maxDim {
				return nil, nil, fmt.Errorf("safetensors: tensor %q dim %d exceeds cap", ne.name, dv)
			}
			if dv != 0 && numel > maxDim/uint64(dv) {
				return nil, nil, fmt.Errorf("safetensors: tensor %q element count exceeds cap", ne.name)
			}
			numel *= uint64(dv)
		}
		begin, end := e.DataOffsets[0], e.DataOffsets[1]
		want := shape.Numel() * dt.size
		if begin != cursor {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: offset %d leaves gap/overlap at %d", ne.name, begin, cursor)
		}
		if end-begin != want {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %d bytes ≠ shape %v × %s", ne.name, end-begin, shape, e.Dtype)
		}
		// DoS guard: when the file size is known, this tensor's declared data
		// cannot extend past the file. Without it a tiny file with a huge declared
		// tensor allocates the whole tensor (measured: 1 GiB from a 79-byte file)
		// before the read discovers the EOF.
		if availPayload >= 0 && int64(end) > availPayload {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: data ends at %d but the file holds only %d bytes of payload — refusing to allocate from a malformed file", ne.name, end, max(availPayload, 0))
		}
		t := tensor.New(dt.out, shape)
		// Fast path: a verbatim-bit dtype on a little-endian host streams straight
		// into the tensor's storage — one copy from r, no decode, no intermediate
		// buffer. A short stream is caught here as io.ErrUnexpectedEOF (this is the
		// streaming replacement for the old end>len(data) bounds check).
		if bv := verbatimStorageBytes(t, e.Dtype); bv != nil && nativeLittleEndian {
			if _, err := io.ReadFull(r, bv); err != nil {
				return nil, nil, fmt.Errorf("safetensors: tensor %q: read data: %w", ne.name, err)
			}
			out[ne.name] = t
			cursor = end
			continue
		}
		// Fallback: read this tensor's bytes into reusable scratch, then decode —
		// the widening dtypes (FP8/int/bool), or verbatim bits on a big-endian host.
		if cap(buf) < want {
			buf = make([]byte, want)
		}
		src := buf[:want]
		if _, err := io.ReadFull(r, src); err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: read data: %w", ne.name, err)
		}
		switch e.Dtype {
		case "F32":
			dst := t.Storage().F32()
			if !rawCopyLE(dst, src, 4) {
				for i := range dst {
					dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
				}
			}
		case "F64":
			dst := t.Storage().F64()
			if !rawCopyLE(dst, src, 8) {
				for i := range dst {
					dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(src[i*8:]))
				}
			}
		case "F16", "BF16":
			dst := t.Storage().U16() // raw 16-bit bits, verbatim
			if !rawCopyLE(dst, src, 2) {
				for i := range dst {
					dst[i] = binary.LittleEndian.Uint16(src[i*2:])
				}
			}
		case "F8_E4M3": // §T577: FP8 and integer dtypes widen on load
			dst := t.Storage().F32()
			for i, b := range src {
				dst[i] = f8e4m3Table[b]
			}
		case "F8_E5M2":
			dst := t.Storage().F32()
			for i, b := range src {
				dst[i] = f8e5m2Table[b]
			}
		case "BOOL":
			dst := t.Storage().F32()
			for i, b := range src {
				if b != 0 {
					dst[i] = 1
				}
			}
		case "I8":
			dst := t.Storage().F32()
			for i, b := range src {
				dst[i] = float32(int8(b))
			}
		case "U8":
			dst := t.Storage().F32()
			for i, b := range src {
				dst[i] = float32(b)
			}
		case "I16":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = float32(int16(binary.LittleEndian.Uint16(src[i*2:])))
			}
		case "U16":
			dst := t.Storage().F32()
			for i := range dst {
				dst[i] = float32(binary.LittleEndian.Uint16(src[i*2:]))
			}
		case "I32":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(int32(binary.LittleEndian.Uint32(src[i*4:])))
			}
		case "U32":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(binary.LittleEndian.Uint32(src[i*4:]))
			}
		case "I64":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(int64(binary.LittleEndian.Uint64(src[i*8:])))
			}
		case "U64":
			dst := t.Storage().F64()
			for i := range dst {
				dst[i] = float64(binary.LittleEndian.Uint64(src[i*8:]))
			}
		}
		out[ne.name] = t
		cursor = end
	}
	// No bytes may follow the last tensor: the offsets must tile the data section
	// exactly. Probe for a trailing byte (streaming replacement for the old
	// cursor!=len(data) check).
	var probe [1]byte
	if n, _ := io.ReadFull(r, probe[:]); n > 0 {
		return nil, nil, fmt.Errorf("safetensors: trailing data after last tensor at offset %d", cursor)
	}
	return out, meta, nil
}

// LoadFile reads a single .safetensors file. Sharded checkpoints — a
// model.safetensors.index.json plus model-00001-of-000NN.safetensors pieces,
// which is how Hugging Face ships every checkpoint above its ~5 GB shard
// limit — go through [LoadSharded] instead; LoadFile deliberately does not
// sniff for an index (explicitness over magic).
func LoadFile(path string) (map[string]*tensor.Tensor, map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	return loadFrom(f, fi.Size())
}
