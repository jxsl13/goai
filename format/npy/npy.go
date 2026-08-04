// Package npy reads and writes single arrays in NumPy's .npy format (v1.0), the
// simplest numpy interchange format — a small ASCII header (dtype, shape,
// element order) followed by the raw C-order little-endian element bytes. It
// supports GoAI's F16/F32/F64 tensors (numpy '<f2'/'<f4'/'<f8'), enabling data
// exchange with numpy/Python (e.g. golden references) via byte-faithful
// round-trips; F16 is stored as its raw IEEE binary16 bits, bit-identical to
// numpy.float16. Format spec: numpy.lib.format (the definitional source; there
// is no paper, §V16).
//
// In plain terms: .npy is NumPy's simple one-array file format — a small header describing shape and number type, then the raw values.
//
// Further reading: the NumPy .npy format specification (numpy/numpy, NEP 1) — the defining reference (file formats have no paper, SPEC V16).
package npy

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"unsafe"

	"github.com/jxsl13/goai/tensor"
)

// nativeLittleEndian reports whether the host stores multi-byte scalars
// little-endian. The npy data section is little-endian on disk (this package
// only handles '<' byte order), so on such hosts (every platform GoAI targets)
// the raw bytes already match a tensor's in-memory F32/F64/F16 layout and the
// copy is one memmove instead of a per-element decode/encode loop. Big-endian
// hosts fall back to the element-wise path, so the bytes are identical either way.
var nativeLittleEndian = func() bool {
	var x uint16 = 1
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// rawCopyLE bulk-copies verbatim little-endian source bytes into a numeric
// slice's backing store (read side); rawStoreLE is the write-side mirror. Both
// return false on a big-endian host or empty slice so the caller falls back to
// the element-wise path.
func rawCopyLE[T any](dst []T, src []byte, elemSize int) bool {
	if !nativeLittleEndian || len(dst) == 0 {
		return false
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize), src)
	return true
}

func rawStoreLE[T any](dst []byte, src []T, elemSize int) bool {
	if !nativeLittleEndian || len(src) == 0 {
		return false
	}
	copy(dst, unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*elemSize))
	return true
}

const (
	maxHeaderLen = 1 << 20 // reject absurd header lengths (fuzz safety)
	maxElems     = 1 << 30 // reject absurd shapes before allocating (fuzz safety)
)

var magic = []byte("\x93NUMPY")

// descrFor maps a GoAI dtype to a little-endian numpy type descriptor.
func descrFor(d tensor.Dtype) (string, error) {
	switch d {
	case tensor.F32:
		return "<f4", nil
	case tensor.F64:
		return "<f8", nil
	case tensor.F16:
		return "<f2", nil // numpy.float16 (IEEE binary16)
	default:
		// BF16 has no standard numpy descriptor (it needs the ml_dtypes extension), and
		// integer/bool types await GoAI integer dtypes (§C4).
		return "", fmt.Errorf("npy: unsupported dtype %v (only F16/F32/F64)", d)
	}
}

// dtypeFromDescr maps a numpy descriptor to a GoAI dtype, accepting little-endian
// or native byte order (big-endian is rejected).
func dtypeFromDescr(descr string) (tensor.Dtype, error) {
	if descr == "" {
		return 0, fmt.Errorf("npy: empty descr")
	}
	switch descr[0] {
	case '<', '=', '|': // little-endian, native (LE host), or not-applicable
	case '>':
		return 0, fmt.Errorf("npy: big-endian arrays (%q) not supported", descr)
	default:
		return 0, fmt.Errorf("npy: unrecognized byte order in descr %q", descr)
	}
	switch descr[1:] {
	case "f2":
		return tensor.F16, nil // numpy.float16
	case "f4":
		return tensor.F32, nil
	case "f8":
		return tensor.F64, nil
	default:
		return 0, fmt.Errorf("npy: unsupported descr %q (only f2/f4/f8)", descr)
	}
}

// shapeTuple formats a shape as a Python tuple literal: () / (5,) / (2, 3).
func shapeTuple(s tensor.Shape) string {
	switch len(s) {
	case 0:
		return "()"
	case 1:
		return "(" + strconv.Itoa(s[0]) + ",)"
	default:
		parts := make([]string, len(s))
		for i, d := range s {
			parts[i] = strconv.Itoa(d)
		}
		return "(" + strings.Join(parts, ", ") + ")"
	}
}

// Save writes t to w as a NumPy .npy (version 1.0, C-order, little-endian).
func Save(w io.Writer, t *tensor.Tensor) error {
	t = t.Contiguous()
	descr, err := descrFor(t.Dtype())
	if err != nil {
		return err
	}
	dict := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shapeTuple(t.Shape()))
	// numpy pads the header (magic+version+len field+header) to a multiple of 64,
	// the header string ending with '\n'. Preamble = 6 magic + 2 version + 2 len.
	const preamble = 10
	pad := (64 - (preamble+len(dict)+1)%64) % 64
	header := dict + strings.Repeat(" ", pad) + "\n"

	if _, err := w.Write(magic); err != nil {
		return err
	}
	if _, err := w.Write([]byte{1, 0}); err != nil { // version 1.0
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint16(len(header))); err != nil {
		return err
	}
	if _, err := io.WriteString(w, header); err != nil {
		return err
	}
	return writeData(w, t)
}

// writeData bulk-encodes the element bytes through a fixed 256 KiB scratch
// chunk — one Write per chunk instead of one bufio.Write call per 2–8-byte
// element (docs/perf-notes-lowlevel.md).
func writeData(w io.Writer, t *tensor.Tensor) error {
	const chunkBytes = 256 << 10
	esize := t.Dtype().Size()
	b := make([]byte, min(t.Numel()*esize, chunkBytes))
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
	case tensor.F16:
		src := t.Storage().U16() // raw IEEE binary16 bits, verbatim
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
	return nil
}

// SaveFile writes t to path as a .npy file.
func SaveFile(path string, t *tensor.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	//perfscan:ignore PS3029 .npy Save serialization, one-time IO
	if err := Save(f, t); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Load reads one array from r (a .npy stream). Only C-order, little-endian
// F16/F32/F64 arrays are supported; malformed input returns an error and never panics.
// Load reads an .npy array from a stream. The stream's total size is unknown, so
// a hostile header claiming a huge shape is bounded only by maxElems; prefer
// [LoadFile], which cross-checks the declared payload against the real file size
// and refuses to allocate more than the file can contain.
func Load(r io.Reader) (*tensor.Tensor, error) {
	return loadFrom(r, -1)
}

// loadFrom is the core reader. fileSize is the total size of the underlying file
// in bytes, or -1 when reading a stream of unknown length. When known, it caps
// the pre-allocation: a header may not declare more payload than the file holds
// after the header, so an 80-byte file claiming an 8 TiB array errors before
// tensor.New instead of triggering a multi-gigabyte allocation (or an
// unrecoverable out-of-memory make) from untrusted input (§B-DoS).
func loadFrom(r io.Reader, fileSize int64) (*tensor.Tensor, error) {
	var head [8]byte
	if _, err := io.ReadFull(r, head[:]); err != nil {
		return nil, fmt.Errorf("npy: reading magic/version: %w", err)
	}
	if string(head[:6]) != string(magic) {
		return nil, fmt.Errorf("npy: bad magic %q", head[:6])
	}
	major := head[6]

	var hlen int
	switch major {
	case 1:
		var h uint16
		if err := binary.Read(r, binary.LittleEndian, &h); err != nil {
			return nil, err
		}
		hlen = int(h)
	case 2, 3:
		var h uint32
		if err := binary.Read(r, binary.LittleEndian, &h); err != nil {
			return nil, err
		}
		hlen = int(h)
	default:
		return nil, fmt.Errorf("npy: unsupported version %d.%d", head[6], head[7])
	}
	if hlen <= 0 || hlen > maxHeaderLen {
		return nil, fmt.Errorf("npy: header length %d out of range", hlen)
	}
	hbuf := make([]byte, hlen)
	if _, err := io.ReadFull(r, hbuf); err != nil {
		return nil, fmt.Errorf("npy: reading header: %w", err)
	}

	descr, fortran, shape, err := parseHeader(string(hbuf))
	if err != nil {
		return nil, err
	}
	if fortran {
		return nil, fmt.Errorf("npy: fortran_order arrays not supported")
	}
	dtype, err := dtypeFromDescr(descr)
	if err != nil {
		return nil, err
	}
	// Overflow-safe product: guard with division BEFORE multiplying. A
	// post-multiply `numel > maxElems` check itself wraps int64 — shape
	// (2^30, 2^34) gives exactly 2^64 ≡ 0 — and would let a hostile header
	// pass the cap with a tensor claiming 2^64 elements over empty storage.
	numel := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("npy: negative dimension in shape %v", shape)
		}
		if d != 0 && numel > maxElems/d {
			return nil, fmt.Errorf("npy: array too large (%v)", shape)
		}
		numel *= d
	}
	// DoS guard: when the file size is known, the header may not declare more
	// payload than the file actually contains after the header. Without this a
	// tiny file with a huge declared shape allocates the whole tensor up front
	// (measured: 1 GiB from a 138-byte file) before readData discovers the EOF.
	if fileSize >= 0 {
		hdrField := int64(2) // v1 uses a uint16 header-length field
		if major != 1 {
			hdrField = 4 // v2/v3 use uint32
		}
		payloadStart := 8 + hdrField + int64(hlen)
		avail := fileSize - payloadStart
		if need := int64(numel) * int64(dtype.Size()); avail < 0 || need > avail {
			return nil, fmt.Errorf("npy: header declares %d bytes of data but the file holds only %d after the header — refusing to allocate from a malformed file", need, max(avail, 0))
		}
	}
	out := tensor.New(dtype, shape)
	if err := readData(r, out, numel); err != nil {
		return nil, err
	}
	return out, nil
}

// readData streams the payload through a fixed 256 KiB scratch chunk instead
// of allocating a payload-sized intermediate buffer (up to 8 GB at the
// maxElems cap) — time-neutral, halves transient memory
// (docs/perf-notes-lowlevel.md).
func readData(r io.Reader, out *tensor.Tensor, numel int) error {
	const chunkBytes = 256 << 10
	buf := make([]byte, min(numel*out.Dtype().Size(), chunkBytes))
	switch out.Dtype() {
	case tensor.F32:
		s := out.Storage().F32()
		for len(s) > 0 {
			m := min(len(s), len(buf)/4)
			c := buf[:4*m]
			if _, err := io.ReadFull(r, c); err != nil {
				return fmt.Errorf("npy: reading data: %w", err)
			}
			if !rawCopyLE(s[:m], c, 4) {
				for i := range m {
					s[i] = math.Float32frombits(binary.LittleEndian.Uint32(c[i*4:]))
				}
			}
			s = s[m:]
		}
	case tensor.F64:
		s := out.Storage().F64()
		for len(s) > 0 {
			m := min(len(s), len(buf)/8)
			c := buf[:8*m]
			if _, err := io.ReadFull(r, c); err != nil {
				return fmt.Errorf("npy: reading data: %w", err)
			}
			if !rawCopyLE(s[:m], c, 8) {
				for i := range m {
					s[i] = math.Float64frombits(binary.LittleEndian.Uint64(c[i*8:]))
				}
			}
			s = s[m:]
		}
	case tensor.F16:
		s := out.Storage().U16() // raw IEEE binary16 bits, verbatim
		for len(s) > 0 {
			m := min(len(s), len(buf)/2)
			c := buf[:2*m]
			if _, err := io.ReadFull(r, c); err != nil {
				return fmt.Errorf("npy: reading data: %w", err)
			}
			if !rawCopyLE(s[:m], c, 2) {
				for i := range m {
					s[i] = binary.LittleEndian.Uint16(c[i*2:])
				}
			}
			s = s[m:]
		}
	}
	return nil
}

// parseHeader extracts descr, fortran_order and shape from a .npy header dict.
func parseHeader(h string) (descr string, fortran bool, shape tensor.Shape, err error) {
	descr, err = quotedField(h, "'descr'")
	if err != nil {
		return "", false, nil, err
	}
	switch {
	case strings.Contains(h, "'fortran_order': True"):
		fortran = true
	case strings.Contains(h, "'fortran_order': False"):
		fortran = false
	default:
		return "", false, nil, fmt.Errorf("npy: missing fortran_order in header")
	}
	shape, err = parseShape(h)
	return descr, fortran, shape, err
}

// quotedField returns the single-quoted string value following key in the header.
func quotedField(h, key string) (string, error) {
	i := strings.Index(h, key)
	if i < 0 {
		return "", fmt.Errorf("npy: missing %s in header", key)
	}
	rest := h[i+len(key):]
	a := strings.IndexByte(rest, '\'')
	if a < 0 {
		return "", fmt.Errorf("npy: malformed %s in header", key)
	}
	rest = rest[a+1:]
	b := strings.IndexByte(rest, '\'')
	if b < 0 {
		return "", fmt.Errorf("npy: unterminated %s in header", key)
	}
	return rest[:b], nil
}

// parseShape parses the 'shape': (...) tuple into a Shape.
func parseShape(h string) (tensor.Shape, error) {
	i := strings.Index(h, "'shape'")
	if i < 0 {
		return nil, fmt.Errorf("npy: missing shape in header")
	}
	rest := h[i:]
	open := strings.IndexByte(rest, '(')
	closeIdx := strings.IndexByte(rest, ')')
	if open < 0 || closeIdx < 0 || closeIdx < open {
		return nil, fmt.Errorf("npy: malformed shape tuple")
	}
	inner := strings.TrimSpace(rest[open+1 : closeIdx])
	if inner == "" {
		return tensor.Shape{}, nil // scalar ()
	}
	var shape tensor.Shape
	for _, part := range strings.Split(inner, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue // trailing comma of a 1-tuple
		}
		d, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("npy: bad shape entry %q", part)
		}
		shape = append(shape, d)
	}
	return shape, nil
}

// LoadFile reads a .npy array from path.
func LoadFile(path string) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return loadFrom(bufio.NewReader(f), fi.Size())
}
