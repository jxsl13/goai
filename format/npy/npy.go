// Package npy reads and writes single arrays in NumPy's .npy format (v1.0), the
// simplest numpy interchange format — a small ASCII header (dtype, shape,
// element order) followed by the raw C-order little-endian element bytes. It
// supports GoAI's F16/F32/F64 tensors (numpy '<f2'/'<f4'/'<f8'), enabling data
// exchange with numpy/Python (e.g. golden references) via byte-faithful
// round-trips; F16 is stored as its raw IEEE binary16 bits, bit-identical to
// numpy.float16. Format spec: numpy.lib.format (the definitional source; there
// is no paper, §V16).
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

	"github.com/jxsl13/goai/tensor"
)

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

func writeData(w io.Writer, t *tensor.Tensor) error {
	bw := bufio.NewWriter(w)
	switch t.Dtype() {
	case tensor.F32:
		var b [4]byte
		for _, v := range t.Storage().F32() {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			if _, err := bw.Write(b[:]); err != nil {
				return err
			}
		}
	case tensor.F64:
		var b [8]byte
		for _, v := range t.Storage().F64() {
			binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
			if _, err := bw.Write(b[:]); err != nil {
				return err
			}
		}
	case tensor.F16:
		var b [2]byte
		for _, bits := range t.Storage().U16() { // raw IEEE binary16 bits, verbatim
			binary.LittleEndian.PutUint16(b[:], bits)
			if _, err := bw.Write(b[:]); err != nil {
				return err
			}
		}
	}
	return bw.Flush()
}

// SaveFile writes t to path as a .npy file.
func SaveFile(path string, t *tensor.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Save(f, t); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Load reads one array from r (a .npy stream). Only C-order, little-endian
// F16/F32/F64 arrays are supported; malformed input returns an error and never panics.
func Load(r io.Reader) (*tensor.Tensor, error) {
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
	numel := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("npy: negative dimension in shape %v", shape)
		}
		numel *= d
		if numel > maxElems {
			return nil, fmt.Errorf("npy: array too large (%v)", shape)
		}
	}
	out := tensor.New(dtype, shape)
	if err := readData(r, out, numel); err != nil {
		return nil, err
	}
	return out, nil
}

func readData(r io.Reader, out *tensor.Tensor, numel int) error {
	buf := make([]byte, numel*out.Dtype().Size())
	if _, err := io.ReadFull(r, buf); err != nil {
		return fmt.Errorf("npy: reading data: %w", err)
	}
	switch out.Dtype() {
	case tensor.F32:
		s := out.Storage().F32()
		for i := range s {
			s[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case tensor.F64:
		s := out.Storage().F64()
		for i := range s {
			s[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
		}
	case tensor.F16:
		s := out.Storage().U16() // raw IEEE binary16 bits, verbatim
		for i := range s {
			s[i] = binary.LittleEndian.Uint16(buf[i*2:])
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
	return Load(bufio.NewReader(f))
}
