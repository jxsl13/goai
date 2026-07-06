// Package npy is a minimal Pure-Go reader/writer for NumPy .npy files, used to
// exchange large golden tensors with the Python reference generator (§T10, §V1).
// It supports little-endian f32 (<f4) and f64 (<f8) in C order — enough for
// goldens; other dtypes/orders return an error rather than silently mis-reading.
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

var magic = []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}

// Read parses a .npy stream into a contiguous tensor.
func Read(r io.Reader) (*tensor.Tensor, error) {
	br := bufio.NewReader(r)
	head := make([]byte, 8)
	if _, err := io.ReadFull(br, head); err != nil {
		return nil, fmt.Errorf("npy: read magic: %w", err)
	}
	if string(head[:6]) != string(magic) {
		return nil, fmt.Errorf("npy: bad magic")
	}
	major := head[6]

	var hlen int
	switch major {
	case 1:
		var l uint16
		if err := binary.Read(br, binary.LittleEndian, &l); err != nil {
			return nil, err
		}
		hlen = int(l)
	case 2, 3:
		var l uint32
		if err := binary.Read(br, binary.LittleEndian, &l); err != nil {
			return nil, err
		}
		hlen = int(l)
	default:
		return nil, fmt.Errorf("npy: unsupported version %d", major)
	}

	// §V15: cap the header length before allocating. A hostile v2/v3 file can
	// claim a ~4 GB header (uint32) → make([]byte, hlen) OOM-crashes the process.
	// 64 MiB is far above any real .npy header.
	const maxHeader = 64 << 20
	if hlen < 0 || hlen > maxHeader {
		return nil, fmt.Errorf("npy: header length %d exceeds cap", hlen)
	}
	hbuf := make([]byte, hlen)
	if _, err := io.ReadFull(br, hbuf); err != nil {
		return nil, fmt.Errorf("npy: read header: %w", err)
	}
	dtype, shape, err := parseHeader(string(hbuf))
	if err != nil {
		return nil, err
	}
	// §V15: hostile shapes must error, never OOM. Read() pre-allocates the full
	// tensor + data buffer from the CLAIMED shape (unlike safetensors/gguf, which
	// slice from already-read bytes and are bounded by the offset check), so the
	// element count is capped low enough that a tiny malicious file cannot force
	// a multi-GB allocation. 32M elems = 256 MB at f64 — ample for golden data.
	const maxDim = 1 << 40
	const maxNumel = 1 << 25
	numel := uint64(1)
	for _, d := range shape {
		if d < 0 || uint64(d) > maxDim {
			return nil, fmt.Errorf("npy: dim %d out of range", d)
		}
		numel *= uint64(d)
		if numel > maxNumel {
			return nil, fmt.Errorf("npy: element count %d exceeds cap %d", numel, maxNumel)
		}
	}

	t := tensor.New(dtype, shape)
	n := shape.Numel()
	buf := make([]byte, n*dtype.Size())
	if _, err := io.ReadFull(br, buf); err != nil {
		return nil, fmt.Errorf("npy: read data: %w", err)
	}
	switch dtype {
	case tensor.F32:
		dst := t.Storage().F32()
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case tensor.F64:
		dst := t.Storage().F64()
		for i := range dst {
			dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
		}
	}
	return t, nil
}

// ReadFile reads a .npy file into a tensor.
func ReadFile(path string) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

// parseHeader extracts dtype and shape from the header dict, rejecting
// fortran_order and unsupported descriptors.
func parseHeader(h string) (tensor.Dtype, tensor.Shape, error) {
	descr, err := field(h, "descr")
	if err != nil {
		return tensor.Invalid, nil, err
	}
	var dtype tensor.Dtype
	switch strings.Trim(descr, "'\" ") {
	case "<f4", "|f4", "float32":
		dtype = tensor.F32
	case "<f8", "|f8", "float64":
		dtype = tensor.F64
	default:
		return tensor.Invalid, nil, fmt.Errorf("npy: unsupported descr %q", descr)
	}

	fo, err := field(h, "fortran_order")
	if err != nil {
		return tensor.Invalid, nil, err
	}
	if strings.Contains(fo, "True") {
		return tensor.Invalid, nil, fmt.Errorf("npy: fortran_order not supported")
	}

	shape, err := parseShape(h)
	if err != nil {
		return tensor.Invalid, nil, err
	}
	return dtype, shape, nil
}

// field returns the raw value text after 'key': up to the next comma.
func field(h, key string) (string, error) {
	_, rest, found := strings.Cut(h, "'"+key+"'")
	if !found {
		return "", fmt.Errorf("npy: header missing %s", key)
	}
	_, rest, found = strings.Cut(rest, ":")
	if !found {
		return "", fmt.Errorf("npy: malformed header at %s", key)
	}
	if before, _, ok := strings.Cut(rest, ","); ok {
		return strings.TrimSpace(before), nil
	}
	return strings.TrimSpace(rest), nil
}

// parseShape reads the 'shape': (a, b, ...) tuple.
func parseShape(h string) (tensor.Shape, error) {
	i := strings.Index(h, "'shape'")
	if i < 0 {
		return nil, fmt.Errorf("npy: header missing shape")
	}
	open := strings.IndexByte(h[i:], '(')
	cl := strings.IndexByte(h[i:], ')')
	if open < 0 || cl < 0 || cl < open {
		return nil, fmt.Errorf("npy: malformed shape")
	}
	inner := h[i+open+1 : i+cl]
	inner = strings.TrimSpace(inner)
	if inner == "" {
		return tensor.Shape{}, nil // scalar
	}
	var shape tensor.Shape
	for part := range strings.SplitSeq(inner, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		d, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("npy: bad shape dim %q", part)
		}
		shape = append(shape, d)
	}
	return shape, nil
}

// Write serializes a contiguous tensor to w in .npy v1.0 format (C order).
func Write(w io.Writer, t *tensor.Tensor) error {
	t = t.Contiguous()
	descr := map[tensor.Dtype]string{tensor.F32: "<f4", tensor.F64: "<f8"}[t.Dtype()]
	if descr == "" {
		return fmt.Errorf("npy: unsupported dtype %v", t.Dtype())
	}

	var sb strings.Builder
	sb.WriteString("(")
	for i, d := range t.Shape() {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(strconv.Itoa(d))
	}
	if t.Ndim() == 1 {
		sb.WriteString(",")
	}
	sb.WriteString(")")
	header := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, sb.String())

	// Pad so that 10 (preamble) + len(header)+1(newline) is a multiple of 64.
	total := 10 + len(header) + 1
	pad := (64 - total%64) % 64
	header += strings.Repeat(" ", pad) + "\n"

	if _, err := w.Write(magic); err != nil {
		return err
	}
	if _, err := w.Write([]byte{1, 0}); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint16(len(header))); err != nil {
		return err
	}
	if _, err := io.WriteString(w, header); err != nil {
		return err
	}

	n := t.Numel()
	switch t.Dtype() {
	case tensor.F32:
		src := t.Storage().F32()
		b := make([]byte, 4*n)
		for i, v := range src {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
		}
		_, err := w.Write(b)
		return err
	case tensor.F64:
		src := t.Storage().F64()
		b := make([]byte, 8*n)
		for i, v := range src {
			binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v))
		}
		_, err := w.Write(b)
		return err
	}
	return nil
}

// WriteFile writes t to a .npy file.
func WriteFile(path string, t *tensor.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Write(f, t); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
