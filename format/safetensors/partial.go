package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/jxsl13/goai/tensor"
)

// TensorInfo describes one tensor in a safetensors file as recorded in its
// header — dtype, shape and the size of its data section — without the data
// itself.
type TensorInfo struct {
	Name   string       // tensor name — the safetensors header key
	Dtype  tensor.Dtype // element type (F32/F64/F16/BF16/…)
	Shape  tensor.Shape // logical shape, row-major
	NBytes int64        // length of this tensor's byte range in the file
}

// Names reads only the header of a safetensors file and returns a descriptor for
// every tensor plus the file metadata, WITHOUT reading or allocating any tensor
// data. Memory and time are O(header), independent of the checkpoint size — so it
// is safe to point at a multi-gigabyte file just to see what is inside.
//
// The descriptors come back sorted by name for a stable listing. For the data of
// a single tensor use [LoadTensor]; for the whole file use [LoadFile].
func Names(path string) ([]TensorInfo, map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	_, raw, meta, err := readHeaderRaw(f)
	if err != nil {
		return nil, nil, err
	}
	infos := make([]TensorInfo, 0, len(raw))
	for name, msg := range raw {
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		dt, err := dtypeOf(e.Dtype)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		infos = append(infos, TensorInfo{
			Name:   name,
			Dtype:  dt.out,
			Shape:  tensor.Shape(e.Shape),
			NBytes: int64(e.DataOffsets[1] - e.DataOffsets[0]),
		})
	}
	sortInfos(infos)
	return infos, meta, nil
}

// LoadTensor loads exactly ONE tensor by name. It seeks straight to that tensor's
// byte range in the file, so no other tensor is read or allocated — the point is
// to pull a single weight out of a large checkpoint without materializing the
// rest (which [LoadFile] would). It returns an error if name is not present.
//
// Decoding reuses the full [Load] path (verbatim-bit fast path, FP8/int widening,
// big-endian fallback) by framing the one tensor's bytes as a minimal in-memory
// safetensors stream, so a tensor loaded this way is bit-identical to the same
// tensor from [LoadFile].
func LoadTensor(path, name string) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	hlen, raw, _, err := readHeaderRaw(f)
	if err != nil {
		return nil, err
	}
	msg, ok := raw[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found in %s", name, path)
	}
	var e entry
	if err := json.Unmarshal(msg, &e); err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
	}
	begin, end := e.DataOffsets[0], e.DataOffsets[1]
	if begin < 0 || end < begin {
		return nil, fmt.Errorf("safetensors: tensor %q: bad data_offsets %v", name, e.DataOffsets)
	}
	size := end - begin

	// Data begins after the 8-byte length field and the header. Read this
	// tensor's exact byte range via ReadAt — no other tensor is touched.
	dataStart := int64(8) + int64(hlen)
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if dataStart+int64(end) > fi.Size() {
		return nil, fmt.Errorf("safetensors: tensor %q: data ends at %d but the file holds only %d bytes of payload — malformed file", name, end, fi.Size()-dataStart)
	}
	payload := make([]byte, size)
	if _, err := f.ReadAt(payload, dataStart+int64(begin)); err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: read data: %w", name, err)
	}

	// Frame the one tensor as a minimal safetensors stream and reuse Load — so the
	// decode is the exact same code path as LoadFile, not a second implementation.
	single, err := json.Marshal(map[string]entry{
		name: {Dtype: e.Dtype, Shape: e.Shape, DataOffsets: [2]int{0, size}},
	})
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(single)))
	buf.Write(single)
	buf.Write(payload)
	ts, _, err := Load(&buf)
	if err != nil {
		return nil, err
	}
	return ts[name], nil
}

// readHeaderRaw reads the length-prefixed JSON header from r (which must be
// positioned at the start of the file), splits off __metadata__, and returns the
// remaining tensor entries as raw messages plus the metadata. No tensor data is
// read.
func readHeaderRaw(r io.Reader) (uint64, map[string]json.RawMessage, map[string]string, error) {
	var hlen uint64
	if err := readHeaderLen(r, &hlen); err != nil {
		return 0, nil, nil, err
	}
	hbuf := make([]byte, hlen)
	if _, err := io.ReadFull(r, hbuf); err != nil {
		return 0, nil, nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hbuf, &raw); err != nil {
		return 0, nil, nil, fmt.Errorf("safetensors: parse header: %w", err)
	}
	var meta map[string]string
	if m, ok := raw["__metadata__"]; ok {
		if err := json.Unmarshal(m, &meta); err != nil {
			return 0, nil, nil, fmt.Errorf("safetensors: parse __metadata__: %w", err)
		}
		delete(raw, "__metadata__")
	}
	return hlen, raw, meta, nil
}

// readHeaderLen reads and validates the 8-byte little-endian header length.
func readHeaderLen(r io.Reader, hlen *uint64) error {
	if err := binary.Read(r, binary.LittleEndian, hlen); err != nil {
		return fmt.Errorf("safetensors: read header length: %w", err)
	}
	if *hlen > maxHeaderSize {
		return fmt.Errorf("safetensors: header size %d exceeds cap %d", *hlen, maxHeaderSize)
	}
	return nil
}

// sortInfos orders descriptors by name (insertion sort — the slice is tiny, one
// entry per tensor, and it avoids pulling in sort for a stable listing).
func sortInfos(s []TensorInfo) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1].Name > s[j].Name; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
