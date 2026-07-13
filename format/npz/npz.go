// Package npz reads and writes NumPy .npz archives — a ZIP container holding one
// .npy array per named entry (numpy.savez / numpy.load). It builds directly on the
// format/npy codec, so a whole set of named tensors (e.g. a model's weights) round-
// trips byte-faithfully with numpy/Python. Definitional source: the numpy .npz
// convention (a ZIP of "<name>.npy" members); there is no paper (§V16).
//
// In plain terms: .npz is simply a zip archive of .npy files — one named array per entry, the standard way NumPy saves several arrays together.
//
// Further reading: the NumPy .npy/.npz format specification (numpy/numpy, NEP 1) — the defining reference (file formats have no paper, SPEC V16).
package npz

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/tensor"
)

const maxEntries = 1 << 16 // reject archives with an absurd number of members (fuzz safety)

// Save writes arrays to w as an uncompressed .npz (ZIP_STORED, matching
// numpy.savez): each entry is stored as "<name>.npy". Entry names are written in
// sorted order for deterministic output.
func Save(w io.Writer, arrays map[string]*tensor.Tensor) error {
	names := make([]string, 0, len(arrays))
	for n := range arrays {
		names = append(names, n)
	}
	sort.Strings(names)

	zw := zip.NewWriter(w)
	for _, n := range names {
		fw, err := zw.CreateHeader(&zip.FileHeader{Name: n + ".npy", Method: zip.Store})
		if err != nil {
			return err
		}
		if err := npy.Save(fw, arrays[n]); err != nil {
			return fmt.Errorf("npz: array %q: %w", n, err)
		}
	}
	return zw.Close()
}

// SaveFile writes arrays to a .npz file at path.
func SaveFile(path string, arrays map[string]*tensor.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Save(f, arrays); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Load reads all arrays from a .npz archive. r must provide random access (a
// *bytes.Reader or *os.File); size is the archive length. Each member's name has
// its ".npy" suffix stripped. Malformed archives return an error and never panic;
// each array is decoded by the fuzz-safe npy.Load.
func Load(r io.ReaderAt, size int64) (map[string]*tensor.Tensor, error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return nil, fmt.Errorf("npz: opening archive: %w", err)
	}
	if len(zr.File) > maxEntries {
		return nil, fmt.Errorf("npz: too many entries (%d)", len(zr.File))
	}
	out := make(map[string]*tensor.Tensor, len(zr.File))
	for _, f := range zr.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("npz: opening member %q: %w", f.Name, err)
		}
		t, err := npy.Load(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("npz: member %q: %w", f.Name, err)
		}
		out[name] = t
	}
	return out, nil
}

// LoadFile reads a .npz archive from path.
func LoadFile(path string) (map[string]*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return Load(f, info.Size())
}
