package pytorch_test

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/pytorch"
)

// hostileZip wraps a hand-crafted data.pkl (and optional data/<key> storage
// blobs) in a minimal, VALID torch-shaped zip container.
//
// Why it exists: pytorch.Load opens the archive before it ever reaches the
// unpickler, so raw pickle bytes handed straight to Load are rejected with
// "pytorch: open archive: zip: not a valid zip file" and prove nothing about
// the interpreter. Every hostile-pickle test must go through a real zip or it
// silently tests the zip reader instead of the parser under test.
func hostileZip(t *testing.T, pkl []byte, blobs map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	add := func(name string, data []byte) {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: name, Method: zip.Store})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	add("archive/data.pkl", pkl)
	for k, v := range blobs {
		add("archive/data/"+k, v)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// loadHostile runs Load over a hostile archive and converts a panic into a test
// failure — the whole point of the hardening is that malformed input yields an
// error value, never a crash.
func loadHostile(t *testing.T, name string, archive []byte) error {
	t.Helper()
	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("%s: panicked: %v", name, r)
			}
		}()
		_, err = pytorch.Load(bytes.NewReader(archive), int64(len(archive)))
	}()
	return err
}

// Regression: opcodes that PEEK at the top of the pickle stack (MEMOIZE,
// BINPUT, LONG_BINPUT, SETITEMS) indexed u.st[len(u.st)-1] with no emptiness
// check, so a hostile data.pkl that reaches one of them on an empty stack
// crashed the process with "index out of range [-1]" instead of returning an
// error. SETITEM was already guarded and is kept here so the family stays
// covered together.
func TestHostileStackUnderflow(t *testing.T) {
	cases := map[string][]byte{
		"MEMOIZE":     {0x80, 4, 0x94, '.'},
		"BINPUT":      {0x80, 2, 'q', 0, '.'},
		"LONG_BINPUT": {0x80, 2, 'r', 0, 0, 0, 0, '.'},
		"SETITEMS":    {0x80, 2, '(', 'u', '.'},
		// SETITEM's two pops were already guarded (this case was clean before the
		// fix) — kept as the model for the error style.
		"SETITEM pops": {0x80, 2, 's', '.'},
		// ...but its TARGET was peeked unguarded, and the pops themselves can
		// empty the stack, so SETITEM was a crasher too — one the original sweep
		// missed because it only probed the pop path.
		"SETITEM target": {0x80, 2, 'K', 1, 'K', 2, 's', '.'},
	}
	for name, pkl := range cases {
		t.Run(name, func(t *testing.T) {
			if err := loadHostile(t, name, hostileZip(t, pkl, nil)); err == nil {
				t.Errorf("%s: must error on empty-stack peek, got nil", name)
			}
		})
	}
}

// pickle builder helpers -----------------------------------------------------

func pBinUnicode(b *bytes.Buffer, s string) {
	b.WriteByte('X')
	_ = binary.Write(b, binary.LittleEndian, uint32(len(s)))
	b.WriteString(s)
}

// pLong1 emits LONG1 for a value too large for BININT1 — the only way a hostile
// pickle can smuggle a 2^62-scale dimension into a shape tuple.
func pLong1(b *bytes.Buffer, v int64) {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], uint64(v))
	b.WriteByte(0x8a)
	b.WriteByte(8)
	b.Write(raw[:])
}

// hostileShapePickle builds a valid proto-2 pickle for {"w": tensor} whose
// _rebuild_tensor_v2 shape is exactly the caller-supplied dims. Everything else
// (storage class, key, offset, stride) is well-formed, so the ONLY thing under
// test is shape validation.
func hostileShapePickle(shape []int64, stride []int64, storageNumel int) []byte {
	var b bytes.Buffer
	b.Write([]byte{0x80, 2}) // PROTO 2
	b.WriteByte('}')         // EMPTY_DICT
	b.WriteByte('(')         // MARK
	pBinUnicode(&b, "w")

	b.WriteString("ctorch._utils\n_rebuild_tensor_v2\n") // GLOBAL
	b.WriteByte('(')                                     // MARK (args)

	// persistent id: ('storage', torch.FloatStorage, '0', 'cpu', numel)
	b.WriteByte('(')
	pBinUnicode(&b, "storage")
	b.WriteString("ctorch\nFloatStorage\n")
	pBinUnicode(&b, "0")
	pBinUnicode(&b, string(backend.CPU))
	b.WriteByte('K')
	b.WriteByte(byte(storageNumel))
	b.WriteByte('t') // TUPLE
	b.WriteByte('Q') // BINPERSID

	b.Write([]byte{'K', 0}) // storage offset 0

	b.WriteByte('(') // shape tuple
	for _, d := range shape {
		pLong1(&b, d)
	}
	b.WriteByte('t')

	b.WriteByte('(') // stride tuple
	for _, s := range stride {
		pLong1(&b, s)
	}
	b.WriteByte('t')

	b.WriteByte(0x89) // requires_grad = False
	b.WriteByte(']')  // backward_hooks = []

	b.WriteByte('t') // TUPLE (args)
	b.WriteByte('R') // REDUCE

	b.WriteByte('u') // SETITEMS
	b.WriteByte('.') // STOP
	return b.Bytes()
}

// Regression: numelOf multiplied shape dims with no overflow guard and no
// negative-dim check, so materialize's "n > storage numel" and
// "len(blob) < n*size" checks compared against a WRAPPED product. A crafted
// shape like (2^62, 4) wraps to exactly 2^64 ≡ 0, passing every check, and Load
// returned err == nil with a tensor advertising a nonsense shape over
// zero-length storage — the worst failure class, since the panic then happens
// far from the load site. A negative dim wrapped the other way and sliced the
// blob with a negative bound, panicking inside materialize.
//
// The fix mirrors format/npy's division-before-multiply guard
// (npy.go, "Overflow-safe product") so the two readers stay one family;
// safetensors and gguf use the same idiom.
func TestHostileNumelOverflow(t *testing.T) {
	blob := map[string][]byte{"0": make([]byte, 16)}
	cases := map[string][]byte{
		// 2^62 * 4 == 2^64 ≡ 0: the exact wrap-to-zero attack.
		"wrap to zero (2^62 x 4)": hostileShapePickle(
			[]int64{1 << 62, 4}, []int64{4, 1}, 4),
		// (2^62+1) * 4 == 2^64+4 ≡ 4, with BOTH dims positive: wraps to a small
		// count that fits the declared storage exactly, so every downstream
		// check passes and the caller gets a 4-element tensor claiming ~4.6e18
		// rows.
		"wrap to small (2^62+1 x 4)": hostileShapePickle(
			[]int64{(1 << 62) + 1, 4}, []int64{4, 1}, 4),
		// negative dim: product goes negative, blob[:n*size] slices out of range.
		"negative dim": hostileShapePickle(
			[]int64{-4, 4}, []int64{4, 1}, 4),
	}
	for name, pkl := range cases {
		t.Run(name, func(t *testing.T) {
			if err := loadHostile(t, name, hostileZip(t, pkl, blob)); err == nil {
				t.Errorf("%s: must error, shape guard bypassed", name)
			}
		})
	}
}
