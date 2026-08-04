package pytorch

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"unsafe"

	"github.com/jxsl13/goai/tensor"
)

// maxTensors caps the number of tensors a checkpoint may declare, so a hostile
// pickle cannot exhaust memory by declaring an enormous state_dict.
const maxTensors = 1 << 20

// storageType is a whitelisted torch typed-storage class: its element byte width
// on disk and the GoAI tensor dtype it decodes to.
type storageType struct {
	size int          // on-disk bytes per element
	out  tensor.Dtype // GoAI storage dtype after (possibly widening) decode
	kind string       // decode kind (distinguishes e.g. uint8 vs int8, both size 1)
}

// storageTypes is the whitelist of torch storage classes. A GLOBAL naming any
// other class is rejected. Float types are verbatim; integer/bool widen exactly
// the way package safetensors widens them.
var storageTypes = map[string]storageType{
	"FloatStorage":    {4, tensor.F32, "f32"},
	"DoubleStorage":   {8, tensor.F64, "f64"},
	"HalfStorage":     {2, tensor.F16, "f16"},
	"BFloat16Storage": {2, tensor.BF16, "bf16"},
	"ByteStorage":     {1, tensor.F32, "u8"},   // uint8 → F32
	"CharStorage":     {1, tensor.F32, "i8"},   // int8  → F32
	"BoolStorage":     {1, tensor.F32, "bool"}, // bool  → F32
	"ShortStorage":    {2, tensor.F32, "i16"},  // int16 → F32
	"IntStorage":      {4, tensor.F64, "i32"},  // int32 → F64
	"LongStorage":     {8, tensor.F64, "i64"},  // int64 → F64 (exact ≤ 2^53)
}

// nativeLittleEndian mirrors the safetensors fast-path guard: on a little-endian
// host the raw storage bytes already match a verbatim-bit tensor's in-memory
// layout, so it loads with one bulk copy instead of a per-element decode.
var nativeLittleEndian = func() bool {
	var x uint16 = 1
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// --- safe restricted unpickler ---------------------------------------------

// global is a whitelisted torch/collections symbol left un-evaluated on the
// stack (we never call it — REDUCE interprets it structurally).
type global struct{ module, name string }

// storageRef is a resolved persistent-id: which data/<key> blob, its element
// type, and element count.
type storageRef struct {
	st    storageType
	key   string
	numel int
}

// tensorSpec is the structural result of _rebuild_tensor_v2(storage, offset,
// size, stride, ...): everything needed to materialize one tensor.
type tensorSpec struct {
	ref    storageRef
	offset int
	shape  []int
	stride []int
}

type mark struct{} // sentinel pushed by MARK

type unpickler struct {
	buf  []byte
	pos  int
	memo map[int]any
	st   []any
}

func (u *unpickler) push(v any) { u.st = append(u.st, v) }

func (u *unpickler) pop() (any, error) {
	if len(u.st) == 0 {
		return nil, fmt.Errorf("pytorch: pickle stack underflow")
	}
	v := u.st[len(u.st)-1]
	u.st = u.st[:len(u.st)-1]
	return v, nil
}

// peek returns the top of the stack WITHOUT removing it, erroring when the
// stack is empty.
//
// The memo-storing and in-place-mutating opcodes (MEMOIZE, BINPUT,
// LONG_BINPUT, SETITEMS) inspect the top of the stack rather than popping it.
// They previously indexed u.st[len(u.st)-1] directly, so a hostile data.pkl
// that reached one of them on an empty stack — e.g. the four bytes
// {0x80, 2, 'q', 0, '.'} inside an otherwise valid zip — crashed the process
// with "index out of range [-1]". Routing every peek through this helper makes
// a malformed checkpoint an error value, matching the checked pop used by the
// popping opcodes.
func (u *unpickler) peek() (any, error) {
	if len(u.st) == 0 {
		return nil, fmt.Errorf("pytorch: pickle stack underflow")
	}
	return u.st[len(u.st)-1], nil
}

// popMark pops back to (and removes) the topmost mark, returning the items above
// it in order.
func (u *unpickler) popMark() ([]any, error) {
	for i := len(u.st) - 1; i >= 0; i-- {
		if _, ok := u.st[i].(mark); ok {
			items := append([]any(nil), u.st[i+1:]...)
			u.st = u.st[:i]
			return items, nil
		}
	}
	return nil, fmt.Errorf("pytorch: no mark on stack")
}

func (u *unpickler) byte() (byte, error) {
	if u.pos >= len(u.buf) {
		return 0, io.ErrUnexpectedEOF
	}
	b := u.buf[u.pos]
	u.pos++
	return b, nil
}

func (u *unpickler) read(n int) ([]byte, error) {
	// Compare against the REMAINING length, not u.pos+n: a hostile 8-byte length
	// (BINUNICODE8/BINBYTES8) near maxint64 makes u.pos+n overflow to a negative
	// value that slips past `> len(u.buf)`, and the slice below then panics (§V29,
	// §B99 class). len(u.buf)-u.pos cannot overflow (u.pos ≤ len(u.buf) always).
	if n < 0 || n > len(u.buf)-u.pos {
		return nil, io.ErrUnexpectedEOF
	}
	b := u.buf[u.pos : u.pos+n]
	u.pos += n
	return b, nil
}

// line reads up to and including a '\n' and returns the content without it.
func (u *unpickler) line() (string, error) {
	i := u.pos
	for i < len(u.buf) && u.buf[i] != '\n' {
		i++
	}
	if i >= len(u.buf) {
		return "", io.ErrUnexpectedEOF
	}
	s := string(u.buf[u.pos:i])
	u.pos = i + 1
	return s, nil
}

func asInt(v any) (int, bool) {
	i, ok := v.(int)
	return i, ok
}

// run interprets the pickle and returns the top-of-stack value at STOP.
func (u *unpickler) run() (any, error) {
	u.memo = map[int]any{}
	//perfscan:ignore PS3003 unpickle memo map read, one-time checkpoint load
	for {
		op, err := u.byte()
		if err != nil {
			return nil, err
		}
		switch op {
		case 0x80: // PROTO
			if _, err := u.byte(); err != nil {
				return nil, err
			}
		case 0x95: // FRAME (proto 4): 8-byte length, skip
			if _, err := u.read(8); err != nil {
				return nil, err
			}
		case 0x94: // MEMOIZE (proto 4): store top at the next memo index
			v, err := u.peek()
			if err != nil {
				return nil, fmt.Errorf("pytorch: MEMOIZE underflow")
			}
			u.memo[len(u.memo)] = v
		case 0x8d: // BINUNICODE8 (proto 4): 8-byte length + utf8
			ln, err := u.read(8)
			if err != nil {
				return nil, err
			}
			s, err := u.read(int(binary.LittleEndian.Uint64(ln)))
			if err != nil {
				return nil, err
			}
			u.push(string(s))
		case 0x93: // STACK_GLOBAL (proto 4): module, name popped from the stack
			nameV, e1 := u.pop()
			moduleV, e2 := u.pop()
			if e1 != nil || e2 != nil {
				return nil, fmt.Errorf("pytorch: STACK_GLOBAL underflow")
			}
			module, ok1 := moduleV.(string)
			name, ok2 := nameV.(string)
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("pytorch: STACK_GLOBAL non-string operands")
			}
			if err := checkGlobal(module, name); err != nil {
				return nil, err
			}
			u.push(global{module, name})
		case '}': // EMPTY_DICT
			u.push(map[string]any{})
		case ']': // EMPTY_LIST
			u.push([]any{})
		case ')': // EMPTY_TUPLE
			u.push([]any{})
		case '(': // MARK
			u.push(mark{})
		case 'N': // NONE
			u.push(nil)
		case 0x88: // NEWTRUE
			u.push(true)
		case 0x89: // NEWFALSE
			u.push(false)
		case 'X': // BINUNICODE
			ln, err := u.read(4)
			if err != nil {
				return nil, err
			}
			s, err := u.read(int(binary.LittleEndian.Uint32(ln)))
			if err != nil {
				return nil, err
			}
			u.push(string(s))
		case 0x8c: // SHORT_BINUNICODE
			n, err := u.byte()
			if err != nil {
				return nil, err
			}
			s, err := u.read(int(n))
			if err != nil {
				return nil, err
			}
			u.push(string(s))
		case 'K': // BININT1
			b, err := u.byte()
			if err != nil {
				return nil, err
			}
			u.push(int(b))
		case 'M': // BININT2
			b, err := u.read(2)
			if err != nil {
				return nil, err
			}
			u.push(int(binary.LittleEndian.Uint16(b)))
		case 'J': // BININT (signed 4-byte)
			b, err := u.read(4)
			if err != nil {
				return nil, err
			}
			u.push(int(int32(binary.LittleEndian.Uint32(b))))
		case 0x8a: // LONG1
			n, err := u.byte()
			if err != nil {
				return nil, err
			}
			b, err := u.read(int(n))
			if err != nil {
				return nil, err
			}
			var v int64
			for i := len(b) - 1; i >= 0; i-- {
				v = v<<8 | int64(b[i])
			}
			if n > 0 && b[n-1]&0x80 != 0 { // sign-extend
				v -= 1 << (8 * uint(n))
			}
			u.push(int(v))
		case 'q': // BINPUT
			i, err := u.byte()
			if err != nil {
				return nil, err
			}
			v, err := u.peek()
			if err != nil {
				return nil, fmt.Errorf("pytorch: BINPUT underflow")
			}
			u.memo[int(i)] = v
		case 'r': // LONG_BINPUT
			b, err := u.read(4)
			if err != nil {
				return nil, err
			}
			v, err := u.peek()
			if err != nil {
				return nil, fmt.Errorf("pytorch: LONG_BINPUT underflow")
			}
			u.memo[int(binary.LittleEndian.Uint32(b))] = v
		case 'h': // BINGET
			i, err := u.byte()
			if err != nil {
				return nil, err
			}
			u.push(u.memo[int(i)])
		case 'j': // LONG_BINGET
			b, err := u.read(4)
			if err != nil {
				return nil, err
			}
			u.push(u.memo[int(binary.LittleEndian.Uint32(b))])
		case 'c': // GLOBAL
			module, err := u.line()
			if err != nil {
				return nil, err
			}
			name, err := u.line()
			if err != nil {
				return nil, err
			}
			if err := checkGlobal(module, name); err != nil {
				return nil, err
			}
			u.push(global{module, name})
		case 't': // TUPLE
			items, err := u.popMark()
			if err != nil {
				return nil, err
			}
			u.push(items)
		case 0x85: // TUPLE1
			a, err := u.pop()
			if err != nil {
				return nil, err
			}
			u.push([]any{a})
		case 0x86: // TUPLE2
			b, err1 := u.pop()
			a, err2 := u.pop()
			if err1 != nil || err2 != nil {
				return nil, fmt.Errorf("pytorch: TUPLE2 underflow")
			}
			u.push([]any{a, b})
		case 0x87: // TUPLE3
			c, e1 := u.pop()
			b, e2 := u.pop()
			a, e3 := u.pop()
			if e1 != nil || e2 != nil || e3 != nil {
				return nil, fmt.Errorf("pytorch: TUPLE3 underflow")
			}
			u.push([]any{a, b, c})
		case 'Q': // BINPERSID
			pid, err := u.pop()
			if err != nil {
				return nil, err
			}
			ref, err := resolvePersid(pid)
			if err != nil {
				return nil, err
			}
			u.push(ref)
		case 'R': // REDUCE
			args, err1 := u.pop()
			fn, err2 := u.pop()
			if err1 != nil || err2 != nil {
				return nil, fmt.Errorf("pytorch: REDUCE underflow")
			}
			v, err := doReduce(fn, args)
			if err != nil {
				return nil, err
			}
			u.push(v)
		case 'u': // SETITEMS
			items, err := u.popMark()
			if err != nil {
				return nil, err
			}
			target, err := u.peek()
			if err != nil {
				return nil, fmt.Errorf("pytorch: SETITEMS underflow")
			}
			if err := setItems(target, items); err != nil {
				return nil, err
			}
		case 's': // SETITEM
			val, e1 := u.pop()
			key, e2 := u.pop()
			if e1 != nil || e2 != nil {
				return nil, fmt.Errorf("pytorch: SETITEM underflow")
			}
			// The two pops above are guarded, but the TARGET is peeked, and the
			// pops themselves can empty the stack: {0x80,2,'K',1,'K',2,'s','.'}
			// pops key and value cleanly and then peeks an empty stack.
			target, e3 := u.peek()
			if e3 != nil {
				return nil, fmt.Errorf("pytorch: SETITEM underflow")
			}
			if err := setItems(target, []any{key, val}); err != nil {
				return nil, err
			}
		case 'e': // APPENDS (list) — drop (used for empty backward_hooks)
			if _, err := u.popMark(); err != nil {
				return nil, err
			}
		case 'b': // BUILD — used to fill an OrderedDict's state; args already applied
			st, e1 := u.pop()
			_ = st
			if e1 != nil {
				return nil, e1
			}
			// leave the target (dict) on the stack unchanged
		case '.': // STOP
			return u.pop()
		default:
			return nil, fmt.Errorf("pytorch: unsupported pickle opcode 0x%02x (not part of the safe state_dict subset)", op)
		}
	}
}

// checkGlobal rejects any symbol outside the tensor-rebuild whitelist — the core
// of the no-code-execution guarantee.
func checkGlobal(module, name string) error {
	switch {
	case module == "torch._utils" && (name == "_rebuild_tensor_v2" || name == "_rebuild_parameter"):
		return nil
	case module == "collections" && name == "OrderedDict":
		return nil
	case module == "torch" && strings.HasSuffix(name, "Storage"):
		if _, ok := storageTypes[name]; ok {
			return nil
		}
	}
	return fmt.Errorf("pytorch: refusing disallowed global %s.%s (only tensor rebuild + storage classes are permitted; this checkpoint may be attempting code execution)", module, name)
}

// resolvePersid turns a persistent-id tuple ('storage', <StorageClass>, key,
// location, numel) into a storageRef.
func resolvePersid(pid any) (storageRef, error) {
	t, ok := pid.([]any)
	if !ok || len(t) < 5 {
		return storageRef{}, fmt.Errorf("pytorch: malformed persistent id")
	}
	if s, _ := t[0].(string); s != "storage" {
		return storageRef{}, fmt.Errorf("pytorch: unsupported persistent id kind %v", t[0])
	}
	g, ok := t[1].(global)
	if !ok {
		return storageRef{}, fmt.Errorf("pytorch: persistent id storage class not a global")
	}
	sty, ok := storageTypes[g.name]
	if !ok {
		return storageRef{}, fmt.Errorf("pytorch: unknown storage class %s", g.name)
	}
	key, ok := t[2].(string)
	if !ok {
		return storageRef{}, fmt.Errorf("pytorch: persistent id key not a string")
	}
	numel, ok := asInt(t[4])
	if !ok {
		return storageRef{}, fmt.Errorf("pytorch: persistent id numel not an int")
	}
	return storageRef{st: sty, key: key, numel: numel}, nil
}

// doReduce interprets the two whitelisted reduces structurally.
func doReduce(fn, args any) (any, error) {
	g, ok := fn.(global)
	if !ok {
		return nil, fmt.Errorf("pytorch: REDUCE on non-global")
	}
	switch {
	case g.module == "collections" && g.name == "OrderedDict":
		return map[string]any{}, nil
	case g.module == "torch._utils" && g.name == "_rebuild_parameter":
		// _rebuild_parameter(data, requires_grad, backward_hooks) → the wrapped
		// tensor (whole-module saves box each weight in an nn.Parameter).
		a, ok := args.([]any)
		if !ok || len(a) < 1 {
			return nil, fmt.Errorf("pytorch: _rebuild_parameter bad args")
		}
		return a[0], nil
	case g.module == "torch._utils" && g.name == "_rebuild_tensor_v2":
		a, ok := args.([]any)
		if !ok || len(a) < 4 {
			return nil, fmt.Errorf("pytorch: _rebuild_tensor_v2 bad args")
		}
		ref, ok := a[0].(storageRef)
		if !ok {
			return nil, fmt.Errorf("pytorch: tensor storage not a persistent id")
		}
		off, ok := asInt(a[1])
		if !ok {
			return nil, fmt.Errorf("pytorch: tensor offset not an int")
		}
		shape, err := intTuple(a[2])
		if err != nil {
			return nil, fmt.Errorf("pytorch: tensor shape: %w", err)
		}
		stride, err := intTuple(a[3])
		if err != nil {
			return nil, fmt.Errorf("pytorch: tensor stride: %w", err)
		}
		return tensorSpec{ref: ref, offset: off, shape: shape, stride: stride}, nil
	}
	return nil, fmt.Errorf("pytorch: disallowed reduce %s.%s", g.module, g.name)
}

//perfscan:ignore PS3033 intTuple parse, one-time checkpoint load
func intTuple(v any) ([]int, error) {
	t, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("not a tuple")
	}
	out := make([]int, len(t))
	for i, e := range t {
		n, ok := asInt(e)
		if !ok {
			return nil, fmt.Errorf("element %d not an int", i)
		}
		out[i] = n
	}
	return out, nil
}

// setItems assigns key/value pairs into a target dict. String keys are taken
// verbatim; integer keys (torch.save of a dict keyed by layer index, e.g. the
// jacobian-lens artifact's {"J": {0: T, 1: T, …}}) are stringified with
// strconv, so a nested int-keyed dict flattens to "J.0", "J.1", … exactly like
// a string-keyed one. Any other key type is rejected.
func setItems(target any, items []any) error {
	m, ok := target.(map[string]any)
	if !ok {
		return fmt.Errorf("pytorch: SETITEMS target not a dict")
	}
	if len(items)%2 != 0 {
		return fmt.Errorf("pytorch: odd SETITEMS")
	}
	for i := 0; i < len(items); i += 2 {
		switch k := items[i].(type) {
		case string:
			m[k] = items[i+1]
		case int:
			m[strconv.Itoa(k)] = items[i+1]
		default:
			return fmt.Errorf("pytorch: unsupported dict key type %T", items[i])
		}
	}
	return nil
}

// --- loading ---------------------------------------------------------------

// contiguousStride returns the C-order stride for a shape.
//
//perfscan:ignore PS3033 contiguousStride, one-time checkpoint load
func contiguousStride(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

// maxElems caps a single tensor's element count, mirroring npy.maxElems. It
// bounds the allocation a hostile shape can request before any storage is
// touched.
const maxElems = 1 << 30

// numelOf returns the product of shape's dimensions, rejecting negative dims
// and any product that would exceed maxElems.
//
// The guard divides BEFORE multiplying, mirroring the overflow-safe product in
// format/npy (see npy.go, "Overflow-safe product") — safetensors and gguf use
// the same idiom, so the four readers stay one discoverable family. A
// post-multiply `n > maxElems` check is not equivalent: it wraps. Shape
// (2^62, 4) is exactly 2^64 ≡ 0, and shape (2^62+1, 4) is 2^64+4 ≡ 4, so a
// crafted checkpoint passed both the "exceeds storage" and "blob too short"
// checks in materialize and Load returned a nil error with a tensor
// advertising a nonsense shape over near-empty storage — a silent wrong result
// whose panic surfaces far from the load site. A negative dim wrapped the other
// way and sliced the storage blob with a negative bound.
func numelOf(shape []int) (int, error) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			//perfscan:ignore PS3013 numelOf shape validation, one-time load
			return 0, fmt.Errorf("pytorch: negative dimension in shape %v", shape)
		}
		if d != 0 && n > maxElems/d {
			return 0, fmt.Errorf("pytorch: tensor too large (shape %v exceeds %d elements)", shape, maxElems)
		}
		n *= d
	}
	return n, nil
}

// flatten walks the (possibly nested) state_dict, joining nested keys with '.'
// and collecting the tensor specs. Non-tensor leaves (scalars, None) are skipped
// — a checkpoint often carries bookkeeping entries beside its weights.
func flatten(prefix string, v any, out map[string]tensorSpec) error {
	switch t := v.(type) {
	case tensorSpec:
		if len(out) >= maxTensors {
			return fmt.Errorf("pytorch: too many tensors (> %d)", maxTensors)
		}
		out[prefix] = t
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			child := k
			if prefix != "" {
				child = prefix + "." + k
			}
			if err := flatten(child, t[k], out); err != nil {
				return err
			}
		}
	default:
		// scalar / None / list bookkeeping entry — not a tensor, ignore.
	}
	return nil
}

// materialize reads a tensor spec's storage blob and builds the GoAI tensor.
func materialize(spec tensorSpec, blob []byte) (*tensor.Tensor, error) {
	shape := spec.shape
	n, err := numelOf(shape)
	if err != nil {
		return nil, err
	}
	// Only plain contiguous, offset-0 views are supported; anything else is a
	// non-trivial alias we refuse rather than mis-read.
	if spec.offset != 0 {
		return nil, fmt.Errorf("pytorch: non-zero storage offset %d unsupported", spec.offset)
	}
	if len(spec.stride) != len(shape) {
		return nil, fmt.Errorf("pytorch: stride/shape rank mismatch")
	}
	if want := contiguousStride(shape); !eqInts(spec.stride, want) && n > 0 {
		return nil, fmt.Errorf("pytorch: non-contiguous tensor (stride %v != %v) unsupported", spec.stride, want)
	}
	st := spec.ref.st
	if n > spec.ref.numel {
		return nil, fmt.Errorf("pytorch: shape %v (%d elems) exceeds storage %d", shape, n, spec.ref.numel)
	}
	if len(blob) < n*st.size {
		return nil, fmt.Errorf("pytorch: storage blob %d bytes < %d needed", len(blob), n*st.size)
	}
	src := blob[:n*st.size]
	out := tensor.New(st.out, tensor.Shape(shape))
	decodeInto(out, src, st)
	return out, nil
}

// decodeInto fills out's storage from the raw little-endian storage bytes.
// Verbatim-bit dtypes (F32/F64/F16/BF16) bulk-copy on a little-endian host; the
// integer/bool storages widen element-wise, mirroring package safetensors.
func decodeInto(out *tensor.Tensor, src []byte, st storageType) {
	switch st.kind {
	case "f32":
		dst := out.Storage().F32()
		if !rawCopyLE(dst, src, 4) {
			for i := range dst {
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
			}
		}
	case "f64":
		dst := out.Storage().F64()
		if !rawCopyLE(dst, src, 8) {
			for i := range dst {
				dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(src[i*8:]))
			}
		}
	case "f16", "bf16":
		dst := out.Storage().U16()
		if !rawCopyLE(dst, src, 2) {
			for i := range dst {
				dst[i] = binary.LittleEndian.Uint16(src[i*2:])
			}
		}
	case "u8":
		dst := out.Storage().F32()
		for i, b := range src {
			dst[i] = float32(b)
		}
	case "i8":
		dst := out.Storage().F32()
		for i, b := range src {
			dst[i] = float32(int8(b))
		}
	case "bool":
		dst := out.Storage().F32()
		for i, b := range src {
			if b != 0 {
				dst[i] = 1
			}
		}
	case "i16":
		dst := out.Storage().F32()
		for i := range dst {
			dst[i] = float32(int16(binary.LittleEndian.Uint16(src[i*2:])))
		}
	case "i32":
		dst := out.Storage().F64()
		for i := range dst {
			dst[i] = float64(int32(binary.LittleEndian.Uint32(src[i*4:])))
		}
	case "i64":
		dst := out.Storage().F64()
		for i := range dst {
			dst[i] = float64(int64(binary.LittleEndian.Uint64(src[i*8:])))
		}
	}
}

// rawCopyLE bulk-copies verbatim little-endian bytes into a numeric slice's
// backing store (the safetensors fast path); false on big-endian or empty.
func rawCopyLE[T any](dst []T, src []byte, elemSize int) bool {
	if !nativeLittleEndian || len(dst) == 0 {
		return false
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize), src)
	return true
}

// Load reads a state_dict from a .pt/.bin archive. r must provide random access
// and size is the archive length (as with archive/zip). Nested modules are
// flattened with '.'-joined keys, matching the checkpoint's parameter names.
func Load(r io.ReaderAt, size int64) (map[string]*tensor.Tensor, error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return nil, fmt.Errorf("pytorch: open archive: %w", err)
	}
	// Locate <root>/data.pkl and the data/ storage directory.
	var pklName, root string
	for _, f := range zr.File {
		if strings.HasSuffix(f.Name, "/data.pkl") || f.Name == "data.pkl" {
			pklName = f.Name
			root = strings.TrimSuffix(f.Name, "data.pkl")
			break
		}
	}
	if pklName == "" {
		return nil, fmt.Errorf("pytorch: no data.pkl in archive (not a torch.save checkpoint?)")
	}
	zf := make(map[string]*zip.File, len(zr.File))
	for _, f := range zr.File {
		zf[f.Name] = f
	}
	readEntry := func(name string) ([]byte, error) {
		f := zf[name]
		if f == nil {
			return nil, fmt.Errorf("pytorch: missing archive entry %q", name)
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		defer rc.Close()
		return io.ReadAll(rc)
	}

	pkl, err := readEntry(pklName)
	if err != nil {
		return nil, fmt.Errorf("pytorch: read data.pkl: %w", err)
	}
	u := &unpickler{buf: pkl}
	root2, err := u.run()
	if err != nil {
		return nil, err
	}
	specs := map[string]tensorSpec{}
	if err := flatten("", root2, specs); err != nil {
		return nil, err
	}

	out := make(map[string]*tensor.Tensor, len(specs))
	for name, spec := range specs {
		blob, err := readEntry(root + "data/" + spec.ref.key)
		if err != nil {
			return nil, fmt.Errorf("pytorch: tensor %q: %w", name, err)
		}
		t, err := materialize(spec, blob)
		if err != nil {
			return nil, fmt.Errorf("pytorch: tensor %q: %w", name, err)
		}
		out[name] = t
	}
	return out, nil
}

// LoadFile reads a state_dict from a .pt/.bin file at path.
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
	//perfscan:ignore PS3029 Load file helper, one-time checkpoint deserialize
	return Load(f, info.Size())
}

func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
