package main

import (
	"strings"
	"testing"
)

func cgoUploadFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "per-dispatch-invariant-upload" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS7004_UploadFreedInBody is the shipped shape: RoPEPartial uploaded its inv-freq
// table and freed it via defer in the SAME call — a cudaMalloc + H2D + cudaFree of invariant
// (attrs-only) data on every dispatch. #997 routed it through a resident cache for +8.1%.
func TestDetectPS7004_UploadFreedInBody(t *testing.T) {
	src := `package p

func (d *T) RoPEPartial(rotaryDim int) error {
	inv32 := freqs(rotaryDim)
	invPtr := C.cu_upload_f32((*C.float)(&inv32[0]), C.int(len(inv32)))
	defer C.cu_free_f32(invPtr)
	C.cu_rope_partial(d.ptr, invPtr, C.int(d.rows))
	return nil
}`
	fs := cgoUploadFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The remedy (an attrs-keyed resident cache) must be in the message.
	if !strings.Contains(fs[0].msg, "attrs") {
		t.Fatalf("message omits the cache remedy:\n%s", fs[0].msg)
	}
}

// TestDetectPS7004_SilentWhenCached is the fix: the upload result is kept (routed through a
// resident cache helper) with no same-body defer free — nothing to flag.
func TestDetectPS7004_SilentWhenCached(t *testing.T) {
	src := `package p

func (d *T) RoPEPartial(rotaryDim int) error {
	inv32 := freqs(rotaryDim)
	invPtr, err := cachedUpload(inv32)
	if err != nil {
		return err
	}
	C.cu_rope_partial(d.ptr, invPtr, C.int(d.rows))
	return nil
}`
	if fs := cgoUploadFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the cached path has no per-call upload+free:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS7004_SilentOnTensorDataUpload is the main precision floor: uploading a tensor's
// own storage (xc.Storage().F32()) is per-call-varying DATA, correctly staged each dispatch.
func TestDetectPS7004_SilentOnTensorDataUpload(t *testing.T) {
	src := `package p

func upload(xc *T) bool {
	s := xc.Storage().F32()
	d := C.cu_upload_f32((*C.float)(&s[0]), C.int(len(s)))
	defer C.cu_free_f32(d)
	C.cu_op(d)
	return true
}`
	if fs := cgoUploadFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — uploading .Storage().F32() varies per call:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS7004_SilentOnParamUpload keeps it off uploads of a function's slice PARAMETER —
// that is the op's input data, not an attrs-derived table.
func TestDetectPS7004_SilentOnParamUpload(t *testing.T) {
	src := `package p

func i8mma(a []int8) {
	dA := C.cu_upload_i8((*C.schar)(&a[0]), C.int(len(a)))
	defer C.cu_free_f32(dA)
	C.cu_mma(dA)
}`
	if fs := cgoUploadFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a param slice is input data, not an invariant table:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS7004_SilentOnOutputAlloc is the precision floor: an alloc+free of an OUTPUT
// scratch buffer (not an upload of host data) is legitimate per-call scratch, not invariant
// staging. The detector keys on `upload`, so cu_alloc_* stays silent.
func TestDetectPS7004_SilentOnOutputAlloc(t *testing.T) {
	src := `package p

func (d *T) MatMul(b *T) error {
	out := C.cu_alloc_f32(C.int(d.rows * b.cols))
	defer C.cu_free_f32(out)
	C.cu_matmul(d.ptr, b.ptr, out)
	return nil
}`
	if fs := cgoUploadFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an alloc+free of an OUTPUT buffer is not an invariant upload:\n%s", len(fs), fs[0].msg)
	}
}
