package main

import "testing"

func unbufferedFileFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "unbuffered-file-to-parser" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3029_UnbufferedFileToParser is the measured shape: a loader that opens a file and
// hands the raw handle to a parser which reads a length and then bytes for every string in the
// header. Buffering it took a header-heavy load from 66.0ms to 5.5ms.
func TestDetectPS3029_UnbufferedFileToParser(t *testing.T) {
	src := `package p

func ReadFile(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}`
	fs := unbufferedFileFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The constant-in-file-size property is the part that decides where to look for this and why a
	// large-payload benchmark will not show it. It must survive into the advice.
	if !containsAll(fs[0].msg, "CONSTANT IN FILE SIZE", "bufio.NewReaderSize") {
		t.Fatalf("message omits the scaling property or the fix:\n%s", fs[0].msg)
	}
}

// TestDetectPS3029_SilentWhenBuffered pins the applied form.
func TestDetectPS3029_SilentWhenBuffered(t *testing.T) {
	src := `package p

func ReadFile(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(bufio.NewReaderSize(f, 1<<20))
}`
	if fs := unbufferedFileFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a wrapped handle is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3029_SilentOnBulkConsumer pins the BULK exclusion. A consumer that swallows the file
// in a few large reads gains nothing from a buffer and pays a copy for it, so handing the raw
// handle there is correct. The fixture keeps the open and the pass-through, and changes only who
// receives it.
func TestDetectPS3029_SilentOnBulkConsumer(t *testing.T) {
	src := `package p

func ReadFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}`
	if fs := unbufferedFileFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a bulk consumer needs no buffer:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3029_SilentOnReaderParameter pins that the handle must be OPENED HERE. A function
// taking an io.Reader cannot know whether its caller already buffered, and wrapping it a second
// time would add a copy; the decision belongs to whoever owns the file. The fixture passes a
// parameter to the same parser the positive calls.
func TestDetectPS3029_SilentOnReaderParameter(t *testing.T) {
	src := `package p

func Parse(r io.Reader) (*File, error) {
	return Read(r)
}`
	if fs := unbufferedFileFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the caller owns the buffering decision", len(fs))
	}
}

// TestDetectPS3029_SilentOnNonFileSource pins that the handle must come from OPENING A FILE. A
// reader obtained some other way may already be buffered, may be a network stream, or may be an
// in-memory slice where a buffer is pure overhead — the syscall-per-field argument does not apply,
// and this scanner has no way to know which it is. The fixture is the positive with the open
// replaced by an ordinary call, so it discriminates the source alone.
func TestDetectPS3029_SilentOnNonFileSource(t *testing.T) {
	src := `package p

func ReadFrom(get func() io.Reader) (*File, error) {
	r := get()
	return Read(r)
}`
	if fs := unbufferedFileFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the reader did not come from opening a file:\n%s",
			len(fs), fs[0].msg)
	}
}
