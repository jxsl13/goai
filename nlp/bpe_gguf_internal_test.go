package nlp

import "testing"

// §R90: the GPT-2 byte→Unicode table maps printable bytes to themselves and the rest
// to U+0100+, is a bijection over all 256 bytes, and puts space at 'Ġ' and newline at
// 'Ċ'.
func TestBytesToUnicode(t *testing.T) {
	b2u, u2b := bytesToUnicode()
	if b2u[' '] != 0x0120 {
		t.Errorf("space → U+%04X, want U+0120 (Ġ)", b2u[' '])
	}
	if b2u['\n'] != 0x010A {
		t.Errorf("newline → U+%04X, want U+010A (Ċ)", b2u['\n'])
	}
	if b2u['a'] != 'a' {
		t.Errorf("printable 'a' must map to itself, got U+%04X", b2u['a'])
	}
	seen := map[rune]bool{}
	for b := range 256 {
		r := b2u[b]
		if seen[r] {
			t.Fatalf("byte→unicode not injective: rune U+%04X repeats at byte %d", r, b)
		}
		seen[r] = true
		if u2b[r] != byte(b) {
			t.Errorf("inverse broken: u2b[b2u[%d]] = %d", b, u2b[r])
		}
	}
}

// §V15: with all 256 byte code points as base tokens, decode∘encode is byte-exact for
// any input — including spaces, multibyte UTF-8 and invalid bytes.
func TestBPEByteRoundTrip(t *testing.T) {
	b2u, _ := bytesToUnicode()
	vocab := make([]string, 256)
	for b := range 256 {
		vocab[b] = string(b2u[b])
	}
	tok, err := NewBPE(vocab, nil) // no merges → every byte is its own token
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"hello world", "  spaced ", "üñîçødé", "tab\tnl\n", "\xff\xfe raw", ""} {
		if back := tok.Decode(tok.Encode(s)); back != s {
			t.Errorf("round-trip %q → %q", s, back)
		}
	}
}

// FuzzBPEByteRoundTrip: with the full 256-byte base vocabulary, decode∘encode must be
// byte-exact for ANY input (§V15), including arbitrary non-UTF-8 byte sequences.
func FuzzBPEByteRoundTrip(f *testing.F) {
	b2u, _ := bytesToUnicode()
	vocab := make([]string, 256)
	for b := range 256 {
		vocab[b] = string(b2u[b])
	}
	tok, err := NewBPE(vocab, nil)
	if err != nil {
		f.Fatal(err)
	}
	f.Add("hello world")
	f.Add("\x00\xff\xfe")
	f.Fuzz(func(t *testing.T, s string) {
		if back := tok.Decode(tok.Encode(s)); back != s {
			t.Errorf("byte round-trip %q → %q", s, back)
		}
	})
}
