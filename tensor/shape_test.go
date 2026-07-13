package tensor

import "testing"

func TestShapeNumel(t *testing.T) {
	cases := []struct {
		s    Shape
		want int
	}{
		{Shape{}, 1},         // scalar: empty product = 1 (§V12 edge)
		{Shape{5}, 5},        // vector
		{Shape{2, 3, 4}, 24}, // tensor
		{Shape{0}, 0},        // empty axis (§V12 edge)
		{Shape{2, 0, 3}, 0},  // any zero dim → empty
	}
	for _, c := range cases {
		if got := c.s.Numel(); got != c.want {
			t.Errorf("%v.Numel() = %d, want %d", c.s, got, c.want)
		}
	}
}

func TestShapePredicates(t *testing.T) {
	if !(Shape{}).IsScalar() {
		t.Error("empty shape must be scalar")
	}
	if (Shape{1}).IsScalar() {
		t.Error("(1) is not scalar")
	}
	if !(Shape{2, 0}).IsEmpty() {
		t.Error("(2,0) must be empty")
	}
	if (Shape{2, 3}).IsEmpty() {
		t.Error("(2,3) must not be empty")
	}
	if (Shape{-1, 2}).IsValid() {
		t.Error("negative dim must be invalid")
	}
	if !(Shape{0, 2}).IsValid() {
		t.Error("zero dim is valid")
	}
}

func TestShapeEqual(t *testing.T) {
	if !(Shape{2, 3}).Equal(Shape{2, 3}) {
		t.Error("equal shapes must compare equal")
	}
	if (Shape{2, 3}).Equal(Shape{3, 2}) {
		t.Error("order matters")
	}
	if (Shape{2, 3}).Equal(Shape{2, 3, 1}) {
		t.Error("rank matters")
	}
	if !(Shape{}).Equal(Shape{}) {
		t.Error("scalar equals scalar")
	}
}

func TestShapeCloneIndependent(t *testing.T) {
	s := Shape{2, 3}
	c := s.Clone()
	c[0] = 99
	if s[0] != 2 {
		t.Error("Clone must not alias the original")
	}
	if got := (Shape)(nil).Clone(); got != nil {
		t.Errorf("nil.Clone() = %v, want nil", got)
	}
}

func TestShapeString(t *testing.T) {
	cases := map[string]Shape{
		"()":        {},
		"(5)":       {5},
		"(2, 3, 4)": {2, 3, 4},
	}
	for want, s := range cases {
		if got := s.String(); got != want {
			t.Errorf("%v.String() = %q, want %q", []int(s), got, want)
		}
	}
}
