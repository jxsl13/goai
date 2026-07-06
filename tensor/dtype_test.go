package tensor

import "testing"

func TestDtypeSize(t *testing.T) {
	cases := []struct {
		d    Dtype
		want int
	}{
		{F32, 4},
		{F64, 8},
		{Invalid, 0},
		{Dtype(200), 0}, // out of range → 0, no panic
	}
	for _, c := range cases {
		if got := c.d.Size(); got != c.want {
			t.Errorf("%v.Size() = %d, want %d", c.d, got, c.want)
		}
	}
}

func TestDtypeString(t *testing.T) {
	cases := map[Dtype]string{
		F32:        "f32",
		F64:        "f64",
		Invalid:    "invalid",
		Dtype(200): "dtype(?)",
	}
	for d, want := range cases {
		if got := d.String(); got != want {
			t.Errorf("Dtype(%d).String() = %q, want %q", d, got, want)
		}
	}
}

func TestDtypePredicates(t *testing.T) {
	if !F32.IsValid() || !F64.IsValid() {
		t.Error("F32/F64 must be valid")
	}
	if Invalid.IsValid() || Dtype(200).IsValid() {
		t.Error("Invalid/out-of-range must not be valid")
	}
	if !F32.IsFloat() || !F64.IsFloat() {
		t.Error("F32/F64 must be float")
	}
	if Invalid.IsFloat() {
		t.Error("Invalid must not be float")
	}
}
