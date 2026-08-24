package llamagpu

import "testing"

type residualScratchQWeight struct{}

func (*residualScratchQWeight) Close() error { return nil }

func residualScratchFixture(eager bool) *Decoder {
	return &Decoder{
		ops: backendOps{
			name: "storage-test", eagerDecoderResidualScratch: eager,
			newBuffer: func(data []float32) (buffer, error) {
				return &gptStorageBuffer{n: len(data)}, nil
			},
		},
		maxLen: 16,
		d:      8,
	}
}

func residualScratchElements(b *bufSlot) int {
	if b == nil || b.b == nil {
		return 0
	}
	return b.b.(*gptStorageBuffer).n
}

func TestDecoderResidualScratchReachability(t *testing.T) {
	for _, tc := range []struct {
		name      string
		configure func(*Decoder)
		wantAO    int
		wantMO    int
	}{
		{name: "f32-prenorm"},
		{name: "eager-control", configure: func(d *Decoder) { d.ops.eagerDecoderResidualScratch = true }, wantAO: 128, wantMO: 128},
		{name: "quantized", configure: func(d *Decoder) { d.qweights = []qweight{&residualScratchQWeight{}} }, wantAO: 128, wantMO: 128},
		{name: "post-norm", configure: func(d *Decoder) { d.postNorm = true }, wantAO: 128, wantMO: 128},
		{name: "sandwich", configure: func(d *Decoder) { d.sandwich = true }, wantAO: 128, wantMO: 128},
		{name: "f32-moe", configure: func(d *Decoder) { d.moe = true }, wantMO: 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := residualScratchFixture(false)
			if tc.configure != nil {
				tc.configure(d)
			}
			var err error
			d.allocResidualScratch(d.mkBuf(&err), d.maxLen)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Release()
			if d.ao == nil || d.mo == nil {
				t.Fatalf("ao/mo placeholders = %v/%v, want non-nil slots", d.ao, d.mo)
			}
			if gotAO, gotMO := residualScratchElements(d.ao), residualScratchElements(d.mo); gotAO != tc.wantAO || gotMO != tc.wantMO {
				t.Fatalf("ao/mo elements = %d/%d, want %d/%d", gotAO, gotMO, tc.wantAO, tc.wantMO)
			}
		})
	}
}

func BenchmarkDecoderResidualScratchResidency(b *testing.B) {
	for _, tc := range []struct {
		name  string
		eager bool
	}{
		{name: "lazy"},
		{name: "eager-control", eager: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			residentBytes := 0
			b.ReportAllocs()
			for range b.N {
				d := residualScratchFixture(tc.eager)
				d.maxLen, d.d = 2048, 2048
				var err error
				d.allocResidualScratch(d.mkBuf(&err), d.maxLen)
				if err != nil {
					b.Fatal(err)
				}
				residentBytes = (residualScratchElements(d.ao) + residualScratchElements(d.mo)) * 4
				d.Release()
			}
			b.ReportMetric(float64(residentBytes), "resident-scratch-B")
		})
	}
}
