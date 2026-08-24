package llamagpu

import (
	"errors"
	"testing"
)

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
		{name: "eager-control", configure: func(d *Decoder) { d.ops.eagerDecoderResidualScratch = true }, wantAO: 8, wantMO: 8},
		{name: "quantized", configure: func(d *Decoder) { d.qweights = []qweight{&residualScratchQWeight{}} }, wantAO: 8, wantMO: 8},
		{name: "post-norm", configure: func(d *Decoder) { d.postNorm = true }, wantAO: 8, wantMO: 8},
		{name: "sandwich", configure: func(d *Decoder) { d.sandwich = true }, wantAO: 8, wantMO: 8},
		{name: "f32-moe", configure: func(d *Decoder) { d.moe = true }, wantMO: 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := residualScratchFixture(false)
			if tc.configure != nil {
				tc.configure(d)
			}
			var err error
			d.allocResidualScratch(d.mkBuf(&err), 1)
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

func decoderScratchFixture(eager bool) *Decoder {
	return &Decoder{
		ops: backendOps{
			name: "storage-test", eagerFullDecoderScratch: eager,
			newBuffer: func(data []float32) (buffer, error) {
				return &gptStorageBuffer{n: len(data)}, nil
			},
		},
		maxLen: 96,
		d:      8,
		qDim:   8,
		kvDim:  8,
		hidden: 16,
		v:      17,
	}
}

func decoderScratchBuffers(s decoderScratch) map[string]*gptStorageBuffer {
	slots := map[string]*bufSlot{
		"dx": s.dx, "xn": s.xn, "xn2": s.xn2,
		"q": s.q, "k": s.k, "v": s.v_, "qkv": s.qkv, "attn": s.attn,
		"ao": s.ao, "gate": s.gate, "up": s.up, "mo": s.mo, "gu": s.gu,
		"moeGate": s.moeGate, "moeW": s.moeW, "moeCol": s.moeCol,
		"mlaCQ": s.mlaCQ, "mlaQ": s.mlaQ, "mlaComp": s.mlaComp,
		"mlaLatent": s.mlaLatent, "mlaKV": s.mlaKV, "mlaAttn": s.mlaAttn,
	}
	out := make(map[string]*gptStorageBuffer)
	for name, slot := range slots {
		if slot != nil && slot.b != nil {
			out[name] = slot.b.(*gptStorageBuffer)
		}
	}
	return out
}

func decoderScratchElements(s decoderScratch) int {
	n := 0
	for _, b := range decoderScratchBuffers(s) {
		n += b.n
	}
	return n
}

func TestDecoderScratchResidencyGrowthAndRelease(t *testing.T) {
	d := decoderScratchFixture(false)
	var err error
	d.allocScratch(d.mkBuf(&err))
	if err != nil {
		t.Fatal(err)
	}
	resident := d.residentScratch()
	if d.scratchRows != 1 || d.fullScratch.rows != 0 || decoderScratchElements(resident) != 112 {
		t.Fatalf("resident scratch rows=%d overflow=%d elements=%d, want 1/0/112",
			d.scratchRows, d.fullScratch.rows, decoderScratchElements(resident))
	}
	for name, want := range map[string]int{
		"dx": 8, "xn": 8, "xn2": 8, "q": 8, "k": 8, "v": 8,
		"qkv": 24, "attn": 8, "gate": 16, "up": 16,
	} {
		if got := decoderScratchBuffers(resident)[name].n; got != want {
			t.Fatalf("resident %s = %d elements, want %d", name, got, want)
		}
	}
	if residualScratchElements(resident.ao) != 0 || residualScratchElements(resident.mo) != 0 {
		t.Fatal("ordinary F32 resident generation retained residual projection scratch")
	}

	first, err := d.scratchForRows(4)
	if err != nil {
		t.Fatal(err)
	}
	if first.rows != 4 || decoderScratchElements(first) != 4*112 {
		t.Fatalf("first high-water workspace rows=%d elements=%d, want 4/%d",
			first.rows, decoderScratchElements(first), 4*112)
	}
	reused, err := d.scratchForRows(2)
	if err != nil {
		t.Fatal(err)
	}
	if reused.dx != first.dx {
		t.Fatal("smaller prefill did not reuse the grouped high-water workspace")
	}
	grown, err := d.scratchForRows(8)
	if err != nil {
		t.Fatal(err)
	}
	if grown.dx == first.dx || decoderScratchElements(grown) != 8*112 {
		t.Fatalf("grown workspace reused old=%v elements=%d, want false/%d",
			grown.dx == first.dx, decoderScratchElements(grown), 8*112)
	}
	for name, b := range decoderScratchBuffers(first) {
		if b.releases != 1 {
			t.Fatalf("replaced %s releases = %d, want 1", name, b.releases)
		}
	}
	d.Release()
	for name, b := range decoderScratchBuffers(grown) {
		if b.releases != 1 {
			t.Fatalf("final %s releases = %d, want 1", name, b.releases)
		}
	}
	if d.fullScratch.rows != 0 {
		t.Fatalf("Release retained full scratch rows=%d", d.fullScratch.rows)
	}

	eager := decoderScratchFixture(true)
	err = nil
	eager.allocScratch(eager.mkBuf(&err))
	if err != nil {
		t.Fatal(err)
	}
	defer eager.Release()
	selected, err := eager.scratchForRows(4)
	if err != nil {
		t.Fatal(err)
	}
	if eager.scratchRows != eager.maxLen || selected.dx != eager.dx || eager.fullScratch.rows != 0 {
		t.Fatalf("eager control rows=%d selectedResident=%v overflow=%d, want %d/true/0",
			eager.scratchRows, selected.dx == eager.dx, eager.fullScratch.rows, eager.maxLen)
	}
}

func TestDecoderScratchPartialAllocationFailureReleasesGeneration(t *testing.T) {
	d := decoderScratchFixture(false)
	wantErr := errors.New("scratch allocation failed")
	var made []*gptStorageBuffer
	calls := 0
	d.ops.newBuffer = func(data []float32) (buffer, error) {
		calls++
		if calls == 4 {
			return nil, wantErr
		}
		b := &gptStorageBuffer{n: len(data)}
		made = append(made, b)
		return b, nil
	}
	if got, err := d.newScratch(4); got.rows != 0 || !errors.Is(err, wantErr) {
		t.Fatalf("failed workspace = (rows %d, %v), want (0, %v)", got.rows, err, wantErr)
	}
	if len(made) != 3 {
		t.Fatalf("partial workspace allocated %d buffers, want 3", len(made))
	}
	for i, b := range made {
		if b.releases != 1 {
			t.Fatalf("partial buffer %d releases = %d, want 1", i, b.releases)
		}
	}
}

func TestDecoderBatchEmbedHostGrowthReuseAndRelease(t *testing.T) {
	d := &Decoder{d: 8}
	first := d.batchEmbedHost(4)
	if len(first) != 32 || len(d.embedBatchHost) != 32 {
		t.Fatalf("first staging lengths = %d/%d, want 32/32", len(first), len(d.embedBatchHost))
	}
	first[0] = 17
	reused := d.batchEmbedHost(2)
	if len(reused) != 16 || len(d.embedBatchHost) != 32 || reused[0] != 17 {
		t.Fatalf("smaller staging len/high-water/first = %d/%d/%v, want 16/32/17",
			len(reused), len(d.embedBatchHost), reused[0])
	}
	grown := d.batchEmbedHost(6)
	if len(grown) != 48 || len(d.embedBatchHost) != 48 || &grown[0] == &first[0] {
		t.Fatalf("grown staging len/high-water/reused = %d/%d/%v, want 48/48/false",
			len(grown), len(d.embedBatchHost), &grown[0] == &first[0])
	}
	d.Release()
	if d.embedBatchHost != nil {
		t.Fatalf("Release retained %d batched embedding elements", len(d.embedBatchHost))
	}
}

func TestDecoderPrefillIntoRejectsOutputLengthBeforeExecution(t *testing.T) {
	d := &Decoder{ops: backendOps{name: "guard-test"}, v: 7, maxLen: 8, mamba: true}
	if err := d.StepNInto([]int{1, 2}, 0, make([]float32, 13)); err == nil {
		t.Fatal("StepNInto accepted a short destination")
	}
	if err := d.StepNLastInto([]int{1, 2}, 0, make([]float32, 8)); err == nil {
		t.Fatal("StepNLastInto accepted a long destination")
	}
	if d.embedBatchHost != nil {
		t.Fatal("destination guards reached prefill staging")
	}
}

func TestDecoderScratchOptionalPathShapes(t *testing.T) {
	for _, tc := range []struct {
		name      string
		configure func(*Decoder)
		want      map[string]int
	}{
		{
			name: "quantized-residuals",
			configure: func(d *Decoder) {
				d.qweights = []qweight{&residualScratchQWeight{}}
			},
			want: map[string]int{"ao": 32, "mo": 32},
		},
		{
			name: "fused-moe",
			configure: func(d *Decoder) {
				d.ops.fusedGateUp = true
				d.moe, d.nExperts = true, 6
			},
			want: map[string]int{"mo": 32, "gu": 128, "moeGate": 24, "moeW": 24, "moeCol": 4},
		},
		{
			name: "mla",
			configure: func(d *Decoder) {
				d.mla = true
				d.h, d.qLoRA, d.kvLoRA = 2, 3, 5
				d.qkNope, d.qkRope, d.vHead, d.qkHead = 2, 2, 3, 4
			},
			want: map[string]int{
				"mlaCQ": 12, "mlaQ": 32, "mlaComp": 28,
				"mlaLatent": 20, "mlaKV": 40, "mlaAttn": 24,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := decoderScratchFixture(false)
			tc.configure(d)
			s, err := d.newScratch(4)
			if err != nil {
				t.Fatal(err)
			}
			defer s.release()
			buffers := decoderScratchBuffers(s)
			for name, want := range tc.want {
				if got := buffers[name].n; got != want {
					t.Fatalf("%s elements = %d, want %d", name, got, want)
				}
			}
		})
	}
}

func BenchmarkDecoderScratchResidency(b *testing.B) {
	for _, tc := range []struct {
		name  string
		eager bool
		rows  int
	}{
		{name: "lazy", rows: 1},
		{name: "eager-control", eager: true, rows: 2048},
	} {
		b.Run(tc.name, func(b *testing.B) {
			residentBytes := 0
			b.ReportAllocs()
			for range b.N {
				d := decoderScratchFixture(tc.eager)
				d.maxLen, d.d, d.qDim, d.kvDim, d.hidden = 2048, 2048, 2048, 256, 5632
				var err error
				d.allocScratch(d.mkBuf(&err))
				if err != nil {
					b.Fatal(err)
				}
				residentBytes = decoderScratchElements(d.residentScratch()) * 4
				if d.scratchRows != tc.rows {
					b.Fatalf("resident scratch rows = %d, want %d", d.scratchRows, tc.rows)
				}
				d.Release()
			}
			b.ReportMetric(float64(residentBytes), "resident-scratch-B")
		})
	}
}
