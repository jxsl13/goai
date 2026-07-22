package classic

import "testing"

func BenchmarkPCAFit(b *testing.B) {
	x, _ := spatialData(4000, 40)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := &PCA{}
		if err := p.Fit(x, 10); err != nil {
			b.Fatal(err)
		}
	}
}
