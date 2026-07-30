package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// AQLM — Additive Quantization of Language Models (Egiazarian, Panferov, Kuznedelev,
// Frantar, Babenko & Alistarh 2024, "Extreme Compression of Large Language Models via
// Additive Quantization", arXiv:2401.06118, ICML'24). AQLM pushes weight quantization into
// the 2–3 bit regime where scalar (GPTQ/AWQ/HQQ) quantizers collapse, by borrowing Additive
// Quantization / Multi-Codebook Quantization (MCQ) from vector-search (Babenko & Lempitsky
// 2014). Each weight ROW is cut into contiguous groups of g scalars, and every group vector
// is approximated NOT by a single codebook entry (that is plain vector quantization) but by a
// SUM of one entry drawn from each of M learned codebooks:
//
//	Ŵ_group = Σ_{m=1..M} C_m[code_m],   C_m ∈ ℝ^{2^B × g},   code_m ∈ {0 … 2^B−1}
//
// The stored form is M small codes per group (M·B bits) plus the M shared codebooks
// (amortized across the whole matrix), so the effective rate is:
//
//	bits/weight ≈ M·B / g   (+ the amortized 2^B·g·M codebook floats, negligible at scale)
//
// This is L3's most aggressive post-training weight quantizer. The representation, the ENCODE
// fit (residual init → iterated code refinement → least-squares codebook refit), and a
// QuantLinear-style Forward are implemented here as pure f64. The calibration/fine-tuning
// stage of the paper — jointly tuning codebooks against layer activations to recover accuracy
// — is a documented follow-up, not built here; EncodeAQLM is data-free (it fits the weight
// matrix itself, like nn.HQQuantize), so quality at a fixed rate trails the paper's calibrated
// numbers while the mechanism and rate-distortion behavior are faithful.
//
// # For the AI professional
//
// The M=1 case degenerates to plain vector quantization (each group ← its single nearest
// codebook entry). Raising M or B monotonically enlarges the code family, so reconstruction
// MSE falls as either grows — the rate-distortion lever this file demonstrates (§V16). The
// encode is the standard MCQ alternation: residual-VQ initialization, per-group Iterated
// Conditional Modes (ICM) to refine the discrete codes, and a global ridge-regularized
// least-squares refit of all codebook entries for the fixed assignment.
//
// Further reading: Egiazarian et al. 2024 (arXiv:2401.06118, the defining paper); Babenko &
// Lempitsky 2014, "Additive Quantization for Extreme Vector Compression" (CVPR) and Martinez
// et al. 2016, "Revisiting Additive Quantization" (ECCV) for the MCQ encode machinery;
// Gray & Neuhoff 1998, "Quantization" (IEEE Trans. Inf. Theory) for rate-distortion
// grounding. Contrast with the scalar quantizers in this package: nn.HQQuantize,
// nn.AWQuantize, nn.GPTQuantize.
type AQLM struct {
	NumCodebooks int         // M: number of additive codebooks summed per group
	Bits         int         // B: bits per code ⇒ each codebook holds 2^B entries
	GroupSize    int         // g: scalars per quantized group (a weight row is cut into Cols/g groups)
	Rows         int         // weight rows (Out features); the matrix is interpreted as [Out, In]
	Cols         int         // weight cols (In features); must be divisible by GroupSize
	Codebooks    [][]float64 // M·2^B entries, each a length-g vector; codebook m entry k at index m·2^B+k
	Codes        []int       // Rows·(Cols/g)·M codes; group gi codebook m at index gi·M+m, gi = row·(Cols/g)+groupCol
	dtype        tensor.Dtype
}

// AQLMOption configures EncodeAQLM (functional-options idiom, §C12).
type AQLMOption func(*aqlmCfg)

type aqlmCfg struct {
	m     int     // codebooks M
	bits  int     // bits per code B
	g     int     // group size
	iters int     // alternating (refit + re-encode) rounds
	seed  uint64  // RNG seed for the deterministic k-means initialization
	ridge float64 // Tikhonov regularization on the codebook least-squares refit
}

// WithAQLMCodebooks sets M, the number of additive codebooks whose entries are summed to
// reconstruct each weight group.
//
// In plain terms: how many "layers" of correction stack up to approximate each little block of
// weights. One codebook is ordinary vector quantization (nearest entry); each extra codebook
// adds another vector to the sum, shrinking the leftover error — so more codebooks means a
// finer fit at a proportionally higher bit-rate (bits/weight = M·B/g). Boundary behavior —
// M=1 is exactly plain vector quantization (the known-behavior anchor); as M grows,
// reconstruction MSE keeps dropping but with diminishing returns while the rate rises linearly
// and encoding gets slower. Must be ≥ 1.
//
// Default 2 (research-grounded: the AQLM "2×8" configuration of Egiazarian et al. 2024 — two
// 8-bit codebooks over groups of 8 = 2 bits/weight — is the paper's headline extreme-quant
// operating point, §R AQLM).
func WithAQLMCodebooks(m int) AQLMOption { return func(c *aqlmCfg) { c.m = m } }

// WithAQLMBits sets B, the bits per code, so each codebook holds 2^B entries.
//
// In plain terms: how many distinct vectors each codebook can choose from — B=8 gives 256
// candidate vectors per codebook. More bits = a richer codebook and a closer fit, at a higher
// rate (bits/weight = M·B/g) and a quadratically larger codebook to store and search.
// Boundary behavior — small B (2–4) is the extreme-compression regime where AQLM shines over
// scalar quantizers; large B approaches full-precision per group but the codebook itself
// (2^B·g floats) stops being negligible and encode cost grows. Must be ≥ 1; with M=1 and large
// B the scheme is high-resolution plain vector quantization.
//
// Default 8 (research-grounded: 8-bit codebooks are the AQLM reference entry width — 256
// entries balances fit quality against codebook size and search cost, §R AQLM).
func WithAQLMBits(b int) AQLMOption { return func(c *aqlmCfg) { c.bits = b } }

// WithAQLMGroupSize sets g, the number of contiguous weights per quantized group (each group
// is the vector that the additive codebooks jointly approximate).
//
// In plain terms: how many neighboring weights share one set of codes. Smaller groups are
// quantized more precisely (fewer weights forced to agree) but spend more bits per weight
// (bits/weight = M·B/g); larger groups compress harder at the cost of accuracy. Boundary
// behavior — g=1 is scalar quantization (no vector structure to exploit, AQLM's advantage
// vanishes); very large g dilutes the bit budget until the fit degrades. The weight's input
// dimension (Cols) MUST be divisible by g, else EncodeAQLM errors.
//
// Default 8 (research-grounded: the AQLM reference group size — with 8-bit codebooks it yields
// the paper's 2-bit "2×8" and 4-bit rates while keeping codebook search tractable, §R AQLM).
func WithAQLMGroupSize(g int) AQLMOption { return func(c *aqlmCfg) { c.g = g } }

// WithAQLMIters sets the number of alternating refinement rounds — each round refits all
// codebooks by least squares (codes fixed) then re-encodes every group's codes (codebooks
// fixed) — run after the residual-VQ initialization.
//
// In plain terms: how many polish passes the fitter runs after its first rough guess; each
// pass jointly re-tunes the codebooks and re-picks the codes to lower the reconstruction
// error. Boundary behavior — SPECIAL VALUE 0 keeps the residual-VQ initialization only (a
// fast, greedy fit, useful as a baseline); a handful of rounds captures most of the gain and
// the error then plateaus, so extra rounds mostly cost time. Must be ≥ 0.
//
// Default 10 (research-grounded: MCQ code/codebook alternation converges within a few
// iterations — Babenko & Lempitsky 2014; Martinez et al. 2016 — 10 is a safe plateau for the
// data-free fit, §R AQLM).
func WithAQLMIters(n int) AQLMOption { return func(c *aqlmCfg) { c.iters = n } }

// WithAQLMSeed sets the RNG seed for the k-means codebook initialization, making EncodeAQLM
// bit-for-bit reproducible.
//
// In plain terms: the fit starts from a random pick of initial codebook centers; this seed
// fixes that pick so re-running the same encode yields identical codes and codebooks.
// Boundary behavior — any value gives a deterministic, reproducible result; different seeds
// explore different initializations and may converge to slightly different local optima (the
// MCQ objective is non-convex), all at the same bit-rate.
//
// Default 0 (a fixed constant — determinism out of the box; §V16 round-trip reproducibility).
func WithAQLMSeed(seed uint64) AQLMOption { return func(c *aqlmCfg) { c.seed = seed } }

// EncodeAQLM fits an additive-quantization representation to the weight matrix w[Out, In]
// (interpreted as a torch-style linear weight: rows = output features, cols = input features).
// It splits each row into groups of GroupSize scalars and fits M codebooks of 2^B entries plus
// one code per (group, codebook) so that Σ_m C_m[code_m] ≈ each group.
//
// Algorithm (data-free, pure f64):
//
//  1. Residual-VQ init: codebook 1 = k-means centroids of the group vectors; assign each group
//     its nearest entry and subtract it; codebook 2 = k-means of the residuals; and so on for M
//     codebooks. This greedy pass alone is a valid (if rough) additive code.
//  2. Refinement: WithAQLMIters rounds, each (a) refits ALL codebook entries jointly by
//     ridge-regularized least squares for the current fixed code assignment (the global optimum
//     for those codes), then (b) re-encodes every group's M codes by Iterated Conditional Modes
//     — sweeping the codebooks, each time picking the entry nearest to the group minus the other
//     codebooks' current contributions (the MCQ beam/ICM encode; ICM = beam width 1).
//
// It returns the fitted *AQLM. Reconstruct recovers Ŵ; Forward applies it as a linear layer.
// Errors on a non-rank-2 weight, a non-positive M/B/g, or an In dimension not divisible by g.
func EncodeAQLM(w *tensor.Tensor, opts ...AQLMOption) (*AQLM, error) {
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: EncodeAQLM needs a rank-2 weight [out,in], got %v", w.Shape())
	}
	cfg := aqlmCfg{m: 2, bits: 8, g: 8, iters: 10, seed: 0, ridge: 1e-3}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.m < 1 || cfg.bits < 1 || cfg.g < 1 {
		return nil, fmt.Errorf("nn: EncodeAQLM needs codebooks≥1, bits≥1, groupSize≥1, got M=%d B=%d g=%d", cfg.m, cfg.bits, cfg.g)
	}
	rows, cols := w.Shape()[0], w.Shape()[1]
	if cols%cfg.g != 0 {
		return nil, fmt.Errorf("nn: EncodeAQLM in-dim %d not divisible by group size %d", cols, cfg.g)
	}
	k := 1 << cfg.bits
	gpr := cols / cfg.g // groups per row
	ng := rows * gpr    // total groups

	// materialize the group vectors (each length g).
	groups := make([][]float64, ng)
	for r := range rows {
		for gc := range gpr {
			vec := make([]float64, cfg.g)
			for t := range cfg.g {
				vec[t] = w.AtF64(r, gc*cfg.g+t)
			}
			groups[r*gpr+gc] = vec
		}
	}

	rng := rand.New(rand.NewPCG(cfg.seed, cfg.seed^0x9E3779B97F4A7C15))
	codebooks := make([][]float64, cfg.m*k)
	codes := make([]int, ng*cfg.m)

	// (1) residual-VQ initialization: fit codebook m to the residual left by codebooks 1..m−1.
	residual := make([][]float64, ng)
	for i := range groups {
		residual[i] = append([]float64(nil), groups[i]...)
	}
	for m := range cfg.m {
		cb := kmeansAQLM(residual, k, cfg.g, rng, 10)
		for j := range k {
			codebooks[m*k+j] = cb[j]
		}
		for i := range residual {
			b := nearestAQLM(residual[i], cb)
			codes[i*cfg.m+m] = b
			for t := range cfg.g {
				residual[i][t] -= cb[b][t]
			}
		}
	}

	// (2) alternate global codebook refit (codes fixed) and per-group re-encode (codebooks
	// fixed). Ending on re-encode keeps the codes optimal for the final refit codebooks.
	for range cfg.iters {
		refitCodebooksAQLM(groups, codes, codebooks, cfg.m, k, cfg.g, cfg.ridge)
		icmEncodeAQLM(groups, codes, codebooks, cfg.m, k, cfg.g)
	}

	return &AQLM{
		NumCodebooks: cfg.m,
		Bits:         cfg.bits,
		GroupSize:    cfg.g,
		Rows:         rows,
		Cols:         cols,
		Codebooks:    codebooks,
		Codes:        codes,
		dtype:        w.Dtype(),
	}, nil
}

// Reconstruct materializes the approximate weight Ŵ[Rows, Cols] by summing, for every group,
// the chosen entry of each codebook: Ŵ_group = Σ_m C_m[code_m]. The result carries the dtype
// of the weight EncodeAQLM was fit on.
func (q *AQLM) Reconstruct() *tensor.Tensor {
	w := tensor.New(q.dtype, tensor.Shape{q.Rows, q.Cols})
	gpr := q.Cols / q.GroupSize
	k := 1 << q.Bits
	acc := make([]float64, q.GroupSize)
	for r := range q.Rows {
		for gc := range gpr {
			gi := r*gpr + gc
			for t := range q.GroupSize {
				acc[t] = 0
			}
			for m := range q.NumCodebooks {
				cb := q.Codebooks[m*k+q.Codes[gi*q.NumCodebooks+m]]
				for t := range q.GroupSize {
					acc[t] += cb[t]
				}
			}
			for t := range q.GroupSize {
				w.SetF64(acc[t], r, gc*q.GroupSize+t)
			}
		}
	}
	return w
}

// BitsPerWeight returns the effective code rate M·B/g bits per weight — the payload cost,
// excluding the M·2^B·g codebook floats that are shared across the whole matrix and so amortize
// to near-zero on real (large) weights. This is the rate axis of the rate-distortion trade-off
// (§V16): raising it (more codebooks or bits, or a smaller group) lowers reconstruction MSE.
func (q *AQLM) BitsPerWeight() float64 {
	return float64(q.NumCodebooks*q.Bits) / float64(q.GroupSize)
}

// Forward computes y[batch, Out] = x[batch, In] · Ŵᵀ using the reconstructed weight — the
// QuantLinear-style application of an AQLM-quantized layer, matching torch nn.Linear semantics
// (weight stored [Out, In], output = x·Wᵀ) with no bias. It runs the matmul through the active
// backend, so results match a dense nn.Linear built from Reconstruct() within the
// reconstruction tolerance (§V16 forward parity). x's inner dimension must equal Cols (In).
func (q *AQLM) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != q.Cols {
		return nil, fmt.Errorf("nn: AQLM.Forward x must be [batch,%d], got %v", q.Cols, x.Shape())
	}
	wHat := q.Reconstruct() // [Out, In]
	// transpose to [In, Out] so a plain row-major matmul computes x·Ŵᵀ.
	wT := tensor.New(x.Dtype(), tensor.Shape{q.Cols, q.Rows})
	for r := range q.Rows {
		for c := range q.Cols {
			wT.SetF64(wHat.AtF64(r, c), c, r)
		}
	}
	out, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, wT}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// nearestAQLM returns the index of the codebook entry closest (squared-Euclidean) to x.
func nearestAQLM(x []float64, cent [][]float64) int {
	best, bd := 0, math.Inf(1)
	j := 0
	for ; j+4 <= len(cent); j += 4 {
		c0, c1, c2, c3 := cent[j], cent[j+1], cent[j+2], cent[j+3]
		var d0, d1, d2, d3 float64
		for t := range x {
			xv := x[t]
			e0 := xv - c0[t]
			d0 += e0 * e0
			e1 := xv - c1[t]
			d1 += e1 * e1
			e2 := xv - c2[t]
			d2 += e2 * e2
			e3 := xv - c3[t]
			d3 += e3 * e3
		}
		if d0 < bd {
			bd, best = d0, j
		}
		if d1 < bd {
			bd, best = d1, j+1
		}
		if d2 < bd {
			bd, best = d2, j+2
		}
		if d3 < bd {
			bd, best = d3, j+3
		}
	}
	for ; j < len(cent); j++ {
		var d float64
		for t := range x {
			diff := x[t] - cent[j][t]
			d += diff * diff
		}
		if d < bd {
			bd, best = d, j
		}
	}
	return best
}

// kmeansAQLM runs Lloyd's algorithm to fit k centroids of dimension dim to data, seeded
// deterministically from rng. Empty clusters keep their previous centroid (deterministic); if
// there are fewer points than clusters, initial centroids repeat points and unused ones are
// zeroed out later by the least-squares refit.
func kmeansAQLM(data [][]float64, k, dim int, rng *rand.Rand, iters int) [][]float64 {
	n := len(data)
	cent := make([][]float64, k)
	perm := rng.Perm(max(n, 1))
	for i := range k {
		if n == 0 {
			cent[i] = make([]float64, dim)
			continue
		}
		cent[i] = append([]float64(nil), data[perm[i%n]]...)
	}
	for range iters {
		sums := make([][]float64, k)
		cnt := make([]int, k)
		for i := range sums {
			sums[i] = make([]float64, dim)
		}
		for _, x := range data {
			b := nearestAQLM(x, cent)
			cnt[b]++
			for t := range dim {
				sums[b][t] += x[t]
			}
		}
		for i := range k {
			if cnt[i] > 0 {
				for t := range dim {
					cent[i][t] = sums[i][t] / float64(cnt[i])
				}
			}
		}
	}
	return cent
}

// icmEncodeAQLM refines each group's M codes by Iterated Conditional Modes (MCQ encode, beam
// width 1): sweeping the codebooks, it repeatedly re-selects code_m as the entry nearest to the
// group minus the sum of the OTHER codebooks' current contributions. A few sweeps converge.
func icmEncodeAQLM(groups [][]float64, codes []int, codebooks [][]float64, m, k, g int) {
	// Each group's re-encode is self-contained: it writes only codes[i*m : i*m+m] and reads
	// groups[i] plus the shared read-only codebooks, so the group loop fans out over GOMAXPROCS
	// bit-identically to the serial loop (the entry scan's argmin is deterministic — ascending-j
	// strict-< ties to the lowest j regardless of worker interleaving). Each worker owns a private
	// target scratch. Gated on ng·(3·m·k·g) so a tiny re-encode stays serial. This is the
	// parallelizable half of EncodeAQLM's refit/re-encode alternation.
	parallelRows(len(groups), 3*m*k*g, func(glo, ghi int) {
		target := make([]float64, g)
		for i := glo; i < ghi; i++ {
			for range 3 { // sweeps
				for a := range m {
					copy(target, groups[i])
					for b := range m {
						if b == a {
							continue
						}
						c := codebooks[b*k+codes[i*m+b]]
						for t := range g {
							target[t] -= c[t]
						}
					}
					best, bd := 0, math.Inf(1)
					j := 0
					for ; j+4 <= k; j += 4 {
						// Unroll-and-jam the entry scan by 4: one target[t] load feeds
						// four independent squared-distance accumulators. Each keeps its
						// own ascending-t sum and the argmin folds in ascending-j strict-<
						// order -> identical winner (first/lowest j on ties). Bit-exact.
						cb0 := codebooks[a*k+j]
						cb1 := codebooks[a*k+j+1]
						cb2 := codebooks[a*k+j+2]
						cb3 := codebooks[a*k+j+3]
						var d0, d1, d2, d3 float64
						for t := range g {
							tv := target[t]
							e0 := tv - cb0[t]
							d0 += e0 * e0
							e1 := tv - cb1[t]
							d1 += e1 * e1
							e2 := tv - cb2[t]
							d2 += e2 * e2
							e3 := tv - cb3[t]
							d3 += e3 * e3
						}
						if d0 < bd {
							bd, best = d0, j
						}
						if d1 < bd {
							bd, best = d1, j+1
						}
						if d2 < bd {
							bd, best = d2, j+2
						}
						if d3 < bd {
							bd, best = d3, j+3
						}
					}
					for ; j < k; j++ {
						cb := codebooks[a*k+j]
						var d float64
						for t := range g {
							diff := target[t] - cb[t]
							d += diff * diff
						}
						if d < bd {
							bd, best = d, j
						}
					}
					codes[i*m+a] = best
				}
			}
		}
	})
}

// refitCodebooksAQLM re-solves ALL codebook entries jointly for the fixed code assignment by
// minimizing Σ_i ‖group_i − Σ_m C_m[code_{i,m}]‖². Stacking every codebook entry as a row of an
// (M·2^B)×g unknown X and the per-group one-hot selections as rows of A, the least-squares
// normal equations are (AᵀA + ridge·I)·X = AᵀG, solved by Gauss-Jordan elimination. The ridge
// term keeps the system non-singular when some entries are unused (which then refit to zero).
func refitCodebooksAQLM(groups [][]float64, codes []int, codebooks [][]float64, m, k, g int, ridge float64) {
	n := m * k
	// ata (n×n) and atg (n×g) each backed by ONE contiguous buffer — the [][] view still
	// hands solveLinearAQLM its rows, but this drops the 2n per-row allocations per refit
	// (× iters). Bit-identical: same values, only the row backing changes.
	ataFlat := make([]float64, n*n)
	atgFlat := make([]float64, n*g)
	ata := make([][]float64, n)
	atg := make([][]float64, n)
	for i := range n {
		ata[i] = ataFlat[i*n : i*n+n : i*n+n]
		atg[i] = atgFlat[i*g : i*g+g : i*g+g]
	}
	cols := make([]int, m)
	for i := range groups {
		for a := range m {
			cols[a] = a*k + codes[i*m+a]
		}
		for a := range m {
			pa := cols[a]
			for t := range g {
				atg[pa][t] += groups[i][t]
			}
			for b := range m {
				ata[pa][cols[b]]++
			}
		}
	}
	for p := range n {
		ata[p][p] += ridge
	}
	x := solveLinearAQLM(ata, atg)
	for p := range n {
		copy(codebooks[p], x[p])
	}
}

// solveLinearAQLM solves the linear system A·X = b (A is n×n, b is n×rhs) by Gauss-Jordan
// elimination with partial pivoting, returning X (n×rhs). A near-zero pivot (a fully
// regularized-away row) leaves that unknown at zero.
func solveLinearAQLM(a, b [][]float64) [][]float64 {
	n := len(a)
	rhs := len(b[0])
	// One contiguous augmented matrix [n, n+rhs] instead of n separately-allocated rows:
	// Gauss-Jordan does column pivoting (aug[r][c], r varying) and full-matrix elimination,
	// both of which walk ACROSS rows, so a single constant-stride block is cache-friendly
	// and drops n allocations. Bit-identical — index arithmetic only, and the O(1) row-
	// POINTER swap becomes a physical swap of the two stride-length segments (same logical
	// matrix). This O(n³) solve is the dominant cost of EncodeAQLM (iters × refit × solve).
	stride := n + rhs
	aug := make([]float64, n*stride)
	for i := range n {
		copy(aug[i*stride:i*stride+n], a[i])
		copy(aug[i*stride+n:i*stride+stride], b[i])
	}
	swap := make([]float64, stride)
	for c := range n {
		p := c
		for r := c + 1; r < n; r++ {
			if math.Abs(aug[r*stride+c]) > math.Abs(aug[p*stride+c]) {
				p = r
			}
		}
		if p != c {
			copy(swap, aug[c*stride:c*stride+stride])
			copy(aug[c*stride:c*stride+stride], aug[p*stride:p*stride+stride])
			copy(aug[p*stride:p*stride+stride], swap)
		}
		piv := aug[c*stride+c]
		if math.Abs(piv) < 1e-300 {
			continue
		}
		inv := 1 / piv
		crow := aug[c*stride : c*stride+stride]
		for j := c; j < stride; j++ {
			crow[j] *= inv
		}
		for r := range n {
			if r == c {
				continue
			}
			f := aug[r*stride+c]
			if f == 0 {
				continue
			}
			rrow := aug[r*stride : r*stride+stride]
			for j := c; j < stride; j++ {
				rrow[j] -= f * crow[j]
			}
		}
	}
	x := make([][]float64, n)
	for i := range n {
		x[i] = append([]float64(nil), aug[i*stride+n:i*stride+stride]...)
	}
	return x
}
