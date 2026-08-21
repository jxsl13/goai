//go:build arm64 && goexperiment.simd

package cpu

// mhaFwdBandRows is the row-band grain of the dynamic forward scheduler.
// Thirty rows align the amd64 6-row AVX2 tile but leave two scalar rows after
// every 4-row ARM64 NEON tile band. A physical-M2 sweep selected 32 over
// aligned candidates 24, 28, 36, and 40 while retaining enough tasks to
// balance causal work across heterogeneous cores.
const mhaFwdBandRows = 32
