# GoAI

Go-native, full-spectrum AI library — **Pure-Go-first, cgo-last**.

Core operations aim as close as possible to C/C++ references (Eigen, OpenBLAS,
oneDNN, ggml, ONNX Runtime, PyTorch/ATen) using the experimental Go 1.26 SIMD
package on CPU, with GPU/NPU as optional, benchmark-gated backends. Correctness
before speed: every operation ships as a validated Pure-Go reference first, then
optimized as a separate step against that reference.

> Status: **early bootstrap.** Architecture and tasks are driven by
> [`SPEC.md`](SPEC.md) (caveman-encoded, see [`FORMAT.md`](FORMAT.md)). Design
> rationale in [`docs/`](docs/).

## Layout (§I)

| Layer | Package | Role |
|------|---------|------|
| L0 | `tensor` | Tensor, Dtype, Device, Allocator, strides/views |
| L1 | `backend`, `backend/ref` | Backend/Kernel interface + Pure-Go reference (truth) |
| L1b | `backend/*` (build-tagged) | swappable accel backends + fallback |
| L2 | `autograd` | tape/graph + VJP |
| L3 | `nn` | layers, init, optimizer, loss, data |
| L4 | `nlp`, `vision`, `classic`, `rl` | domains |
| L5 | `format` | safetensors, GGUF, ONNX |

## Build

```sh
make build   # CGO_ENABLED=0 pure-Go build (§V7)
make test    # unit + golden + property tests
make bench   # benchmarks feeding the cgo gate (§C3)
```

Requires Go 1.26+. No C toolchain needed for the default build.

## License

TBD.
