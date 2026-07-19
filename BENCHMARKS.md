# GoAI benchmark matrix

This is the canonical entry point for comparing GoAI with established machine
learning libraries and deployment software. It separates measurements that
exist today from comparisons that are planned. A blank cell is never a zero,
and an old result is never presented as a current ranking.

> **Current state:** coverage is partial. The repository already contains
> same-machine GoAI, NumPy, PyTorch and llama.cpp measurements, but it does not
> yet contain a complete cross-framework or serving matrix. The missing work is
> booked explicitly in [`SPEC.md`](SPEC.md) as T879–T888. Detailed historical
> measurements and optimization notes remain in
> [`docs/benchmarking.md`](docs/benchmarking.md).

## Status vocabulary

- **Measured** means a reproducible, same-machine comparison and its inputs are
  present in the repository.
- **Partial** means some required systems or workloads are measured, but the
  category is not complete.
- **Planned** means the comparison has a concrete SPEC task but no publishable
  result yet.
- **Unsupported** means GoAI does not currently expose the capability. It does
  not mean zero performance.

## Coverage at a glance

| Category | Required comparison set | Status | Evidence or task |
| --- | --- | --- | --- |
| Core tensors and automatic differentiation | NumPy, PyTorch, JAX, TensorFlow, MLX, ONNX (Open Neural Network Exchange) Runtime | Partial | Existing NumPy/PyTorch snapshot in [`docs/benchmarking.md`](docs/benchmarking.md); T879, T881 |
| Local large-language-model inference | llama.cpp, MLX-LM (MLX language-model tooling), Hugging Face Transformers | Partial | Same-weight llama.cpp snapshot in [`docs/benchmarking.md`](docs/benchmarking.md); T882 |
| Large-language-model serving | vLLM, SGLang, TensorRT-LLM, llama.cpp server, LMDeploy, Text Generation Inference | Planned; GoAI has no production Hypertext Transfer Protocol server | T883 |
| Training and model workloads | PyTorch, JAX, TensorFlow, MLX, Hugging Face Transformers | Partial inside GoAI; external matrix planned | T884 |
| Tokenization | tiktoken, Hugging Face Tokenizers, SentencePiece | Partial historical generative pre-trained transformer (GPT-2) evidence | T885 |
| Tensor and model formats | Hugging Face safetensors, NumPy, GGUF (llama.cpp's single-file model format) via llama.cpp/gguf-py, PyTorch safe loading | Partial historical evidence | T886 |
| Classic machine learning and reinforcement learning | scikit-learn, XGBoost, LightGBM (gradient-boosting-machine library), LIBSVM (support-vector-machine library), Faiss, Stable-Baselines3 | Partial historical evidence | T887 |
| Product and deployment surfaces | Ollama, LMDeploy | Planned | T888 |

The required set above is GoAI's operational definition of “major incumbent”:
a widely deployed reference or specialized performance floor with an official,
automatable interface and meaningful overlap with GoAI. A new incumbent is
added by citing its official harness or application programming interface and
booking a SPEC task. Graphical-only products and hosted services are not ranked
unless their version, model, request contract and raw results can be pinned.

## Fair-comparison contract

Every published ranking must satisfy all of these rules:

1. **Same system.** Ranked rows run on the same machine, operating system,
   power mode and accelerator. Results from different machines are labeled as
   independent snapshots, not compared as a league table.
2. **Correctness first.** Outputs, gradients, token identifiers or quality
   metrics pass the category's parity gate before timing starts. Faster wrong
   answers do not enter the table.
3. **Same work.** Shapes, data type, batch size, thread count, model weights,
   tokenizer, prompt and output budget are identical. Any conversion or
   quantization is named and checked.
4. **Separate execution contracts.** Cold load/compile/transfer, warm
   end-to-end execution and device-resident kernel execution are different
   rows. Memory mapping, owned materialization and decompression are different
   format rows.
5. **Synchronize asynchronous devices.** A graphics processing unit result is
   synchronized before the timer stops. Just-in-time compilation is warmed and
   reported separately. This follows the official
   [PyTorch benchmark guidance](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
   and [JAX benchmark guidance](https://docs.jax.dev/en/latest/benchmarking.html).
6. **Repeat and disclose variance.** Record warm-up, repetition count, raw
   samples, median and spread. Interleave paired baseline/change (A/B) runs when
   comparing a change with its baseline.
7. **Report memory and failures.** Include peak or resident memory where it
   matters, plus error and non-convergence counts. Missing support is written as
   `unsupported`; an unexecuted cell is `not measured`.
8. **Pin provenance.** Record GoAI commit, competitor version or commit,
   compiler/runtime, hardware, driver, seed and every material flag.

The industry-wide [MLPerf Inference](https://docs.mlcommons.org/inference/)
and [MLPerf Training](https://mlcommons.org/benchmarks/training/) suites inform
the scenario and accuracy discipline. GoAI results must say
“MLPerf-inspired” unless they follow the complete official rules and submission
process; this repository currently makes no MLPerf-compliance claim.

## Required result record

| Field group | Required values |
| --- | --- |
| Identity | category, workload, GoAI commit, competitor name and version/commit |
| System | processor, accelerator, memory, operating system, driver, power mode |
| Workload | dataset or fixture digest, shape/model, data type, quantization, batch, threads, seed |
| Execution | cold or warm, host/device residency, synchronization point, compile and transfer inclusion |
| Sampling | warm-up, repetitions, raw samples, median and spread |
| Correctness | parity or quality metric, tolerance or target, pass/fail |
| Performance | latency or throughput with unit, peak/resident memory, failures |

T879 tracks the machine-readable form of this record and the renderer that will
replace hand-maintained result tables.

## Category protocols

### Core tensors and automatic differentiation

GoAI variants are the readable reference backend, optimized central processing
unit backend, Metal, Vulkan and CUDA (Compute Unified Device Architecture).
The mandatory external set is
[NumPy](https://numpy.org/doc/stable/),
[PyTorch](https://docs.pytorch.org/),
[JAX](https://docs.jax.dev/),
[TensorFlow](https://www.tensorflow.org/),
[MLX](https://github.com/ml-explore/mlx) and
[ONNX (Open Neural Network Exchange) Runtime](https://onnxruntime.ai/docs/performance/).

The matrix covers matrix multiplication, convolution, multi-head attention,
FlashAttention, normalization, softmax, elementwise operations and reductions.
Forward, backward and optimizer-step rows appear only where both systems expose
the same operation. Eager and compiled paths are separate. See T881.

### Local large-language-model inference

The engine comparison uses identical redistributable weights and tokenizer
artifacts against [llama.cpp](https://github.com/ggml-org/llama.cpp),
[MLX-LM](https://github.com/ml-explore/mlx-lm) and
[Hugging Face Transformers](https://huggingface.co/docs/transformers/).
It reports load time, prompt processing, token generation, context-length
sweeps, key/value-cache memory and prompt-cache behavior. Tokenization and
sampling are excluded in engine-only rows and included in separate end-to-end
rows. The official
[`llama-bench` contract](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
is mirrored so its exclusions stay visible. See T882.

### Large-language-model serving

The required engines are
[vLLM](https://docs.vllm.ai/en/stable/cli/index.html),
[SGLang](https://docs.sglang.ai/developer_guide/bench_serving),
[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html),
llama.cpp server,
[LMDeploy](https://lmdeploy.readthedocs.io/en/latest/benchmark/benchmark.html)
and Hugging Face
[Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index).

The matrix uses a shared open-loop load generator and reports request and token
throughput, time to first token, time per output token, inter-token latency,
end-to-end latency percentiles, errors, service-level-objective goodput and
memory over a concurrency sweep. GoAI currently has no production Hypertext
Transfer Protocol server; T883 therefore requires a clearly labeled
benchmark-only adapter and forbids a production-serving claim.

### Training and model workloads

Training compares identical small GPT/Llama, Vision Transformer or convolutional
network, and multilayer-perceptron graphs against PyTorch, JAX, TensorFlow and
MLX. One table measures forward/backward/optimizer-step throughput; another
measures time to a fixed quality target over multiple seeds. Compile time and
peak memory stay separate. See T884.

### Tokenization

GoAI byte-pair encoding, Unigram and WordPiece are compared with
[tiktoken](https://github.com/openai/tiktoken),
[Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) and
[SentencePiece](https://github.com/google/sentencepiece). A speed ranking is
valid only when vocabulary, merge rules, normalization and special-token
semantics are identical. Natural language, code, multilingual Unicode and
hostile whitespace/special-token corpora are measured separately. See T885.

### Tensor and model formats

GoAI safetensors, NumPy array files, NumPy zip archives, GGUF (the llama.cpp
single-file model format) and safe PyTorch loading are compared with the
official safetensors, NumPy, llama.cpp/gguf-py and PyTorch paths. Parse-only,
first-tensor access, full owned loading, memory-mapped views, reading, writing
and accelerator transfer are separate. Warm page-cache and controlled cold-read
numbers are never mixed. See T886.

### Classic machine learning and reinforcement learning

The general classic-machine-learning baseline is
[scikit-learn](https://scikit-learn.org/stable/computing/computational_performance.html),
with specialized floors from [XGBoost](https://xgboost.readthedocs.io/),
[LightGBM](https://lightgbm.readthedocs.io/),
[LIBSVM](https://github.com/cjlin1/libsvm) and
[Faiss](https://github.com/facebookresearch/faiss). Fit/index-build, atomic
prediction and bulk prediction are separate. Approximate nearest-neighbor
search is a recall-at-k versus latency curve, never ranked against exact search
as if they returned the same answer.

Reinforcement learning compares overlapping PPO (proximal policy optimization),
A2C (advantage actor-critic), DDPG (deep deterministic policy gradient) and SAC
(soft actor-critic) workloads with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Environment-step
cost is isolated, while algorithm results report time and steps to a return
target over multiple seeds. See T887.

### Product and deployment surfaces

[Ollama](https://github.com/ollama/ollama) and LMDeploy are measured as
end-to-end products, not as kernel proxies. Model digest, quantization, embedded
runner/backend, API behavior and externally observed wall time are pinned.
These rows stay separate from T882's local-engine and T883's serving-core
rankings. See T888.

## Run the harnesses available today

Run all Go benchmarks without cgo (the Go foreign-function interface):

```sh
make bench
```

On macOS with MoltenVK installed, collect the current GoAI cross-backend and
Python rows, then render the partial table:

```sh
make bench-compare > /tmp/goai-bench.txt
make bench-python >> /tmp/goai-bench.txt
go run ./internal/benchcompare/rendertables /tmp/goai-bench.txt
```

The Python command requires the project virtual environment with NumPy and
PyTorch. `bench-compare` currently covers the reference, central processing
unit, Metal and Vulkan paths; it is not the future all-platform runner tracked
by T879.

On the documented Linux/Vulkan host, collect the pinned llama.cpp side of the
comparison with an explicit GGUF model path:

```sh
./scripts/bench-llamacpp.sh /path/to/model.gguf
```

That script collects a quantized llama.cpp baseline; the companion GoAI CUDA
measurement is run separately and uses a different data type. The script header
explains the asymmetry. The two results must not be presented as an
equal-quantization ranking without T882's corrected contract.

## Further reading

- [Detailed GoAI measurements and optimization history](docs/benchmarking.md)
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)
- [MLPerf Inference methodology](https://docs.mlcommons.org/inference/submission/)
- [MLPerf Training suite](https://mlcommons.org/benchmarks/training/)
- [llama.cpp benchmark tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
