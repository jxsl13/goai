## §G — goals

G1: idiomatic modular Go lib ∀ AI domains — linalg, autograd, classic-ML, deep-learning, NLP/LLM-inference, CV, RL, probabilistic.

G2: core ops → as close as possible to C/C++ refs (Eigen, OpenBLAS, oneDNN, ggml, ONNX Runtime, PyTorch/ATen) via Pure-Go-SIMD; GPU/NPU only where needed.

G3: correctness before speed. ∀ op → valid Pure-Go reference first, optimize as separate step.

G4: math/scientific grounding ! per unit; numeric decisions (stability, accuracy, overflow) documented ≠ implicit.

G5: numeric parity = acceptance. "done" → results match ref within fixed tolerance, proven ≠ claimed.

G6: perf measurable | ⊥ exist. no "faster" without benchmark + baseline compare.

G7: native macOS & Windows & Linux on CPU & GPU & (later) NPU. ∀ accel op → Pure-Go fallback runs everywhere.

"vollwertig" (measurable): L0–L3 green + parity ∀ ops; ≥1 optimized CPU backend beating scalar ref w/ V-BENCH numbers; safetensors IO; ≥1 end-to-end trained model converging; ≥1 GPU backend gated by V-CGO; ≥1 LLM inference path.
