# Test model weights (local, gitignored)

The `.gguf` files in this directory are **not committed** (`models/` is in
`.gitignore`).

All state about these assets — the file table with sources, fetch/requantize
commands, loader rules (incl. the §B59 SPM-vs-Unigram tokenizer rule), VRAM/RAM
budgets and per-model validation coverage — lives in the worker runner spec:

**`SPEC-worker-linux-amd64-cuda.md` → §MODELS** (protocol: §RUN).

Quick fetch example (see §MODELS M1 for the full table):

```sh
mkdir -p models
curl -L -o models/tinyllama-1.1b-chat-q8_0.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```
