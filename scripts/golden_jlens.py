#!/usr/bin/env python3
"""J-lens tier-1 golden generator (§T812, §R250, §V16).

Runs the ACTUAL Anthropic reference implementation (github.com/anthropics/
jacobian-lens, Apache-2.0, installed as `jlens` in the repo .venv) against a
tiny seeded transformers LlamaForCausalLM and exports:

  nlp/testdata/jlens_hf.safetensors   the model weights (f64, HF names) for
                                      the Go side's LlamaFromHF
  nlp/testdata/jlens_corpus.npy       the fit corpus token ids [n_seq, seq] f8
  nlp/testdata/jlens_ref.pt           the RAW JacobianLens.save artifact
                                      (fp16 J, int-keyed "J" dict) — the
                                      JLensFromPT import fixture
  nlp/testdata/jlens_ref_J.npy        the fitted J matrices as stored by the
                                      reference (f32 values), stacked
                                      [n_source_layers, d, d], written f8
  nlp/testdata/jlens_probe.npy        the readout probe token ids [seq] f8
  nlp/testdata/jlens_ref_readout.npy  lens.apply logits: rows 0..K-1 are the
                                      source-layer readouts, row K the model's
                                      own logits, at PROBE_POSITIONS —
                                      [K+1, n_pos, vocab] f8

Everything is seeded and CPU-f64 (the reference's own accumulation is f32,
which sets the fit-parity tolerance on the Go side). Model geometry mirrors
the llama_hf_gqa golden (dim 16, heads 4, kv 2) with 4 layers so the default
reference fit covers source layers 0..2.

Bootstrap (once):
  .venv/bin/pip install git+https://github.com/anthropics/jacobian-lens
Run:
  .venv/bin/python scripts/golden_jlens.py
"""

import os
import random

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

import jlens

SEED = 7
VOCAB = 32
DIM = 16
LAYERS = 4
HEADS = 4
KV_HEADS = 2
SEQ = 24  # corpus sequence length
N_SEQ = 16  # corpus size (quality saturates fast on a tiny model, §R250)
SKIP_FIRST = 2  # reference default 16 is tuned for 128-token web text
PROBE_POSITIONS = [3, 10]

DEST = os.path.join(os.path.dirname(__file__), "..", "nlp", "testdata")


class WordTokenizer:
    """Toy word-level tokenizer: token strings are "w0".."w31", text is
    whitespace-joined. Deterministic; just enough surface for jlens.from_hf
    (__call__ -> input_ids) and the vis helpers (decode)."""

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        ids = [int(w[1:]) for w in text.split()][:max_length]

        class Enc:
            input_ids = torch.tensor([ids])

        return Enc()

    def decode(self, ids, **kw):
        return " ".join(f"w{int(i)}" for i in ids)


def words(ids):
    return " ".join(f"w{i}" for i in ids)


def main():
    torch.manual_seed(SEED)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=DIM,
        intermediate_size=2 * DIM,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    hf = LlamaForCausalLM(cfg).double().eval()
    save_file(
        {k: v.contiguous() for k, v in hf.state_dict().items()},
        os.path.join(DEST, "jlens_hf.safetensors"),
    )

    rng = random.Random(123)
    corpus_ids = [[rng.randrange(VOCAB) for _ in range(SEQ)] for _ in range(N_SEQ)]
    np.save(
        os.path.join(DEST, "jlens_corpus.npy"),
        np.asarray(corpus_ids, dtype=np.float64),
    )

    model = jlens.from_hf(hf, WordTokenizer())
    lens = jlens.fit(
        model,
        [words(ids) for ids in corpus_ids],
        dim_batch=8,
        max_seq_len=SEQ,
        skip_first=SKIP_FIRST,
    )
    assert lens.source_layers == list(range(LAYERS - 1)), lens.source_layers

    # (c) the raw artifact (fp16 J, int-keyed dict) and (a) the exact-as-stored
    # J matrices (reference accumulation is f32; f8 write is lossless).
    lens.save(os.path.join(DEST, "jlens_ref.pt"))
    np.save(
        os.path.join(DEST, "jlens_ref_J.npy"),
        np.stack(
            [lens.jacobians[l].numpy() for l in lens.source_layers]
        ).astype(np.float64),
    )

    # (b) readout probes: lens.apply at fixed (prompt, layers, positions),
    # plus the model's own logits as the final row.
    probe_ids = [rng.randrange(VOCAB) for _ in range(12)]
    np.save(
        os.path.join(DEST, "jlens_probe.npy"),
        np.asarray(probe_ids, dtype=np.float64),
    )
    lens_logits, model_logits, input_ids = lens.apply(
        model, words(probe_ids), positions=PROBE_POSITIONS
    )
    assert input_ids[0].tolist() == probe_ids
    rows = [lens_logits[l].numpy() for l in lens.source_layers]
    rows.append(model_logits.numpy())
    np.save(
        os.path.join(DEST, "jlens_ref_readout.npy"),
        np.stack(rows).astype(np.float64),
    )

    print(
        f"wrote nlp/testdata/jlens_* goldens: torch {torch.__version__}, "
        f"jlens source_layers={lens.source_layers}, n_prompts={lens.n_prompts}, "
        f"probe positions {PROBE_POSITIONS}"
    )


if __name__ == "__main__":
    main()
