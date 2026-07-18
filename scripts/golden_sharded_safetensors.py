#!/usr/bin/env python3
"""Sharded-safetensors golden generator (LoadSharded, §V1/§V16 spirit).

Runs the ACTUAL Hugging Face sharding writer — transformers save_pretrained,
which delegates the split to huggingface_hub.split_torch_state_dict_into_shards
and emits model-XXXXX-of-YYYYY.safetensors plus model.safetensors.index.json —
against a tiny seeded LlamaForCausalLM, forcing >= 3 shards via max_shard_size.
The SAME model is also saved unsharded as the byte-equality anchor:

  format/safetensors/testdata/sharded/model.safetensors.index.json
  format/safetensors/testdata/sharded/model-0000N-of-0000M.safetensors
  format/safetensors/testdata/sharded_single.safetensors

Go-side contract anchored here (verified against transformers 5.14.1 source,
modeling_utils.save_pretrained + utils/hub.get_checkpoint_shard_files):

  * index = {"metadata": {"total_parameters": N, "total_size": M},
             "weight_map": {tensor_name: shard_filename, ...}},
    json.dumps(indent=2, sort_keys=True) + trailing newline.
  * The HF LOADER never validates metadata.total_size (it only takes
    sorted(set(weight_map.values())) and joins paths) — so LoadSharded
    mirrors that and treats index metadata as opaque bookkeeping.
  * Shard filenames in weight_map are bare basenames (no path separators);
    the loader resolves them relative to the index's directory.

The model is llama-shaped on purpose (geometry mirrors the llama_hf_gqa
golden: dim 16, heads 4, kv 2) so nlp's LlamaFromHF smoke test can consume
the LoadSharded map end-to-end. transformers init is seeded and non-uniform
(§B68 spirit: no all-equal tensors that would mask permutation bugs).

Run:
  .venv/bin/python scripts/golden_sharded_safetensors.py
"""

import json
import os
import shutil
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

SEED = 11
VOCAB = 32
DIM = 16
LAYERS = 2
HEADS = 4
KV_HEADS = 2
MAX_SHARD_SIZE = "6KB"  # tiny model, tiny cap -> several few-KB shards

DEST = os.path.join(os.path.dirname(__file__), "..", "format", "safetensors", "testdata")


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
        tie_word_embeddings=False,  # tied storage would be deduplicated across shards
        use_cache=False,
    )
    model = LlamaForCausalLM(cfg).eval()  # default f32

    sharded_dir = os.path.join(DEST, "sharded")
    with tempfile.TemporaryDirectory() as tmp:
        # Sharded export: the real HF writer produces the shards + index.
        shard_tmp = os.path.join(tmp, "sharded")
        model.save_pretrained(shard_tmp, max_shard_size=MAX_SHARD_SIZE)
        shards = sorted(
            f for f in os.listdir(shard_tmp) if f.endswith(".safetensors")
        )
        index_name = "model.safetensors.index.json"
        assert os.path.isfile(os.path.join(shard_tmp, index_name)), "expected sharded save"
        assert len(shards) >= 3, f"want >=3 shards, got {shards}"
        with open(os.path.join(shard_tmp, index_name)) as f:
            index = json.load(f)
        assert set(index["weight_map"]) == set(model.state_dict()), "weight_map must be exact"

        # Single-file export of the SAME weights: the equality anchor.
        single_tmp = os.path.join(tmp, "single")
        model.save_pretrained(single_tmp, max_shard_size="10GB")
        single = os.path.join(single_tmp, "model.safetensors")
        assert os.path.isfile(single), "expected single-file save"

        # Commit only the weight files (not config.json etc.).
        shutil.rmtree(sharded_dir, ignore_errors=True)
        os.makedirs(sharded_dir)
        for f in shards + [index_name]:
            shutil.copy(os.path.join(shard_tmp, f), os.path.join(sharded_dir, f))
        shutil.copy(single, os.path.join(DEST, "sharded_single.safetensors"))

    total = sum(
        os.path.getsize(os.path.join(sharded_dir, f)) for f in os.listdir(sharded_dir)
    )
    print(
        f"wrote {len(shards)} shards + index ({total} bytes) and "
        f"sharded_single.safetensors: torch {torch.__version__}, "
        f"total_size={index['metadata']['total_size']}"
    )


if __name__ == "__main__":
    main()
