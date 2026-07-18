#!/usr/bin/env python3
"""Real-consumer verification of SaveSharded (§V1 spirit, write direction).

The mirror image of golden_sharded_safetensors.py: there the REAL Hugging Face
writer produces a checkpoint for GoAI's LoadSharded to read; here GoAI's
SaveSharded produces a checkpoint for the REAL Hugging Face consumer to read.

The script runs `go test -run TestSaveShardedPythonExport` with
GOAI_SHARDED_EXPORT_DIR pointing at a temp dir — the Go test writes a >=3-shard
export plus reference.json holding every tensor's dtype/shape/exact values —
then loads the export exactly the way the transformers loader does
(utils/hub.get_checkpoint_shard_files: read index weight_map, open each shard
with the official safetensors library) and asserts:

  * every tensor loads with the expected torch dtype, shape, and bit-exact
    values (f16/bf16 compare exactly — float() widening is lossless);
  * every shard carries the metadata GoAI wrote ({"format": "pt"});
  * shard filenames match model-XXXXX-of-YYYYY.safetensors;
  * the index bytes are EXACTLY what transformers save_pretrained emits for
    that content: json.dumps(index, indent=2, sort_keys=True) + "\\n";
  * metadata.total_size equals the summed tensor data bytes
    (huggingface_hub get_torch_storage_size semantics — not file bytes).

Manual gate, not CI (needs the venv + a Go toolchain). Run from the repo root:

  .venv/bin/python scripts/verify_sharded_export.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import safetensors
import torch
from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TORCH_DTYPES = {
    "F32": torch.float32,
    "F64": torch.float64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GOAI_SHARDED_EXPORT_DIR=tmp)
        subprocess.run(
            ["go", "test", "./format/safetensors/", "-run",
             "TestSaveShardedPythonExport", "-count=1", "-v"],
            cwd=REPO, env=env, check=True,
        )

        index_path = os.path.join(tmp, "model.safetensors.index.json")
        with open(index_path) as f:
            raw = f.read()
        index = json.loads(raw)
        # Byte fidelity: transformers modeling_utils.save_pretrained writes
        # json.dumps(index, indent=2, sort_keys=True) + "\n" — the GoAI index
        # must be byte-identical to that rendering of its own content.
        assert raw == json.dumps(index, indent=2, sort_keys=True) + "\n", \
            "index bytes are not transformers-identical"
        weight_map = index["weight_map"]
        for shard in weight_map.values():
            assert re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", shard), shard

        # Load exactly as the transformers loader does: sorted unique shard
        # list from weight_map, each shard through the official safetensors
        # library.
        shard_files = sorted(set(weight_map.values()))
        assert len(shard_files) >= 3, f"want >=3 shards, got {shard_files}"
        state = {}
        for shard in shard_files:
            with safe_open(os.path.join(tmp, shard), framework="pt") as f:
                assert f.metadata() == {"format": "pt"}, (shard, f.metadata())
                for name in f.keys():
                    assert weight_map[name] == shard, (name, shard)
                    state[name] = f.get_tensor(name)
        assert set(state) == set(weight_map), "weight_map is not exact"

        with open(os.path.join(tmp, "reference.json")) as f:
            ref = json.load(f)
        assert set(ref) == set(state), "tensor names differ from reference"
        total = 0
        for name, r in sorted(ref.items()):
            t = state[name]
            assert t.dtype == TORCH_DTYPES[r["dtype"]], (name, t.dtype)
            assert list(t.shape) == r["shape"], (name, t.shape)
            got = [float(x) for x in t.reshape(-1)]  # C-order, matches Go
            assert got == r["values"], f"{name}: values differ"
            total += t.numel() * t.element_size()
            print(f"  {name}: {r['dtype']} {r['shape']} OK ({t.numel()} values bit-exact)")
        assert index["metadata"] == {"total_size": total}, \
            (index["metadata"], total)

        print(
            f"OK: {len(state)} tensors across {len(shard_files)} shards "
            f"({shard_files[0]}..{shard_files[-1]}) load bit-exactly via "
            f"safetensors {safetensors.__version__} / torch {torch.__version__}; "
            f"index byte-identical to the transformers emitter; "
            f"total_size={total} data bytes verified"
        )


if __name__ == "__main__":
    sys.exit(main())
