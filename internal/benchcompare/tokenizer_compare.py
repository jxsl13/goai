#!/usr/bin/env python3
"""T882 companion: time tiktoken (the Rust-cored industry incumbent) on the EXACT
same corpus and vocab as nlp.BenchmarkGPT2Encode, so GoAI's pure-Go BPE can be
compared apples-to-apples.

Fairness contract:
  * identical corpus  -- the same unit string repeated until >= 1_000_000 bytes,
    reproduced verbatim from bpe_throughput_test.go (1_000_116 bytes).
  * identical vocab   -- tiktoken "gpt2" (r50k_base) is the same GPT-2 BPE that
    nlp.LoadGPT2("testdata/gpt2_ranks.txt") loads; the committed parity golden
    (nlp/testdata/tokenizer.json) already pins GoAI's ids == tiktoken's ids.
  * cross-check       -- this script prints tiktoken's token COUNT; it must equal
    GoAI's count on the same bytes (237208) or the comparison is void.

Run:  .venv/bin/python internal/benchcompare/tokenizer_compare.py
Needs: pip install tiktoken   (version is printed and recorded per V13).
"""

import sys
import time

try:
    import tiktoken
except ImportError:
    sys.exit("tiktoken not installed: run `.venv/bin/pip install tiktoken`")

# Verbatim from nlp/bpe_throughput_test.go -- keep byte-identical.
UNIT = (
    "The quick brown fox jumps over the lazy dog. Tokenization throughput "
    "matters for LLM serving; a pure-Go BPE competes with tiktoken's Rust across megabytes. "
)


def build_corpus() -> str:
    parts = []
    total = 0
    while total < 1_000_000:
        parts.append(UNIT)
        total += len(UNIT)
    return "".join(parts)


def best_of(fn, reps: int = 5) -> float:
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        d = time.perf_counter() - s
        best = min(best, d)
    return best


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")
    text = build_corpus()
    data = text.encode("utf-8")
    nbytes = len(data)

    ids = enc.encode(text)
    ntok = len(ids)

    # encode_ordinary is tiktoken's FASTEST path (no special-token scan); we time
    # it too so the Rust incumbent is measured at its best, not handicapped.
    enc_best = best_of(lambda: enc.encode(text), reps=9)
    ord_best = best_of(lambda: enc.encode_ordinary(text), reps=9)
    dec_best = best_of(lambda: enc.decode(ids), reps=9)

    def line(label, secs):
        return f"{label:22}: {secs*1e3:7.3f} ms  =>  {nbytes/secs/1e6:6.2f} MB/s, {ntok/secs:12,.0f} tok/s"

    print(f"tiktoken     version : {tiktoken.__version__}")
    print(f"python       version : {sys.version.split()[0]}")
    print(f"encoding             : gpt2 (r50k_base)")
    print(f"corpus         bytes : {nbytes}")
    print(f"token          count : {ntok}   <-- must match GoAI (237208)")
    print(line("encode  best-of-9", enc_best))
    print(line("encode_ordinary b/9", ord_best))
    print(line("decode  best-of-9", dec_best))


if __name__ == "__main__":
    main()
