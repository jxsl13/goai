import os, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import LLM, SamplingParams
try:
    from vllm import TokensPrompt
except Exception:
    TokensPrompt = None

MODEL = os.path.expanduser("~/.local/share/goai-vllm/tinyllama-hf")
BATCH = int(os.environ.get("BATCH", "512"))
CTXS = [128, 256, 384, 512]

llm = LLM(model=MODEL, dtype="float16", enforce_eager=False,
          gpu_memory_utilization=0.90, max_model_len=560,
          disable_log_stats=True)

def mk_prompts(prompt_len):
    ids = [[1] + [500 + (i % 2000) for i in range(prompt_len - 1)] for _ in range(BATCH)]
    if TokensPrompt is not None:
        return [TokensPrompt(prompt_token_ids=x) for x in ids]
    return [{"prompt_token_ids": x} for x in ids]

def gen_time(prompt_len, gen_tokens):
    prompts = mk_prompts(prompt_len)
    sp = SamplingParams(temperature=0.0, max_tokens=gen_tokens, min_tokens=gen_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0

gen_time(128, 4)  # warm
print(f"# vLLM decode marginal throughput, batch={BATCH}, f16, graphs ON, prefill-cancelled", flush=True)
print("# ctx    tok/s", flush=True)
for L in CTXS:
    G1, G2 = 4, 20
    t1 = gen_time(L, G1)
    t2 = gen_time(L, G2)
    dt = t2 - t1
    tps = BATCH * (G2 - G1) / dt if dt > 0 else 0
    print(f"{L:5d}  {tps:8.1f}   (avg ctx ~{L + (G1+G2)//2}, dt={dt:.3f}s)", flush=True)
