"""
Real-model KV cache benchmark using GPT-2 (CPU).

We greedily decode `N_GEN` tokens from a fixed prompt, twice:
  (A) use_cache=False — every step, the model re-attends over the entire
      prefix, which means recomputing K and V for every past position.
  (B) use_cache=True  — model returns past_key_values; on the next step we
      feed only the single new token and reuse the cache.

For each token we record the wall-clock time of one forward pass.
Theory: (A) grows roughly linearly with current context length; (B) is
roughly flat (each step does O(seq_len) attention but only O(1) projections).
"""

import json
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAME = "gpt2"          # 124M params, fine on CPU
N_GEN = 200                   # tokens to generate
PROMPT = "The history of the printing press begins"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def time_forward(fn):
    t0 = time.perf_counter()
    out = fn()
    # CPU torch ops are synchronous, no need for cuda.synchronize
    return out, (time.perf_counter() - t0) * 1000.0  # ms


@torch.no_grad()
def decode_no_cache(model, input_ids, n_gen):
    """Each step: feed the full sequence, take the last logits, append argmax."""
    seq = input_ids.clone()
    per_token_ms = []
    for _ in range(n_gen):
        def step():
            return model(seq, use_cache=False)
        out, ms = time_forward(step)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
        per_token_ms.append(ms)
    return seq, per_token_ms


@torch.no_grad()
def decode_with_cache(model, input_ids, n_gen):
    """Prefill once, then feed only the new token each step using past_key_values."""
    per_token_ms = []
    # Prefill — we charge this to "step 0" so the curves are comparable
    def prefill():
        return model(input_ids, use_cache=True)
    out, ms = time_forward(prefill)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    seq = torch.cat([input_ids, next_tok], dim=1)
    per_token_ms.append(ms)

    for _ in range(n_gen - 1):
        def step():
            return model(next_tok, past_key_values=past, use_cache=True)
        out, ms = time_forward(step)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
        per_token_ms.append(ms)
    return seq, per_token_ms, past


def main():
    torch.set_num_threads(4)
    print(f"Loading {MODEL_NAME} ...")
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).eval()
    cfg = model.config
    print(f"  n_layer={cfg.n_layer}  n_head={cfg.n_head}  n_embd={cfg.n_embd}")

    input_ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {PROMPT!r}  ({prompt_len} tokens)")
    print(f"Generating {N_GEN} tokens with each strategy ...\n")

    # Warmup so the first step doesn't pay one-time JIT/allocation cost
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
        _ = model(input_ids, use_cache=True)

    print("[A] decode_no_cache  ...")
    seq_a, ms_a = decode_no_cache(model, input_ids, N_GEN)
    mean_a = sum(ms_a) / len(ms_a)
    print(f"    total = {sum(ms_a)/1000:.2f} s   "
          f"throughput = {1000/mean_a:6.1f} tok/s   "
          f"(mean {mean_a:.1f} ms/tok)")

    print("[B] decode_with_cache ...")
    seq_b, ms_b, final_past = decode_with_cache(model, input_ids, N_GEN)
    mean_b = sum(ms_b) / len(ms_b)
    print(f"    total = {sum(ms_b)/1000:.2f} s   "
          f"throughput = {1000/mean_b:6.1f} tok/s   "
          f"(mean {mean_b:.1f} ms/tok)")
    print(f"    TTFT (first / prefill step) = {ms_b[0]:.1f} ms")

    # Sanity: greedy outputs should be identical
    same = torch.equal(seq_a, seq_b)
    print(f"\nGreedy outputs identical : {same}")
    if not same:
        # Show first divergence point for debugging
        diff = (seq_a != seq_b).nonzero()
        print(f"  first divergence at index {diff[0].tolist()}")

    # Cache size at the end. transformers 5.x returns a DynamicCache;
    # access per-layer K/V tensors through .layers[i].keys / .values.
    # Each tensor shape: (batch, n_head, seq_len, head_dim).
    layers = final_past.layers
    n_layer = len(layers)
    k0 = layers[0].keys
    bytes_per_elem = k0.element_size()
    total_elems = sum(L.keys.numel() + L.values.numel() for L in layers)
    print(f"\nFinal KV cache:")
    print(f"  per-layer K shape : {tuple(k0.shape)}  dtype={k0.dtype}")
    print(f"  total layers      : {n_layer}")
    print(f"  total elements    : {total_elems:,}")
    print(f"  total bytes       : {total_elems * bytes_per_elem:,} "
          f"({total_elems * bytes_per_elem / 1024:.1f} KiB)")

    # Dump data
    data = {
        "model": MODEL_NAME,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "prompt": PROMPT,
        "prompt_len": prompt_len,
        "n_gen": N_GEN,
        "ms_no_cache": ms_a,
        "ms_with_cache": ms_b,
        "outputs_match": same,
        "final_kv_bytes": total_elems * bytes_per_elem,
    }
    (OUT_DIR / "hf_benchmark.json").write_text(json.dumps(data, indent=2))
    print(f"\nWrote {OUT_DIR / 'hf_benchmark.json'}")

    # Plot per-token latency
    xs = list(range(N_GEN))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ms_a, label="no cache  (recompute every step)", linewidth=1.6)
    ax.plot(xs, ms_b, label="with cache  (feed only new token)", linewidth=1.6)
    ax.set_xlabel("generation step")
    ax.set_ylabel("forward-pass time (ms)")
    ax.set_title(f"GPT-2 per-token decoding latency  (CPU, prompt={prompt_len} tokens)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_per_token.png", dpi=140)
    print(f"Wrote {OUT_DIR / 'latency_per_token.png'}")

    # Also plot cumulative time — makes the gap viscerally clear
    cum_a = [sum(ms_a[: i + 1]) / 1000 for i in xs]
    cum_b = [sum(ms_b[: i + 1]) / 1000 for i in xs]
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(xs, cum_a, label="no cache", linewidth=1.6)
    ax2.plot(xs, cum_b, label="with cache", linewidth=1.6)
    ax2.set_xlabel("generation step")
    ax2.set_ylabel("cumulative time (s)")
    ax2.set_title("GPT-2 cumulative decoding time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "latency_cumulative.png", dpi=140)
    print(f"Wrote {OUT_DIR / 'latency_cumulative.png'}")


if __name__ == "__main__":
    main()
