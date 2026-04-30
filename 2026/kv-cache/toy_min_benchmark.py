"""
KV cache benchmark — minimal version. Same spirit as toy_attention_min.py:
  - single causal self-attention layer
  - single head, no split/merge
  - pure functions, no nn.Module, no FFN, no LayerNorm
  - input is a sequence of vectors (skip embedding / lm_head)

Scaled up just enough to make timing measurable: D=512, prompt=16, generate 256.

This is the leanest possible side-by-side of "recompute every step" vs
"keep a KV cache and feed only the new token". Same n²-vs-n curve as a real
LLM, on ~30 lines of model code.
"""

import json
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)

# --- Model: 4 weight matrices, that's it -----------------------------------
D = 512
PROMPT_LEN = 16
N_GEN = 256

W_q = torch.randn(D, D) / D ** 0.5
W_k = torch.randn(D, D) / D ** 0.5
W_v = torch.randn(D, D) / D ** 0.5
W_o = torch.randn(D, D) / D ** 0.5


def attn(x, k_cache=None, v_cache=None):
    """
    x: (T_new, D)  — full sequence if no cache, else only the new vector(s).
    Returns: y (T_new, D), k_full (T_total, D), v_full (T_total, D).
    """
    q = x @ W_q
    k_new = x @ W_k
    v_new = x @ W_v

    k = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=0)
    v = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=0)

    T_new, T_tot = q.shape[0], k.shape[0]
    T_past = T_tot - T_new
    i_idx = torch.arange(T_new).unsqueeze(1)
    j_idx = torch.arange(T_tot).unsqueeze(0)
    mask = torch.where(j_idx <= T_past + i_idx, 0.0, float("-inf"))

    scores = q @ k.T / D ** 0.5 + mask
    return torch.softmax(scores, dim=-1) @ v @ W_o, k, v


# --- Decoding paths --------------------------------------------------------
@torch.no_grad()
def decode_no_cache(prompt, n_gen):
    """Each step: feed the entire sequence, take the last output as next vector."""
    seq = prompt.clone()
    times_ms = []
    for _ in range(n_gen):
        t0 = time.perf_counter()
        y, _, _ = attn(seq)
        times_ms.append((time.perf_counter() - t0) * 1000)
        seq = torch.cat([seq, y[-1:]], dim=0)
    return seq, times_ms


@torch.no_grad()
def decode_with_cache(prompt, n_gen):
    """Prefill once, then feed only the latest output vector each step."""
    times_ms = []
    t0 = time.perf_counter()
    y, k_cache, v_cache = attn(prompt)            # prefill
    times_ms.append((time.perf_counter() - t0) * 1000)
    seq = torch.cat([prompt, y[-1:]], dim=0)
    next_vec = y[-1:]

    for _ in range(n_gen - 1):
        t0 = time.perf_counter()
        y, k_cache, v_cache = attn(next_vec, k_cache, v_cache)
        times_ms.append((time.perf_counter() - t0) * 1000)
        next_vec = y[-1:]
        seq = torch.cat([seq, next_vec], dim=0)
    return seq, times_ms, k_cache, v_cache


def main():
    torch.set_num_threads(4)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Minimal toy attention   D={D}  single head  no FFN  no nn.Module")
    print(f"Prompt = {PROMPT_LEN} random vectors, generating {N_GEN} steps each\n")

    prompt = torch.randn(PROMPT_LEN, D) * 0.5

    # warmup so first measurement isn't a cold-cache outlier
    _ = decode_no_cache(prompt, 2)
    _ = decode_with_cache(prompt, 2)

    print("[A] decode_no_cache  ...")
    seq_a, ms_a = decode_no_cache(prompt, N_GEN)
    mean_a = sum(ms_a) / len(ms_a)
    print(f"    total = {sum(ms_a)/1000:.2f} s   "
          f"throughput = {1000/mean_a:7.1f} tok/s   (mean {mean_a:.2f} ms/tok)")
    print(f"    first step = {ms_a[0]:.2f} ms   last step = {ms_a[-1]:.2f} ms")

    print("[B] decode_with_cache ...")
    seq_b, ms_b, k_cache, v_cache = decode_with_cache(prompt, N_GEN)
    mean_b = sum(ms_b) / len(ms_b)
    print(f"    total = {sum(ms_b)/1000:.2f} s   "
          f"throughput = {1000/mean_b:7.1f} tok/s   (mean {mean_b:.2f} ms/tok)")
    print(f"    TTFT (first / prefill) = {ms_b[0]:.2f} ms   "
          f"last step = {ms_b[-1]:.2f} ms")

    # numerical equivalence
    diff = (seq_a - seq_b).abs().max().item()
    print(f"\nMax |seq_a - seq_b| = {diff:.2e}   "
          f"(allclose: {torch.allclose(seq_a, seq_b, atol=1e-4)})")

    print(f"\nFinal cache: K {tuple(k_cache.shape)}  V {tuple(v_cache.shape)}")
    bytes_total = (k_cache.numel() + v_cache.numel()) * k_cache.element_size()
    print(f"  {bytes_total:,} bytes  ({bytes_total/1024:.1f} KiB)")

    speedup = sum(ms_a) / sum(ms_b)
    print(f"\nSpeedup (no_cache / with_cache) = {speedup:.2f}x")

    # dump
    (out_dir / "toy_min_benchmark.json").write_text(json.dumps({
        "D": D, "prompt_len": PROMPT_LEN, "n_gen": N_GEN,
        "ms_no_cache": ms_a, "ms_with_cache": ms_b,
        "max_abs_diff": diff,
        "throughput_no_cache_tps": 1000 / mean_a,
        "throughput_with_cache_tps": 1000 / mean_b,
        "speedup": speedup,
    }, indent=2))
    print(f"Wrote {out_dir / 'toy_min_benchmark.json'}")

    # plots
    xs = list(range(N_GEN))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ms_a, label="no cache  (recompute every step)", linewidth=1.6)
    ax.plot(xs, ms_b, label="with cache  (feed only new vector)", linewidth=1.6)
    ax.set_xlabel("generation step")
    ax.set_ylabel("forward-pass time (ms)")
    ax.set_title(f"Minimal toy attention per-step latency  "
                 f"(D={D}, single head, prompt={PROMPT_LEN})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "toy_min_latency_per_token.png", dpi=140)
    print(f"Wrote {out_dir / 'toy_min_latency_per_token.png'}")

    cum_a = [sum(ms_a[: i + 1]) / 1000 for i in xs]
    cum_b = [sum(ms_b[: i + 1]) / 1000 for i in xs]
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(xs, cum_a, label="no cache", linewidth=1.6)
    ax2.plot(xs, cum_b, label="with cache", linewidth=1.6)
    ax2.set_xlabel("generation step")
    ax2.set_ylabel("cumulative time (s)")
    ax2.set_title("Minimal toy attention cumulative time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "toy_min_latency_cumulative.png", dpi=140)
    print(f"Wrote {out_dir / 'toy_min_latency_cumulative.png'}")


if __name__ == "__main__":
    main()
