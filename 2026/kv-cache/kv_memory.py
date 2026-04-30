"""
KV cache memory accounting for real LLMs.

Per-token KV size formula (one request):
    bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
                      ^                ^
                      K and V          GQA:  n_kv_heads << n_attn_heads
                                       MHA:  n_kv_heads == n_attn_heads

For a sequence of length L, the cache for one request is L * bytes_per_token.
For a batch of B concurrent requests, multiply by B.

This is independent of model parameter count — KV cost scales with serving
load (concurrent users × context length), while weight cost is paid once.
That's the whole reason serving systems care so much about KV.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Model specs. n_kv_heads < n_attn_heads means Grouped-Query Attention (GQA),
# which directly shrinks the KV cache by the head ratio.
MODELS = [
    # name              layers  attn_h  kv_h  head_d  params(B)  notes
    ("GPT-2 small",        12,     12,    12,    64,     0.124,  "MHA"),
    ("GPT-2 XL",           48,     25,    25,    64,     1.5,    "MHA"),
    ("Llama-1 7B",         32,     32,    32,   128,     7.0,    "MHA"),
    ("Llama-2 7B",         32,     32,    32,   128,     7.0,    "MHA"),
    ("Llama-2 70B",        80,     64,     8,   128,    70.0,    "GQA 8:1"),
    ("Llama-3 8B",         32,     32,     8,   128,     8.0,    "GQA 4:1"),
    ("Llama-3 70B",        80,     64,     8,   128,    70.0,    "GQA 8:1"),
    ("Qwen2.5 7B",         28,     28,     4,   128,     7.6,    "GQA 7:1"),
    ("Mistral 7B",         32,     32,     8,   128,     7.3,    "GQA 4:1"),
]

DTYPE_BYTES = 2          # fp16 / bf16 — the standard for KV in production
WEIGHT_DTYPE_BYTES = 2   # weights also fp16/bf16


def per_token_kv_bytes(n_layers, n_kv_heads, head_dim, dtype_bytes=DTYPE_BYTES):
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KiB"
    if b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MiB"
    return f"{b / 1024 ** 3:.2f} GiB"


def main():
    SEQ_LENS = [2_048, 8_192, 32_768, 131_072]   # 2K, 8K, 32K, 128K

    print("=" * 100)
    print("PER-TOKEN KV CACHE SIZE  (fp16, one request)")
    print("=" * 100)
    print(f"{'model':16s}  {'layers':>6s}  {'attn_h':>6s}  {'kv_h':>5s}  "
          f"{'head_d':>6s}  {'KV/token':>10s}  {'note':10s}")
    print("-" * 100)
    rows = []
    for name, nl, nh, nkv, hd, params, note in MODELS:
        bpt = per_token_kv_bytes(nl, nkv, hd)
        rows.append((name, nl, nh, nkv, hd, params, note, bpt))
        print(f"{name:16s}  {nl:6d}  {nh:6d}  {nkv:5d}  {hd:6d}  "
              f"{fmt_bytes(bpt):>10s}  {note:10s}")

    print()
    print("=" * 100)
    print("KV CACHE FOR ONE REQUEST AT VARIOUS CONTEXT LENGTHS  (fp16)")
    print("=" * 100)
    header = f"{'model':16s}  " + "  ".join(f"{l:>10}" for l in
                                            [f"{s:>5}".replace("_", "")
                                             + (" tok" if False else "")
                                             for s in SEQ_LENS])
    header = f"{'model':16s}  " + "  ".join(
        f"{(str(L // 1024) + 'K tok'):>11s}" for L in SEQ_LENS
    ) + f"  {'weights(fp16)':>14s}"
    print(header)
    print("-" * len(header))
    table_data = []
    for name, nl, nh, nkv, hd, params, note, bpt in rows:
        cells = [bpt * L for L in SEQ_LENS]
        weight_bytes = int(params * 1e9 * WEIGHT_DTYPE_BYTES)
        line = f"{name:16s}  " + "  ".join(f"{fmt_bytes(c):>11s}" for c in cells)
        line += f"  {fmt_bytes(weight_bytes):>14s}"
        print(line)
        table_data.append({
            "model": name,
            "n_layers": nl,
            "n_attn_heads": nh,
            "n_kv_heads": nkv,
            "head_dim": hd,
            "params_B": params,
            "note": note,
            "kv_bytes_per_token": bpt,
            "kv_bytes_at_lens": {str(L): bpt * L for L in SEQ_LENS},
            "weight_bytes_fp16": weight_bytes,
        })

    print()
    print("Read this table sideways:")
    print(" - Llama-3 70B at 32K context = 10 GiB JUST for one user's KV;")
    print("   at 128K it's 40 GiB. That's on top of ~130 GiB of weights.")
    print("   With B concurrent users, multiply by B. This is why serving")
    print("   is KV-bound, not weight-bound.")
    print(" - Llama-3 8B's KV/token is 1/4 of Llama-1 7B (same n_layers / head_dim)")
    print("   purely because GQA dropped n_kv_heads from 32 to 8.")

    # JSON dump
    (OUT_DIR / "kv_memory.json").write_text(json.dumps({
        "dtype_bytes": DTYPE_BYTES,
        "seq_lens": SEQ_LENS,
        "rows": table_data,
    }, indent=2))
    print(f"\nWrote {OUT_DIR / 'kv_memory.json'}")

    # Plot: KV memory vs context length, for a few representative models,
    # with horizontal lines for model weight size.
    plot_models = ["GPT-2 small", "Llama-2 7B", "Llama-3 8B",
                   "Llama-3 70B"]
    xs = [2 ** k for k in range(8, 18)]   # 256 .. 131072

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, nl, nh, nkv, hd, params, note, bpt in rows:
        if name not in plot_models:
            continue
        ys = [bpt * L / 1024 ** 3 for L in xs]   # GiB
        ax.plot(xs, ys, marker="o", markersize=3, label=f"{name}  ({note})")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length (tokens, log scale)")
    ax.set_ylabel("KV cache (GiB, log scale)  — fp16, one request")
    ax.set_title("KV cache memory grows linearly with context length")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # annotate weight sizes as horizontal reference lines
    for name in plot_models:
        row = next(r for r in rows if r[0] == name)
        params = row[5]
        wgib = params * 1e9 * WEIGHT_DTYPE_BYTES / 1024 ** 3
        ax.axhline(wgib, linestyle=":", linewidth=0.7, alpha=0.4)
        ax.text(xs[-1], wgib, f" {name} weights", fontsize=7,
                va="center", alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "kv_memory_vs_context.png", dpi=140)
    print(f"Wrote {OUT_DIR / 'kv_memory_vs_context.png'}")


if __name__ == "__main__":
    main()
