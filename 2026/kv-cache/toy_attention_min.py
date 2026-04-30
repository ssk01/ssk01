"""
KV cache, the smallest possible version.

Single-head causal self-attention, no nn.Module, no multi-head, no LM head.
Input is already a sequence of vectors. Two paths produce identical output:
  (A) recompute K, V over the entire sequence at every step
  (B) keep K, V as a growing tensor; each step append one row
"""

import torch

torch.manual_seed(0)
D = 8                          # embedding / model dim
W_q = torch.randn(D, D)
W_k = torch.randn(D, D)
W_v = torch.randn(D, D)


def attn(x, k_cache=None, v_cache=None):
    """
    x: (T_new, D) — new tokens only when cache is given, else the full sequence.
    Returns: y (T_new, D), k_full (T_total, D), v_full (T_total, D)
    """
    q = x @ W_q                                      # (T_new, D)
    k_new = x @ W_k
    v_new = x @ W_v

    k = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=0)
    v = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=0)

    T_new, T_tot = q.shape[0], k.shape[0]
    T_past = T_tot - T_new
    # causal mask: row i (over q) sees cols [0 .. T_past + i]
    i_idx = torch.arange(T_new).unsqueeze(1)
    j_idx = torch.arange(T_tot).unsqueeze(0)
    mask = torch.where(j_idx <= T_past + i_idx, 0.0, float("-inf"))

    scores = q @ k.T / D ** 0.5 + mask
    y = torch.softmax(scores, dim=-1) @ v
    return y, k, v


# --- Path A: recompute everything every step --------------------------------
seq = torch.randn(5, D)        # 5-token "prompt" of plain vectors
ys_a = []
for t in range(1, len(seq) + 1):
    y, _, _ = attn(seq[:t])    # attend over the whole prefix
    ys_a.append(y[-1])         # only the last position is "new"

# --- Path B: KV cache, feed one new token per step --------------------------
ys_b = []
k_cache = v_cache = None
for t in range(len(seq)):
    y, k_cache, v_cache = attn(seq[t : t + 1], k_cache, v_cache)
    ys_b.append(y[0])

# --- Compare ----------------------------------------------------------------
A = torch.stack(ys_a)
B = torch.stack(ys_b)
print("Path A (recompute) shape :", tuple(A.shape))
print("Path B (cached)    shape :", tuple(B.shape))
print("Max |A - B|              :", (A - B).abs().max().item())
print("allclose                 :", torch.allclose(A, B, atol=1e-6))
print("Final cache shape K, V   :", tuple(k_cache.shape), tuple(v_cache.shape))
