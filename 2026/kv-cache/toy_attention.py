"""
Toy single-layer multi-head self-attention to demonstrate KV cache mechanics.

Two decoding paths produce identical outputs:
  (A) recompute_all  — at step t, recompute K, V over the entire prefix [0..t]
  (B) use_kv_cache   — at step t, compute only K_t, V_t for the new token, append to cache

Goal: show that K/V at position i depend ONLY on token i (not on later tokens),
so they can be cached. Q at the current step always needs to be freshly computed
because we are answering the question "what should the *new* token attend to".
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# --- Model dims (toy) ------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
VOCAB = 100
MAX_LEN = 16


# --- One causal multi-head self-attention layer ----------------------------
class ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D_MODEL)
        self.W_q = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_k = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_v = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_o = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.lm_head = torch.nn.Linear(D_MODEL, VOCAB, bias=False)

    def split_heads(self, x):
        # (B, T, D) -> (B, H, T, Hd)
        B, T, _ = x.shape
        return x.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)

    def merge_heads(self, x):
        # (B, H, T, Hd) -> (B, T, D)
        B, H, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def forward(self, tokens, past_kv=None):
        """
        tokens:  (B, T_new)  — NEW tokens only when past_kv is given, else the full prefix
        past_kv: (K_cache, V_cache) each of shape (B, H, T_past, Hd), or None

        Returns: logits (B, T_new, VOCAB), new_past_kv
        """
        x = self.embed(tokens)              # (B, T_new, D)
        q = self.split_heads(self.W_q(x))  # (B, H, T_new, Hd)
        k_new = self.split_heads(self.W_k(x))
        v_new = self.split_heads(self.W_v(x))

        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k_new], dim=2)  # grow along time
            v = torch.cat([v_past, v_new], dim=2)
        else:
            k, v = k_new, v_new

        # causal mask: q at position i (within T_new) can attend to all k positions
        # up to (T_past + i). Build a mask sized (T_new, T_total).
        T_new = q.shape[2]
        T_total = k.shape[2]
        T_past = T_total - T_new
        # row i of the mask allows columns [0 .. T_past + i]
        i_idx = torch.arange(T_new).unsqueeze(1)        # (T_new, 1)
        j_idx = torch.arange(T_total).unsqueeze(0)      # (1, T_total)
        allowed = j_idx <= (T_past + i_idx)             # (T_new, T_total)
        mask = torch.where(allowed, 0.0, float("-inf"))

        scores = (q @ k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)  # (B, H, T_new, T_total)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                          # (B, H, T_new, Hd)
        out = self.W_o(self.merge_heads(out))                  # (B, T_new, D)
        logits = self.lm_head(out)
        return logits, (k, v)


# --- Two decoding strategies -----------------------------------------------
def generate_recompute_all(model, prompt, n_steps):
    """No cache: at every step, feed the entire sequence through the layer."""
    seq = prompt.clone()
    last_logits_log = []
    for _ in range(n_steps):
        logits, _ = model(seq, past_kv=None)
        next_logits = logits[:, -1, :]
        last_logits_log.append(next_logits.clone())
        next_tok = next_logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
    return seq, last_logits_log


def generate_with_cache(model, prompt, n_steps):
    """With cache: prefill once on the prompt, then feed only the new token each step."""
    # Prefill: process the whole prompt, keep its K/V
    logits, past = model(prompt, past_kv=None)
    seq = prompt.clone()
    last_logits_log = [logits[:, -1, :].clone()]
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    seq = torch.cat([seq, next_tok], dim=1)

    for _ in range(n_steps - 1):
        # Feed ONLY the new token; cache holds K/V for all prior positions
        logits, past = model(next_tok, past_kv=past)
        next_logits = logits[:, -1, :]
        last_logits_log.append(next_logits.clone())
        next_tok = next_logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
    return seq, last_logits_log, past


# --- Demo ------------------------------------------------------------------
def main():
    model = ToyAttention().eval()
    prompt = torch.randint(0, VOCAB, (1, 4))   # B=1, prompt of 4 tokens
    n_steps = 8

    with torch.no_grad():
        seq_a, logits_a = generate_recompute_all(model, prompt, n_steps)
        seq_b, logits_b, final_cache = generate_with_cache(model, prompt, n_steps)

    print("=" * 70)
    print("TOY KV-CACHE DEMO  (single-layer multi-head self-attention)")
    print("=" * 70)
    print(f"Prompt tokens         : {prompt.tolist()[0]}")
    print(f"Generated (recompute) : {seq_a.tolist()[0]}")
    print(f"Generated (with cache): {seq_b.tolist()[0]}")
    print()

    # Numerical equivalence: argmax sequences must match, and per-step logits
    # must be (numerically) the same.
    seqs_match = torch.equal(seq_a, seq_b)
    max_logit_diff = max(
        (a - b).abs().max().item() for a, b in zip(logits_a, logits_b)
    )
    print(f"Output sequences identical : {seqs_match}")
    print(f"Max per-step |logit diff|  : {max_logit_diff:.2e}  (numerical noise only)")
    print()

    # Show what the cache looks like at the end
    K, V = final_cache
    print(f"Final cache shapes : K {tuple(K.shape)}  V {tuple(V.shape)}")
    print(f"  layout = (batch, n_heads, seq_len, head_dim)")
    print(f"  one layer, one request: {K.numel() + V.numel()} fp32 elements "
          f"= {(K.numel() + V.numel()) * 4} bytes")
    print()

    # The key invariant: K_i depends only on token_i. Demonstrate by computing
    # K for the prompt alone vs for the prompt embedded in the longer sequence,
    # and showing the prompt's K rows are identical.
    with torch.no_grad():
        _, (K_prompt_only, _) = model(prompt, past_kv=None)
        _, (K_full, _) = model(seq_b, past_kv=None)
    K_prefix = K_full[:, :, : prompt.shape[1], :]
    invariant_ok = torch.allclose(K_prompt_only, K_prefix, atol=1e-6)
    print(f"Invariant check  K(prompt alone) == K(prompt within full seq) : {invariant_ok}")
    print("  -> this is WHY we can cache: K_i never changes once token i is fixed.")


if __name__ == "__main__":
    main()
