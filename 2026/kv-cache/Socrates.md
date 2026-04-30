### Q: KV cache 到底在 cache 什么？为什么 K、V 能 cache，Q 不能？
在 causal self-attention 里，每个位置 i 的 K_i = x_i · W_K 和 V_i = x_i · W_V 只依赖
token i 自己（一旦 token 落定，K/V 永远不变）。Q_i 同理也只依赖 x_i，但我们在第 t 步
**只需要 Q_t**（"新 token 想问什么"），历史的 Q 已经用过、再不需要——所以没必要存。
`toy_attention.py` 直接验证了这个不变量：把 prompt 单独跑出来的 K，和把 prompt 嵌在更长
序列里跑出来的 K（取前缀部分），逐元素 allclose。这就是缓存合法性的根基。

更深一层（"消费者"模型）：
- K_j、V_j 是**被未来反复读取的**——所有 j 之后的 output 都要用它们 → 公共资源，必须存
- Q_i 是**一次性的**——产生 output_i 之后再也不被任何人引用 → 用完即扔
所以 KV cache 这个名字里没有 Q 不是巧合：Q 没有"跨步浪费"问题，K、V 才有。
(2026-04-30 14:16, 更新于 15:08)

### Q: 用 cache 到底能快多少？为什么？
GPT-2 small（12 层）在 CPU 上 greedy decode 200 个 token：
- 无 cache：6.36 s（31.7 tok/s）
- 有 cache：1.27 s（159 tok/s）—— 5x 吞吐，输出完全一致
机制：无 cache 时第 t 步要把长度 (prompt+t) 的整段重过模型，K/V 投影 + attention 都是
O(seq_len)，累计 O(n²)。有 cache 时只对 1 个新 token 算 Q/K/V（O(1) 投影），attention
对历史 K/V 做一次 O(seq_len) 扫描，总累计 O(n)。看 `results/latency_per_token.png`：
蓝线（无 cache）从 12ms 线性涨到 50ms，橙线（有 cache）几乎平在 6ms。

LLM benchmark 的标准指标：
- **Throughput / TPS**（tokens/sec）—— 头条数字
- **TTFT**（time-to-first-token）—— prefill 耗时，决定首字延迟
- **TPOT** / **ITL**（time-per-output-token / inter-token latency）—— 解码每 token 耗时，决定流式体验
(2026-04-30 14:21, 更新于 14:46)

### Q: KV cache 占多少显存？为什么大模型推理是 KV-bound？
公式（一个请求，单位 byte）：`2 × n_layers × n_kv_heads × head_dim × dtype_bytes`
（× 2 是因为 K 和 V 两份；GQA 让 n_kv_heads << n_attn_heads，直接砍 KV 几倍）

几个真实数字（fp16）：
- Llama-2 7B（MHA 32 kv heads）：512 KiB / token → 32K context = 16 GiB
- Llama-3 8B（GQA 8 kv heads）：128 KiB / token → 32K context = 4 GiB
- Llama-3 70B（GQA 8 kv heads）：320 KiB / token → 32K context = 10 GiB，128K = 40 GiB

权重一份所有用户共享，KV 是**每用户 × 每 token**。Llama-3 70B 在一张 H100（80 GiB）上
服务 100 个 8K context 的并发用户，光 KV 就要 250 GiB——装不下。这就是 vLLM /
PagedAttention / prefix caching / KV 量化 / MLA 这些工程优化全都围绕 KV 转的原因。
(2026-04-30 14:22)

### Q: GQA 对 KV 显存的影响有多大？
直接按比例砍：n_kv_heads 减少几倍，KV 显存就减少几倍。Llama-3 8B 和 Llama-1 7B
n_layers / head_dim 完全相同，但 Llama-3 把 n_kv_heads 从 32 降到 8，KV/token 就
从 512 KiB 变成 128 KiB（1/4）。如果 Llama-3 70B 用 MHA，它在 32K 的 KV 会是 80 GiB
而不是 10 GiB——根本不可能上线。GQA 不是小优化，是让长 context 推理可行的关键。
(2026-04-30 14:22)

### Q: self.W_k 为什么不直接叫 self.k？
**会撞名**。`W_k` 是**投影矩阵**（学习参数，shape `(D, D)`），`k` 是它的**输出**（每步算
出来的 key 张量，shape `(B, H, T, Hd)`）。如果把参数叫 `self.k`，forward 里 `k_new = x @ self.k`
读起来像"x 乘以 key"，但实际意思是"x 乘以**生成 key 的那个矩阵**"。然后下一行 `k = cat([k_past, k_new])`
里的 `k` 又是另一个东西——两个 k 撞在一起。

`W_k` 来自 *Attention Is All You Need* 原论文里的 W^K, W^V, W^Q 记号。HuggingFace 几乎所有模型
用的另一种写法是 **`self.k_proj`**——意思一样，更工程化。`W_k` 偏数学，`k_proj` 偏代码。
两个都行，**`self.k` 不行**。
(2026-04-30 14:50)

### Q: attn() 函数到底在做什么？
一句话：**给输入 x，对每个位置让它从（历史 + 自己）里按相关度加权拿信息**。Q/K/V 三个角色：
- q = "我想找什么"（查询关键词）
- k = "我能被什么找到"（文档标题）
- v = "找到我能给你什么"（文档内容）

代码分四块：
1. **三个投影** `q = x @ W_q; k_new = x @ W_k; v_new = x @ W_v`
2. **cache 拼接** `k = cat([k_cache, k_new])`，`v` 同理 → cache 在物理上"长大"的地方
3. **causal mask** 行 i 看列 [0..T_past+i]，未来位置 = -inf（softmax 后 = 0）
4. **核心**：`scores = q @ k.T / √D + mask`；`out = softmax(scores) @ v @ W_o`

返回 `(out, k, v)`：out 是新输出，k/v 是更新后的 cache，下一步传回来用。
(2026-04-30 15:02)

### Q: Path A 输入整段、Path B 只输入 1 个，为什么？两条路径输出怎么会一致？
**Path A 没记忆**：函数无状态，每步要让它从原始向量重算所有 K, V → 必须喂整段前缀。
返回 (t, D)，但**只取 y[-1]**——前 t-1 行是重复劳动，扔掉。

**Path B 有记忆**：cache 里已存历史 K, V。只喂 1 个新向量 → 算 Q_t, K_t, V_t → 拼到 cache 里
→ 用新 q 跟全部 k 算 attention → 出 1 个 output。

```
Path A 第 t 步：input (t, D)，output (t, D) 取最后一行
Path B 第 t 步：input (1, D)，output (1, D) 全要
```
两边数学上算的是同一个 output_i 公式，所以结果一致（差 1e-7 浮点噪声）。
(2026-04-30 15:25)

### Q: Path A 不是也能优化吗？q 算了 t 行只用最后一行。
**对，能**。可以先做"Path A.5"（Q 单行）：q 只算最后一行，K, V 还得全算（attention 要扫全部 K）：

```python
q     = x[-1:] @ W_q  # 只 1 行
k_new = x @ W_k       # 必须全 t 行
v_new = x @ W_v       # 同上
scores = q @ k.T      # (1, t)
out    = softmax @ v @ W_o  # (1, D)
```

省了 Q 投影、O 投影、attention 的 (t-1) 行，**但 K, V 投影依然 t 次**——这是"位置内浪费"的修复，
不是"跨步浪费"的修复。

per-step FLOPs 对比（D=512, t=256）：
| | FLOPs | vs Path B |
|---|---:|---:|
| Path A 原版 | 671 M | 510× |
| Path A.5（Q 单行）| 134 M | 102× |
| Path B（KV cache）| 1.3 M | 1× |

A.5 比 A 省一个数量级，但比 B 还差 100×——那 100× 全是 K, V 投影的"跨步重复"，**只能靠存历史解决**。
"KV cache"这个名字里只有 KV 不是巧合：Q 是位置内可省（切片），KV 是跨步可省（必须存）。
(2026-04-30 15:28)

### Q: prefill 投影 vs decode 的 softmax(q·k)·v，算力配比多少？
**per decode step (with cache)**：
- 4 个投影（W_q, W_k, W_v, W_o，每个 (1,D)·(D,D)）：**4 D² FLOPs**（常数，与 T 无关）
- attention 本体（q·k^T 加 attn·v）：**2 D · T FLOPs**（线性 T）

**比例 = 4 D² : 2 D·T = 2D : T**，交叉点 **T = 2D**。

D=512 toy 的实测对照：
| T | 投影 FLOPs | attention FLOPs | 投影/attention |
|---:|---:|---:|---:|
| 16 | 1.05 M | 16 K | 64×（投影主导）|
| 256 | 1.05 M | 262 K | 4× |
| **1024** | **1.05 M** | **1.05 M** | **1× 交叉** |
| 4 K | 1.05 M | 4.2 M | 0.25×（attention 主导）|
| 32 K | 1.05 M | 33 M | 0.03× |

**真实 LLM 还要加 FFN（~8 D²）**，所以"常数项"≈ 12 D²，交叉点 = **6 D**：
- GPT-2 small (D=768)：~4.6 K
- Llama-3 8B (D=4096)：~25 K
- Llama-3 70B (D=8192)：~50 K

意思：**短 context 下"FFN + 投影"主导，长 context 下 attention 本体主导**。KV cache 把"常数项"
压到 O(1)/token，但救不了 attention 本体的 O(T)——长 context 还要靠 FlashAttention /
PagedAttention / 稀疏 attention。
(2026-04-30 15:14, 更新于 15:36)

### Q: KV cache 是谁发明的？
**没有明确发明者**。它是 transformer decoder 一旦用于自回归生成就显然会出现的优化——folk-level
实现技巧，不是某篇论文的"贡献"。

时间线：
- **2017** *Attention Is All You Need* 提到 decoder 推理是 incremental 的；同期 Google tensor2tensor、
  Fairseq、OpenNMT 实现里就有 incremental decoding 代码
- **2018-2019** GPT-1 / GPT-2 / Transformer-XL，KV cache 成 inference 标配，仍无人写论文
- **2019 Noam Shazeer** *Fast Transformer Decoding: One Write-Head is All You Need* —— **第一个把
  KV cache 当一等公民来分析**的论文，明确指出 decoder inference 瓶颈是 KV 内存带宽，提出
  Multi-Query Attention（MQA）。这是后来 GQA、MLA 整条进化链的起点。
- **2022-2023+** LLM serving 商业化后 KV cache 成全行业核心问题：MQA→GQA, vLLM/PagedAttention,
  FlashAttention, MLA(DeepSeek-V2)

Noam Shazeer 是 transformer 史上的关键人物：Vaswani 2017 八作者之一、推动 MQA→GQA 主线、
写过 GLU Variants（gated FFN 的 SwiGLU 来源）。
(2026-04-30 15:20)

### Q: 喂下游 FFN 也只喂 y 吗？前面的不用喂了？
**对，FFN 只喂那 1 个新 y**。理由：FFN 是**逐位置独立**的，没有跨位置交互。

整个 transformer 里**只有 attention 是跨位置的**：
| 子层 | 跨位置？ | decode 输入 | 需 cache？ |
|---|---|---|---|
| Attention | ✅ | 1 个 q | ✅ K, V 历史 |
| FFN (gated MLP) | ❌ | 1 个向量 | ❌ |
| RMSNorm / LayerNorm | ❌ | 1 个向量 | ❌ |
| residual / embedding / LM head | ❌ | 1 个向量 | ❌ |

整条 32 层 stack decode 一步全程只有 1 个向量在流动。每层有自己的 KV cache（W_k 各异），
所以 32 层 = 32 份独立 KV cache，这就是公式里 `n_layers` 的由来。

decode 是 **memory-bound** 不是 compute-bound——每出 1 个 token 都要把整个模型权重从 HBM 读
一遍。这也是 batch 同时跑很多用户能极大提升单卡总 tok/s 的原因（权重读一次 N 个用户共享）。
(2026-04-30 15:32)

### Q: FFN 的算力有多少？
**~8 D² per token**，比 attention 投影（4 D²）还大。

两种 FFN：
- **GPT-2 风格**：`W_down @ gelu(W_up @ y)`，d_ff = 4D，FLOPs = 4 D² + 4 D² = **8 D²**
- **Gated MLP / SwiGLU**（Llama / Mistral / Qwen / Gemma 全用）：
  `W_down @ ( silu(W_gate @ y) * (W_up @ y) )`，d_ff ≈ 8/3 D，FLOPs = 3 · D · d_ff ≈ **8 D²**
  Llama-3 8B 实际 d_ff = 14336 ≈ 3.5 D，FFN ≈ 10.5 D²

加进 per-step 账本（每层）：
- 常数项：4 D²（attn 投影） + 8-10 D²（FFN）≈ **12 D²**
- T-线性项：**2 D · T**（attention 本体）
- 新交叉点 = **6 D**

D=512 toy 加上 gated FFN，T=256 时 FFN 占 60%+ 总 FLOPs。GPU decode 时 FFN 那几个 GEMM 是
时间大头。
(2026-04-30 15:36)

### Q: MHA 是什么，有什么用？
**Multi-Head Attention**：把 D 拆成 H 份（每份 head_dim = D/H），并行做 H 个小 attention 再拼回。

为什么：单 head 整层只产一个 attention pattern，表达力受限。MHA 让每个 head 学一种不同的关注模式
（一个跟踪语法，一个跟踪指代，一个跟踪位置等等），表达力 ×H。

**总 FLOPs 完全没变**——分头是 reshape 不是切计算。但 cache 大小要 ×H（每 head 自己的 K, V）。

经验值：head_dim = **64 或 128** 是甜蜜点。

| 模型 | D | H | head_dim |
|---|---:|---:|---:|
| GPT-2 small | 768 | 12 | 64 |
| Llama-3 8B | 4096 | 32 | 128 |
| Llama-3 70B | 8192 | 64 | 128 |

**进化链**：
| 变体 | n_kv_heads | 思路 | 谁用 |
|---|---|---|---|
| MHA | n_attn_heads | 每 head 独立 K, V | GPT-2, Llama-1/2 7B |
| **MQA** | 1 | 所有 head 共享 K, V | PaLM, Falcon。质量损失大 |
| **GQA** | n_attn / G | 分组共享，比如 8:1 | **Llama-3 全系、Mistral、Qwen、Gemma**。质量损失小 |
| **MLA** | — | K, V 投到低维 latent，存 latent | DeepSeek-V2 / V3 |

整条链是为"省 KV cache"演化出来的——看一个新模型用什么 attention 变体，就是在看它怎么权衡
"质量 vs KV 显存"。
(2026-04-30 15:40)

<!-- 以下继续记录 -->
