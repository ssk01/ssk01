# DuckDB 压缩算法详解

每个算法覆盖: 适用类型、采样状态、采样逻辑、落盘大小估算公式。
源码: `src/storage/compression/`，分析接口: `src/include/duckdb/function/compression_function.hpp`

---

## 分析时机与粒度

Checkpoint 时，`ColumnDataCheckpointer` 扫描列的**所有已有 Segment**，对所有候选算法做采样分析，选出一个 score 最低的算法。该 Checkpoint 生成的新 Segment 全部使用这个算法。

```cpp
// compression_function.hpp:159-167
typedef unique_ptr<AnalyzeState> (*compression_init_analyze_t)();    // ① 创建采样状态
typedef bool (*compression_analyze_t)(state, vector);                // ② 喂数据 (返回 false 自我淘汰)
typedef idx_t (*compression_final_analyze_t)(state);                 // ③ 返回 score (预估压缩字节数)

// 选分最低的
```

---

## 分值语义

score 的语义是**压缩后预计占用的字节数**。值越小越优。部分算法直接给精确值，部分用启发式估算。

---

## 各算法详解

### RLE (Run-Length Encoding)

**源码:** `src/storage/compression/rle.cpp`
**适用:** 所有整数类型 + FLOAT/DOUBLE/LIST

**采样状态:**
```
seen_count       — 出现了多少个 run (值变化次数)
last_value       — 当前 run 的值
last_seen_count  — 当前 run 的重复次数 (上限 uint16_max)
```

**采样逻辑:**
```
逐行扫描:
  if (值 == last_value) → last_seen_count++
  else → seen_count++, 记录新值, 开始新 run
```

**Score 公式:**
```
score = (sizeof(rle_count_t) + sizeof(T)) * seen_count
      = (2字节 + 值大小) * run数
```

10000 行全是同一个值 → 1 个 run → score = 2+4 = 6 字节。
10000 行全部不同 → 10000 个 run → score = (2+4)*10000 = 60KB (接近原始 40KB，无收益)。

---

### Bitpacking (含 Frame-of-Reference 和 Delta 编码)

**源码:** `src/storage/compression/bitpacking.cpp`
**适用:** 所有整数类型

**采样状态 (每组 BITPACKING_METADATA_GROUP_SIZE 个值为一组):**
```
minimum, maximum, min_max_diff   — FOR 评估
delta_offset, min/max_delta_diff — Delta 评估
can_do_for, can_do_delta         — 标志
total_size                        — 累计字节估算
compression_buffer_idx            — 分组计数
```

**采样逻辑 (每组内):**
```
① 计算 min/max → 如果 min == max → CONSTANT 模式 (只存一个值)
② 计算相邻差值 → 如果差值范围小 → DELTA_FOR 模式 (存一个基准值+差值)
③ 否则 → FOR 模式 (存 min + bitpacked offsets)
```

**Score 公式 (每组选择模式后累加):**

| 模式 | Score |
|------|-------|
| CONSTANT | `sizeof(T) + sizeof(metadata)` (4+4=8 字节) |
| CONSTANT_DELTA | `sizeof(T)*2 + sizeof(metadata)` (8+4=12 字节) |
| DELTA_FOR | `sizeof(T) + Align(width) + sizeof(T) + GetRequiredSize(count, width)` |
| FOR | `GetRequiredSize(count, width) + sizeof(T) + Align(width)` |

其中 `GetRequiredSize(count, width)` = 最少位数 × count / 8 取整。

跨组累加得到最终 score。

---

### ALP (Adaptive Lossless Floating-Point)

**源码:** `src/storage/compression/alp/`
**适用:** FLOAT / DOUBLE

**采样状态:**
```
total_bytes_used         — 累计预估字节
vectors_sampled_count    — 采样了多少个 Vector
total_values_count       — 总行数
rowgroup_sample          — 采样的浮点值 (用于找最优 exponent/factor)
compression_data         — ALP 压缩状态
```

**采样逻辑:**
```
① 采样部分 Vector → rowgroup_sample
② 从样本中找出 Top-K 的 (exponent, factor) 组合
③ 对每个采样 Vector 用组合压缩 → 实测压缩后大小
④ 取 min(压缩大小, 不压缩大小) 作为该 Vector 的得分
```

**Score 公式:**
```
factor_of_sampling = total_values_count / compressed_values  (采样比例外推)
score = TotalUsedBytes() * factor_of_sampling
```

实际压缩了两个数据流: encoded integers + exceptions。score 是精确的累计值（因为采样阶段真的跑了压缩）。

---

### ALPRD (ALP Real/Double — 字典式浮点压缩)

**源码:** `src/storage/compression/alprd/`
**适用:** FLOAT / DOUBLE (适合连续值、重复精度低的场景)

**采样状态:**
```
total_values_count       — 总行数
rowgroup_sample          — 采样值 (用 uint64/uint128 存 float/double 的位模式)
compression_data         — 字典状态
```

**采样逻辑:**
```
① 采样浮点值 → rowgroup_sample
② FindBestDictionary() → 找出最佳的字典大小/位宽
③ estimated_bits_per_value = 字典索引的位宽
④ estimated_compressed_bytes = bits × sampled_count / 8
```

**Score 公式:**
```
factor = 1 / (采样比例)

estimated_compressed_bytes = estimated_bits_per_value × sampled_count / 8
estimated_size = estimated_compressed_bytes × factor

n_vectors = ceil(total_count / ALP_VECTOR_SIZE)
per_vector_overhead = METADATA_POINTER_SIZE + EXCEPTIONS_COUNT_SIZE
per_segment_overhead = HEADER_SIZE + MAX_DICTIONARY_SIZE_BYTES

estimated_n_blocks = ceil(estimated_size / (block_size - per_segment_overhead))
final_size = estimated_size + n_vectors × per_vector_overhead
           + estimated_n_blocks × per_segment_overhead
```

---

### ZSTD (Zstandard)

**源码:** `src/storage/compression/zstd.cpp`
**适用:** VARCHAR (长字符串)

**采样状态:**
```
total_size    — 所有字符串字节总长
count         — 行数
vectors_per_segment, segment_count — segment 布局估算
```

**采样逻辑:**
```
① 统计 total_size + count
② average_length = total_size / count
③ 按 segment 布局估算 metadata 开销 (偏移量数组 + Vector 元数据)
```

**Score 公式:**
```
// 估算 ZSTD 能压到 50%
expected_compressed_size = total_size / 2.0

// 惩罚系数: 短字符串不值得用 ZSTD
penalty = (average_length >= ZstdMinStringLength) ? 1.0 : 1000.0

score = (expected_compressed_size
        + count × sizeof(string_length_t)     // 每行 4 字节长度
        + vector_metadata_size)               // Segment 元数据
        × penalty
```

短字符串平均长度不够 → 1000x 惩罚 → 自动输给 DictFSST。

---

### DictFSST (字典 + FSST — 当前默认字符串压缩)

**源码:** `src/storage/compression/dict_fsst/`，`analyze.cpp`
**适用:** VARCHAR (自 V1_3_0 起为默认)

**采样状态:**
```
max_string_length   — 最长字符串
total_count         — 总行数
total_string_length — 所有字符串字节之和
contains_nulls      — 是否含 NULL
```

**采样逻辑 (非常简单):**
```
逐行: 更新 max_length, total_count, total_string_length
```

**Score 公式:**
```
score = total_string_length / 2.0
```

纯粹启发式——假设字典去重 + FSST 符号表后大约能压到原大小的 50%。简单粗暴但有效。

---

### Roaring (Bitmap)

**源码:** `src/storage/compression/roaring/`
**适用:** BIT (validity masks) / BOOL

**采样状态 (每 2048 位为一个 container):**
```
Per-container:
  one_count, zero_count, run_count  — run 容器评估
  last_bit_set                       — 数组容器评估
Aggregated:
  data_size                          — 压缩数据累计
  metadata_size                      — 元数据累计
  total_size = data_size + metadata_size
```

**采样逻辑:**
```
每个 2048-bit container:
  if (连续0/1很长)     → run container (存 run_length = 4 字节)
  else if (1很稀疏)    → array container (存每个1的索引 = 2 字节/个)
  else                 → bitset container (不压缩, 2048/8 = 256 字节)
```

**Score 公式:**
```
score = total_size × ROARING_COMPRESS_PENALTY   // 2.0 惩罚系数
```

必须比不压缩小 2 倍才用 Roaring——因为 Roaring 扫描时需要解压，有 CPU 开销。

---

### Constant / Empty

**无采样阶段** — 由统计信息直接决定:

| 场景 | 算法 | 存储 |
|------|------|------|
| 列 stat 中 min == max | Constant | 只存一个值 |
| Validity 全有效/全 NULL | Empty | 存 0 字节 |

---

### Uncompressed (保底)

**源码:** `src/storage/compression/fixed_size_uncompressed.cpp` (定长)，`string_uncompressed.cpp` (STRING)，`validity_uncompressed.cpp` (BIT)

**Score 公式:**

| 类型 | Score |
|------|-------|
| 定长类型 | `sizeof(T) × count` |
| STRING | `count × 4 + total_string_size + overflow × marker_size` |
| BIT (validity) | `(count + 7) / 8` (1 bit/行) |

永远可用，作为**所有算法的对比基线**。其他算法必须在 score 上比 Uncompressed 更低（或通过惩罚系数被压制）。

---

## 每种物理类型的候选算法列表

```
BOOL            → Constant, Uncompressed, Bitpacking, RLE, Roaring
INT8..INT128    → Constant, Uncompressed, Bitpacking, RLE
FLOAT/DOUBLE    → Constant, Uncompressed, ALP, ALPRD, RLE
VARCHAR         → Constant, StringUncompressed, DictFSST, ZSTD
BIT (validity)  → Empty, ValidityUncompressed, Roaring
LIST            → Constant, Uncompressed, Bitpacking, RLE
```

CHIMP、Patas、FSST、旧 Dictionary 已废弃（返回 nullptr 直接淘汰）。

---

## 完整决策树 (以 Double 列为例)

```
同一列, 本次 Checkpoint 扫描:

候选: Constant / RLE / ALP / ALPRD / Uncompressed

① Constant.analyze = nullptr → 不参与采样, 靠 statistics 决定
② RLE.Analyze      → 逐行统计 run_count → score = (2+8) × runs
③ ALP.Analyze      → 采样浮点模式 → 实测压缩大小 → 外推
④ ALPRD.Analyze   → 采样找最优字典 → 估算位宽 → 外推
⑤ Uncompressed     → score = 8 × count

选出 min(score) → 用该算法压缩全列 → 写到 DataBlock
```
