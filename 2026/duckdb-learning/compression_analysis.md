# DuckDB 压缩算法采样选优机制

压缩不是用户指定的，而是 Checkpoint 时**对所有候选算法采样评估、自动选最优**。
源码: `src/storage/table/column_data_checkpointer.cpp`

---

## 三阶段流水线

```
Checkpoint 触发
  ↓
① init_analyze: 每个候选算法创建 AnalyzeState
  ↓
② analyze: 逐 Vector 喂数据, 所有算法同时采样
  ↓
③ final_analyze: 每个算法返回 score (预估压缩后字节数), 最低者胜出
```

---

## 阶段 ①: init_analyze — 初始化采样状态

```cpp
// 文件: column_data_checkpointer.cpp:154-170
void ColumnDataCheckpointer::InitAnalyze() {
    for each column:
        for each candidate_function:
            state = func->init_analyze(col_data, physical_type);
```

每个算法创建自己的 `AnalyzeState`，不同算法关心不同的事情。

以 **RLE** 为例 (`src/storage/compression/rle.cpp:93-96`):

```cpp
unique_ptr<AnalyzeState> RLEInitAnalyze(ColumnData &col_data, PhysicalType type) {
    return make_uniq<RLEAnalyzeState<T>>(block_manager);
}

// RLEAnalyzeState 核心字段:
struct RLEState<T> {
    idx_t seen_count;          // 目前为止出现了多少个 run (值变化的次数)
    T last_value;              // 当前 run 的值
    rle_count_t last_seen_count; // 当前 run 已经重复了多少次
};
```

以 **Dictionary** 为例 (`src/storage/compression/dictionary/dictionary_compression.cpp:70-78`):

```cpp
struct DictionaryAnalyzeState {
    idx_t current_tuple_count;     // 当前 segment 的行数
    idx_t current_unique_count;    // 当前 segment 的唯一值数
    idx_t current_dict_size;       // 字典的原始字节数
    StringHeap heap;               // 拥有字符串内存
    string_set_t current_set;      // 哈希集合, 跟踪唯一字符串
    bitpacking_width_t current_width; // 编码索引需要的位数
};
```

---

## 阶段 ②: analyze — 逐 Vector 喂数据, 所有算法并行采样

```cpp
// 文件: column_data_checkpointer.cpp:195-212
ScanSegments([&](Vector &scan_vector) {
    for each column:
        for each candidate_function:
            if (!func->analyze(*state, scan_vector))  // 返回 false = 这个算法自我淘汰
                state = nullptr;                       // 标记为无效, 不再考虑
});
```

**RLE 的 analyze** (`rle.cpp:98-110`):

```cpp
bool RLEAnalyze(AnalyzeState &state, const Vector &input) {
    for each row in input:
        if (value == last_value)
            last_seen_count++;           // 还在同一个 run → 计数器++
        else
            seen_count++;                // 值变了 → 记录旧 run, 开始新 run
            last_value = new_value;
}
```

一次遍历, 统计出"这一列有多少个 run"。值越有序越重复 → run 越少 → 压缩效果越好。

**Dictionary 的 analyze** (`dictionary_compression.cpp:80-83`):

```cpp
bool StringAnalyze(AnalyzeState &state_p, const Vector &input) {
    for each string in input:
        if (string 在哈希集合中已存在)
            current_tuple_count++;        // 见过, 只递增行计数
        else
            current_unique_count++;       // 新字符串, 入哈希集合
            current_dict_size += string.size();
            current_width = MinBitWidthToEncode(current_unique_count);

        if (估算当前 segment 会超出 Block 大小)
            Flush();                       // 当前 segment 满了, 开新 segment
    return true;
}
```

`CalculateSpaceRequirements` 估算公式:

```cpp
// dictionary/common.cpp:8-22
idx_t RequiredSpace(count, unique_count, dict_size, bit_width) {
    return DICTIONARY_HEADER_SIZE    // 固定头
         + dict_size                 // 字典字符串本身
         + unique_count * 4          // 索引(每个唯一值一个 uint32 偏移)
         + BitpackGetSize(count, bit_width);  // 位压缩后的索引数组
}
```

如果任意一个字符串大到 Block 放不下 → 直接返回 `false` → 淘汰 Dictionary 算法。

---

## 阶段 ③: final_analyze — 返回分数, 最低者胜

```cpp
// 文件: column_data_checkpointer.cpp:217-253
for each candidate_function:
    // score = 预估压缩后的字节数
    score = func->final_analyze(*state);

    if (score < best_score || 是被强制指定的压缩算法)
        best_score = score;
        winner = this_function;

// 选出 score 最低的算法
```

**RLE 的 final_analyze** (`rle.cpp:112-116`):

```cpp
idx_t RLEFinalAnalyze(AnalyzeState &state) {
    // 每个 run 需要 2 字节(run_count) + sizeof(T)(值)
    return (sizeof(rle_count_t) + sizeof(T)) * rle_state.seen_count;
}
```

50000 行只有 100 个 run → 得分: `(2 + 4) * 100 = 600 字节`。这是**精确值**, 不是估算。

**Dictionary 的 final_analyze** (`dictionary_compression.cpp:85-98`):

```cpp
idx_t StringFinalAnalyze(AnalyzeState &state_p) {
    auto total_space = segment_count * BLOCK_SIZE + 最后一段的大小;
    return 1.2 * total_space;  // 乘以 1.2 倍惩罚系数
}
```

`1.2` 的惩罚系数 (`MINIMUM_COMPRESSION_RATIO`) 是一个启发式设计——Dictionary 必须**至少节省 20% 空间**才值得用（因为查询时字典解压有开销）。如果字典压缩比不够 1:1.2, 得分偏高, 可能输给 FSST 或 RLE。

对于无压缩方法 (`FixedSizeUncompressed`)，得分就是 `count * sizeof(T)` —— 原始大小。

---

## 完整决策树

以 `INTEGER` 列为例，候选算法和打分逻辑:

```
原始大小: 50000000 × 4 = 200MB

COMPRESSION_CONSTANT:    score = 4  (所有值相同, 存一个值就行)
COMPRESSION_RLE:         score = (2+4) × run_count  (有序 → run 少 → 分数低)
COMPRESSION_DICTIONARY:  score = 1.2 × estimated_size
COMPRESSION_BITPACKING:  score = ceil(bits_needed/8) × count  (小整数 → 分数低)
COMPRESSION_FOR_DELTA:   score = 类似 bitpacking, 但先算差值
COMPRESSION_UNCOMPRESSED: score = 200MB (保底)

选出 score 最低的
```

最终效果就是我们前面 benchmark 看到的——同一列的 Segment 0 用 RLE（前一半有序），Segment 1 用 ZSTD（后一半乱序），算法自适应。

---

## 用户可干预的方式

如果不想让系统自动选:

```sql
SET force_compression = 'zstd';   -- 全局强制某个算法
SET force_compression = 'auto';   -- 恢复自动选择
```

代码在 `column_data_checkpointer.cpp:178-190`:

```cpp
if (compression_type != COMPRESSION_AUTO) {
    forced_method = ForceCompression(compression_type);  // 强制指定
} else if (force_compression_setting != COMPRESSION_AUTO) {
    forced_method = ForceCompression(force_compression_setting);  // 全局设置
}
```

强制模式跳过 `init_analyze` → `analyze` → `final_analyze` 流程, 直接用指定算法。
