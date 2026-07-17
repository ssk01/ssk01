# TFRecord 格式详解

## 概述

TFRecord 是 TensorFlow 的标准训练数据格式，本质上是**带 CRC 校验的 protobuf 记录序列**。每条记录是一个 `tf.train.Example` protobuf 消息，承载一条样本的所有特征和 label。

实际使用中，TFRecord 承担两个角色：
1. **存储格式**：比 CSV/JSON 更紧凑、解析更快
2. **流式读取**：TF 的 `tf.data.TFRecordDataset` 原生支持分片、预取、并行解析

## 文件结构

TFRecord 文件就是一条接一条的记录，没有文件头、没有 footer。

```
┌──────────────────────────────────────┐
│  Record 0                            │
│  ┌────────────────────────────────┐  │
│  │ length        (8 bytes, uint64) │  │  ← 数据部分的字节数
│  │ length_crc    (4 bytes, uint32) │  │  ← length 的 CRC32C 校验
│  │ data          (length bytes)    │  │  ← tf.train.Example 序列化
│  │ data_crc      (4 bytes, uint32) │  │  ← data 的 CRC32C 校验
│  └────────────────────────────────┘  │
├──────────────────────────────────────┤
│  Record 1                            │
│  ┌────────────────────────────────┐  │
│  │ length        (8 bytes)         │  │
│  │ length_crc    (4 bytes)         │  │
│  │ data          (length bytes)    │  │
│  │ data_crc      (4 bytes)         │  │
│  └────────────────────────────────┘  │
├──────────────────────────────────────┤
│  ...                                 │
└──────────────────────────────────────┘

每条记录 overhead = 8 + 4 + 4 = 16 bytes
```

一条实际记录的 hex dump (本 demo 的样本)：

```
0000: 53 00 00 00 00 00 00 00    ← length = 0x53 = 83 bytes
0008: 0d 7d 4e 91                ← length 的 CRC32C
000c: 0a 51 0a 1d 0a 09 69 74 ... ← Example protobuf 开始 (83 bytes)
                                ...
005f: 6b                        ← data 结束
0060: d6 a4 95                   ← data 的 CRC32C
```

**CRC 的细节：**

- 使用 CRC32C (Castagnoli)，比标准 CRC32 检测能力更强
- `length_crc` 校验的是 `length` 这个 8 字节——防止 length 损坏导致读飞
- `data_crc` 校验的是 `data` 这个 `length` 字节——确保数据完整性
- 两个 CRC 的原始值在写入时做了 `masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8` 掩码处理，防止全零数据产生全零 CRC

**本 demo 文件统计分析：**

```
文件大小: 99000 bytes (96.7 KB)
记录数:   1000 条
每记录长: 83 bytes (固定，因为每条样本结构相同)
每 overhead: 16 bytes (8+4+4)
有效载荷: 83000 bytes (83.8%)
overhead: 16000 bytes (16.2%)
```

## 内部数据格式：tf.train.Example

`data` 部分是一个 protobuf 序列化的 `Example` 消息。protobuf 定义：

```protobuf
message Example {
  Features features = 1;
}

message Features {
  map<string, Feature> feature = 1;   // key = 特征名
}

message Feature {
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
}

message BytesList  { repeated bytes value = 1; }
message FloatList  { repeated float value = 1 [packed=true]; }
message Int64List  { repeated int64 value = 1 [packed=true]; }
```

三种列表类型的实现细节：

```
BytesList: 每个元素是一个 length-delimited 字段 (1-5 bytes 长度 + data)
  例: ["new", "vip"] → 0a 03 6e 65 77 0a 03 76 69 70

FloatList: 固定 4 bytes × N (packed repeated, 不支持 float64)
  例: [0.1, 0.5, 0.9] → 0a 0c 9a 99 99 3d 00 00 00 3f 66 66 66 3f

Int64List: varint 编码 × N (packed repeated, 小整数省空间)
  例: [12345] → 0a 02 b9 60
```

**Example 序列化后的文本表示：**

```
features {
  feature {
    key: "user_feat"
    value: { float_list { value: 0.496714145, value: -0.138264298, ... } }
  }
  feature {
    key: "item_feat"
    value: { float_list { value: 1.5230298, value: -0.234153375, ... } }
  }
  feature {
    key: "label"
    value: { float_list { value: 1 } }
  }
}
```

## 写 TFRecord

```python
import tensorflow as tf

with tf.io.TFRecordWriter("samples.tfrecord") as writer:
    for sample in dataset:
        example = tf.train.Example(features=tf.train.Features(feature={
            "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[sample.uid])),
            "age":     tf.train.Feature(float_list=tf.train.FloatList(value=[sample.age])),
            "tags":    tf.train.Feature(bytes_list=tf.train.BytesList(value=sample.tags)),
            "label":   tf.train.Feature(float_list=tf.train.FloatList(value=[sample.label])),
        }))
        writer.write(example.SerializeToString())
```

**压缩选项：**

```python
# 不压缩 (默认)
writer = tf.io.TFRecordWriter("data.tfrecord")

# GZIP 压缩 (整文件压缩, 不支持随机读取)
writer = tf.io.TFRecordWriter("data.tfrecord", options="GZIP")

# ZLIB 压缩
writer = tf.io.TFRecordWriter("data.tfrecord", options="ZLIB")
```

注意：压缩是整文件级别的，无法单独读取某条记录——适合离线训练，不适合在线检索。

## 读 TFRecord

```python
# 1. 创建 dataset
dataset = tf.data.TFRecordDataset(["samples.tfrecord"])

# 2. 定义解析函数
feature_desc = {
    "user_id": tf.io.FixedLenFeature([], tf.int64),
    "age":     tf.io.FixedLenFeature([], tf.float32),
    "tags":    tf.io.VarLenFeature(tf.string),       # 变长列表
    "label":   tf.io.FixedLenFeature([], tf.float32),
}

def parse_fn(serialized):
    parsed = tf.io.parse_single_example(serialized, feature_desc)
    return parsed["user_id"], parsed["age"], parsed["tags"], parsed["label"]

# 3. 应用
dataset = dataset.map(parse_fn).batch(32).prefetch(1)
```

两种解析方式对比：

| | FixedLenFeature | VarLenFeature |
|---|---|---|
| 输出 shape | 固定 (如 `[]` 标量, `[3]` 定长数组) | `SparseTensor` |
| 缺失值处理 | 用 `default_value` 填充 | 返回空 SparseTensor |
| 适用场景 | 必定存在的定长特征 | 可选特征、变长列表 (如用户历史点击序列) |

## 完整数据集构建范式

```
原始数据 (CSV/Parquet/DB)
        │
        ▼
[特征工程] 清洗、分桶、hash、归一化
        │
        ▼
[序列化] tf.train.Example → TFRecord 文件
        │
        ▼
[读取] tf.data.TFRecordDataset → 解析 → shuffle → batch → prefetch
        │
        ▼
[训练] model.fit() 或自定义训练循环
```

**关键约束 (来自生产文档)：训练样本的特征处理逻辑必须和在线 serving 的特征处理逻辑完全一致**，否则产生 train/serving skew。

## 为什么用 TFRecord

| | TFRecord | CSV | Parquet |
|---|---|---|---|
| 流式读取 | 原生 `TFRecordDataset` | 需解析 | 需要额外库 |
| 压缩 | 整文件 gzip/zlib | 依赖文件系统 | 列级压缩 (snappy/zstd) |
| Schema | 无 (每条独立 protobuf) | 有 header 行 | 有内嵌 schema |
| 随机访问 | 无 (顺序流) | 差 | 有 (row group) |
| TF 集成 | 无缝 | 需适配 | 需适配 |
| 人类可读 | 否 (protobuf 二进制) | 是 | 否 |

**结论：** TFRecord 是 TensorFlow 生态的首选格式，核心优势是 `TFRecordDataset` 的原生支持——自动分片、乱序、预取、并行解析，无需额外依赖。它不是最紧凑或最通用的格式，但在 TF 训练流水线里是最简单高效的选择。

## 常见问题

**Q: 为什么同一样本的不同 feature 存放顺序不影响训练？**

A: Example 内的 `map<string, Feature>` 的 protobuf 序列化顺序可能和写入顺序不同 (protobuf map 无序)。但 TF 解析时按 feature 名字查找，不受顺序影响。也就是说：`parse_single_example` 靠的是 key 名字匹配，不靠位置。

**Q: 为什么不用 JSON？**

A: JSON 的人类可读性带来的代价是解析慢 3-5×、体积大 2×。TFRecord 用 protobuf 二进制 + packed repeated 编码，float32 直接 memcpy，int64 用 varint。在大规模训练中差距显著。

**Q: TFRecord 能存张量 (tensor) 吗？**

A: 能。BytesList 可以存任意序列化数据，常见做法是用 `tf.io.serialize_tensor(tensor).numpy()` 把张量序列化成 bytes 再存进 TFRecord，读取时用 `tf.io.parse_tensor` 还原。

```python
# 写入
tensor_bytes = tf.io.serialize_tensor(my_tensor).numpy()
feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_bytes]))

# 读取
tensor = tf.io.parse_tensor(parsed["my_tensor"], out_type=tf.float32)
```

**Q: 如何检查一个 TFRecord 文件内容？**

```bash
# 列出前几条记录
python -c "
import tensorflow as tf
ds = tf.data.TFRecordDataset('samples.tfrecord').take(3)
for i, record in enumerate(ds):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    print(f'Record {i}:\n{example}\n')
"
```

## 参考

- [TFRecordWriter 源码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/tfrecord_writer.h) — 底层 C++ 实现 (recordio 格式)
- [TensorFlow Data API 指南](https://www.tensorflow.org/guide/data)
- [demo_two_tower.py](demo_two_tower.py) — 本 demo 中 TFRecord 的实际使用

## 附录: protobuf 三个关键机制的 wire format 实现

以下结合 Example 的实际 hex 逐字节标注，展示 protobuf 的 **map**、**oneof**、**packed repeated** 三种机制在二进制层面如何实现。

构造一个包含两种 feature 类型的 Example：

```python
example = tf.train.Example(features=tf.train.Features(feature={
    'uid':   tf.train.Feature(int64_list=tf.train.Int64List(value=[123])),
    'score': tf.train.Feature(float_list=tf.train.FloatList(value=[0.5])),
}))
```

序列化后的 34 bytes:

```
0a 21                                    ← 顶层 field 1 (Features.feature)
0a 0c 0a 03 75 69 64 12 05 1a 03 0a 01 7b  ← 第一条 map entry (uid: int64)
0a 11 0a 05 73 63 6f 72 65 12 08 12 06 0a 04 00 00 00 3f  ← 第二条 (score: float)
```

逐字节标注：

```
[ 0] 0a                   field 1 type=2 → Features.feature (顶层 repeated 消息)
[ 1] 21                     length=33 → 后续 33 bytes 是两条 map entry
[ 2] 0a                     field 1 type=2 → 第一条 map entry (FeatureEntry 消息)
[ 3] 0c                       length=12
[ 4] 0a 03 75 69 64           field 1 type=2 len=3: FeatureEntry.key = "uid"
[ 9] 12 05                    field 2 type=2 len=5: FeatureEntry.value (Feature 消息)
[11] 1a 03                      field 3 type=2 len=3 → oneof kind = int64_list
[13] 0a 01                        field 1 type=2 len=1 → Int64List.value (packed)
[15] 7b                           varint = 123
[16] 0a                     field 1 type=2 → 第二条 map entry
[17] 11                       length=17
[18] 0a 05 73 63 6f 72 65     field 1 type=2 len=5: FeatureEntry.key = "score"
[24] 12 08                    field 2 type=2 len=8: FeatureEntry.value (Feature 消息)
[26] 12 06                      field 2 type=2 len=6 → oneof kind = float_list
[28] 0a 04                        field 1 type=2 len=4 → FloatList.value (packed)
[30] 00 00 00 3f                  0.5 (float32 little-endian)
```

### 1. `map<string, Feature>` 的实现

protobuf 的 `map<K,V>` 在 wire format 上不是一种新的编码，而是直接展开为 `repeated` 嵌套消息。

```
message Features {
  map<string, Feature> feature = 1;
}
```

在 wire format 上等价于:

```
message Features {
  repeated FeatureEntry feature = 1;
}
message FeatureEntry {
  string key   = 1;   // ← map key
  Feature value = 2;   // ← map value
}
```

所以每条 map entry 都是三段：

```
0a <总长>          ← Features.feature (repeated FeatureEntry)
  0a <key长> <key> ← FeatureEntry.key   (field 1)
  12 <val长> <Feature消息> ← FeatureEntry.value (field 2)
```

要点：
- 同一个 key 出现两次 → 最后一个生效
- map 的顺序不保证有序，`parse_single_example` 靠 key 名字匹配，不靠位置
- 这就是为什么 Example 里的 feature 写入顺序和实际序列化顺序可能不同——不影响解析

### 2. `oneof kind` 的实现

```protobuf
message Feature {
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
}
```

protobuf **没有任何特殊的 wire format 标记来表示 oneof**。实现方式非常简单：

- 编码时只写入被选中的那一个字段，其他字段不出现，
- 解码时根据出现的 field number 判断类型

从 hex 中可以看到：

```
uid 的 Feature:  1a 03 ...  ← field 3 (int64_list) 出现 → kind = int64_list
score 的 Feature: 12 06 ... ← field 2 (float_list) 出现 → kind = float_list
bytes 的 Feature:  0a ...   ← field 1 (bytes_list) 出现 → kind = bytes_list
```

没有额外的 discriminant 字节或 union tag。这也是为什么 oneof 里的字段 **必须有不同的 field number**——靠 field number 区分类型。

### 3. `packed repeated` 的实现

protobuf 对 `repeated` 标量字段有一个 `[packed=true]` 优化选项。

**非 packed (默认行为)** — 每个元素独立带 tag 编码，靠 varint 自定界区分元素边界：

```
repeated int32 vals = 1;
[1, 2, 3]

wire format:
  08 01    ← tag=0x08 (field 1, varint), 读 varint → bit7=0 停止, value=1
  08 02    ← tag=0x08 又来了, 还是 field 1 → repeated 的下一个元素, value=2
  08 03    ← tag=0x08 继续, value=3
  0a ...   ← tag=0x0a (field 1 但 wire_type 变成 2) → 不是 varint, field 1 的 varint repeated 结束
```

varint 每个字节的 bit7=1 表示「还有后续字节」，bit7=0 表示「这是最后一个字节」。所以解析器读 varint 时自然知道它在哪里结束——不需要 length 前缀。

**packed** — 整体写成一个 length-delimited 块：

```
repeated int32 vals = 1 [packed=true];
[1, 2, 3]

wire format:
  0a 03    ← field 1 type=2 (length-delimited), length=3
  01 02 03 ← 3 个连续 varint, 没有 field tag
```

对比两种的 hex：

```
非 packed: 08 01 08 02 08 03          ← 6 bytes, 每个元素带 08 tag
packed:    0a 03 01 02 03             ← 5 bytes, 一次 tag + 连续值
```

在本 demo 的 Example 中，`Int64List.value` 和 `FloatList.value` 都声明了 `[packed=true]`：

```
0a 01 7b               → packed, [123]         (int64 varint × 1 = 1 byte value)
0a 04 00 00 00 3f      → packed, [0.5]         (float32 × 1 = 4 bytes value)
```

**packed 的收益：**

| | 非 packed | packed |
|---|---|---|
| tag 开销 | 每个元素 1 byte | 整个数组 1 byte |
| 内存布局 | 不连续 | 连续 (可 memcpy) |
| 适用类型 | 所有 | 仅标量 (varint/32bit/64bit) |
| 对 float 列表 | 每个 float 带 tag → 125% 大小 | 4N + 2 bytes, 几乎无开销 |

这就是为什么 TFRecord 里用 `FloatList` 存连续特征 (3 个 float32 = 12 bytes + 2 bytes overhead)，比 JSON 的 `[0.1, 0.5, 0.9]` (15+ bytes 再加解析) 高效得多。

### 4. oneof 与 optional 的关系

**oneof 在 wire format 上就是 optional 的语法糖**。字节层面完全一样——都只写被选中的那个 field tag，没有额外的 discriminant 标记。

```protobuf
message A {
  oneof kind {
    int32  a = 1;
    string b = 2;
  }
}

message B {
  optional int32  a = 1;
  optional string b = 2;
}
```

两者序列化后字节完全相同。区别在生成的代码层面：

| | oneof | 多个 optional |
|---|---|---|
| wire format | 只出现选中的 field tag | 只出现赋值的 field tag |
| 互斥 | set a → 自动清 b | 无, 可同时设 |
| `which_oneof()` | 有 | 无 |
| 内存布局 | 共享存储空间 (union) | 各自独立 |

所以“oneof”的能力是**语义约束**（互斥、节省内存），而不是**格式创新**。

### 5. tag 编码详解

protobuf 的每个字段都以一个 varint 编码的 **tag** 开头：

```
tag = (field_number << 3) | wire_type
```

wire_type 只有 6 种 (3 bits, 值 0~5)：

| wire_type | 值 | 负载格式 | 适用的 proto 类型 |
|-----------|----|---------|------------------|
| varint | 0 | varint 编码的整数 | int32, int64, uint32, uint64, bool, enum, sint32, sint64 |
| 64-bit | 1 | 固定 8 bytes | fixed64, sfixed64, double |
| length-delimited | 2 | varint 长度 + 数据 | string, bytes, nested message, packed repeated |
| start_group | 3 | 嵌套内容 + end_group | **废弃 (proto2 group)** |
| end_group | 4 | 仅 tag, 无负载 | **废弃 (proto2 group)** |
| 32-bit | 5 | 固定 4 bytes | fixed32, sfixed32, float |

**各种 wire_type 实例 (从本 demo 的 Example 提取)：**

```
wire_type 0 (varint):
  step = 160
  → 08 a0 01       tag=0x08(field1,varint), value=0xa001→160

wire_type 2 (length-delimited):
  map key "uid"
  → 0a 03 75 69 64   tag=0x0a(field1,len-del), length=3, data="uid"

wire_type 5 (32-bit):
  float 0.5
  → pack 到 FloatList 内部时是 raw bytes 00 00 00 3f
  (在 packed repeated 的 len-del 块内, 不需要独立 tag)
  (如果是独立的 optional float, tag=1<<3|5=0x0d)
```

**wire_type 3/4 (start_group / end_group) — 已废弃，但了解一下：**

group 是 proto2 的语法，等价于嵌套 message，但编码方式不同：

```protobuf
message OldStyle {
  optional group MyGroup = 1 {   // proto2 only
    optional int32 a = 2;
    optional int32 b = 3;
  }
}
```

序列化时：
- wire_type 3 标记 group 开始 (tag = field_num<<3 | 3)
- 内部字段正常编码
- wire_type 4 标记 group 结束 (tag = field_num<<3 | 4, 无负载)

对比等价的嵌套 message (wire_type 2)：
```
message → tag(2) + length + [fields...]   ← 知道总长, 可跳过未知字段
group   → tag(3) + [fields...] + tag(4)   ← 必须逐个解析直到 end_group
```

group 扫描慢且难以跳过未知字段，proto3 已移除，现在全用嵌套 message (wire_type 2)。

### 6. repeated message 的边界判定

这个问题是理解 protobuf 解析的核心。考虑这个例子：

```protobuf
message Outer {
  repeated Inner items = 1;   // 多条 Inner
  int32          demo  = 2;   // 另一个字段, 紧跟在 repeated 后面
}
message Inner {
  string name  = 1;
  int32  value = 2;
}
```

写两条 Inner: `{"a", 1}`, `{"b", 2}`，再加 `demo=99`。序列化为：

```
0a 05              ← tag=0x0a (Outer field1, wire2), length=5
   0a 01 61        ←   tag=0x0a (Inner field1, wire2), len=1, "a"
   10 01           ←   tag=0x10 (Inner field2, varint), value=1
                   ← ↑ 用完 5 字节 → 自动出栈, 回到 Outer 层
0a 05              ← tag=0x0a (Outer field1, wire2), length=5  第二笔 Inner
   0a 01 62        ←   tag=0x0a (Inner field1, wire2), len=1, "b"
   10 02           ←   tag=0x10 (Inner field2, varint), value=2
10 63              ← tag=0x10 (Outer field2, varint), value=99
```

**关键问题**：解析器读到第一条 Inner 的 `10 01` 后，内部已经读完 `0a 01 61 10 01` 这 5 个字节。Inner 的定义只有 field 1 和 field 2——两个都读完了。下一个字节是 `0a`。

这个 `0a` 是第二条 Inner (Outer field1) 的 tag，**还是** Inner 内某个未知字段的 tag？

**答案取决于 length 前缀，不取决于 Inner 有没有更多 field：**

- `0a 05` 声明了这条 Inner 总共 5 字节内容
- 解析器读完 5 字节后，**不论读到哪、不论是否还有未出现的字段**，都会无条件退出 Inner，回到 Outer 层
- 所以下一个 `0a` 必然是 Outer 层的 tag

同理，第二条 Inner 也是 5 字节 (`0a 01 62 10 02`)，用完 5 字节后，读到的下一个 tag 是 `10` (Outer field2) —— 这就是 `demo=99`。

**解析器的消息栈 (嵌套层级) 变化过程：**

```
进入: 0a 05  → push Outer_field1 → 配额 5 字节
  进入: 0a 01 61 → push Inner_field1 → 用完 3 字节
  回到:      → pop Inner_field1
  进入: 10 01 → push Inner_field2 → 用完 2 字节
  回到:      → pop Inner_field2
  总共用了 5 字节, 配额用完
回到:        → pop Outer_field1

读取: 0a 05  → push Outer_field1 → 配额 5 字节   (第二条 Inner)
  ... (同上)

读到: 10 63  → Outer field2 → 不是 field1, 所以 field1 的 repeated 结束
              → demo=99
```

**一句话：wire_type 2 的 length 前缀是「硬性配额」。解析器用完配额就出栈，不关心内层消息还有没有更多字段。这保证了即使遇到不认识的新字段，解析器也能精确跳过而不影响后续字段。**

**三种 wire_type 的定界对比：**

```
wire 0 (varint):   每个 varint 的 bit7 标记结束 → 元素自定界
                   如何知道 repeated 结束? 下一个 tag 的 field number 不同

wire 1/5 (定长):   固定 4 或 8 字节 → 自定界
                   如何知道 repeated 结束? 下一个 tag 的 field number 不同

wire 2 (len-del):  0a 05 ...5字节数据...  ← length 前缀 → 配额到了就出栈
                   0a 05 ...5字节数据...  ← 下一个 tag 还是 field1 → 下一个元素
                   10 63                 ← 下一个 tag ≠ field1 → repeated 结束
```
