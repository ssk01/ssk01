# StringView 实现与性能分析

## 是什么

`StringView` 是一个**非拥有（non-owning）** 的字符串视图，只存两个东西：

```
const char* data_;   // 8 字节
size_t      size_;   // 8 字节
```

它**不分配、不拷贝、不释放**底层字符数据。类比：`const char*` + 长度的安全包装。

## 核心 API

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 构造 | O(1) 或 O(n) | `const char*` 需 strlen，`string` 直接取 data/size |
| 拷贝构造/赋值 | O(1) | 浅拷贝，指向同一块数据 |
| 移动构造/赋值 | O(1) | 偷走 ptr+size，源置空 |
| 析构 | O(1) | 无动作 |
| `substr` | **O(1)** | 不拷贝，只偏移指针+缩小 size |
| `remove_prefix/suffix` | **O(1)** | 同上 |
| `find` | O(n) | 线性扫描 |
| `compare` | O(min(n,m)) | 字典序比较 |
| 值传递 | O(1) | 仅 16 字节，通常寄存器传参 |

## 特殊成员函数

```cpp
// 拷贝 — 浅拷贝，多个 view 共享同一数据
StringView(const StringView& other) noexcept : data_(other.data_), size_(other.size_) {}

// 拷贝赋值 — 含自赋值保护
StringView& operator=(const StringView& other) noexcept {
    if (this != &other) { data_ = other.data_; size_ = other.size_; }
    return *this;
}

// 移动 — 偷走，源重置为空
StringView(StringView&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr; other.size_ = 0;
}

// ~StringView = default — 无所有权，无需释放
```

## 与 `std::string` 的关键区别

| | `std::string` | `StringView` |
|---|---|---|
| 数据所有权 | **拥有** | **不拥有** |
| 拷贝 | 深拷贝 (new + memcpy) | 浅拷贝 (指针赋值) |
| substr | O(k) 分配+拷贝 | O(1) 偏移 |
| 值传递 | 拷贝整个字符串 | 拷贝 16 字节 |
| 修改内容 | 可以 | 不可以（只读） |
| 空终止保证 | `c_str()` 保证 `\0` | 不保证 |

## Benchmark

测试环境：macOS, Apple Silicon, `g++ -std=c++17 -O2`，50 万次重复

| 操作 | `std::string` | `StringView` | 加速比 |
|------|-------------|-------------|--------|
| split | 82 ms | 43 ms | **1.9x** |
| substr | 323 ms | 23 ms | **14.0x** |
| compare | 42 ms | 29 ms | 1.4x |
| pass-by-value | 16 ms | 0.26 ms | **63.5x** |

核心差异在**内存分配**：`string::substr` 每次都 `new` + `memcpy`，`StringView::substr` 只是 `{data_+offset, new_size}`。

## 使用指南

**适合：**
- 函数参数（替代 `const std::string&`）
- 字符串切割、解析 —— 零拷贝
- 读取配置、协议解析

**不适合（危险）：**
- 持有字符串内容（底层数据可能被释放 —— 悬空指针）
- 修改字符内容（只读）
- 需要 `c_str()` 保证 `\0` 结尾的场景

**黄金法则：StringView 的寿命不能超过它指向的数据。**
