# Socrates — DuckDB Q&A 知识库

### Q: DuckDB 有类似 AggregatingMergeTree 的结构吗？
DuckDB 没有 AggregatingMergeTree 这类预聚合表引擎。DuckDB 的设计哲学是查询时实时向量化聚合——列存 + SIMD + 并行哈希聚合，5 千万行 4 维聚合只需 0.07 秒。查即算的成本低于预聚合维护成本。如果确实需要物化，手动 `CREATE TABLE agg AS SELECT ... GROUP BY ...` 即可。
(2026-07-13)

### Q: ClickHouse AggregatingMergeTree 预聚合后查询为什么还是慢？
客户端-服务端固定开销 ~0.5s（Rosetta 转译 + TCP 连接 + 协议握手），不是 ClickHouse 查询处理慢（服务端毫秒级）。排除网络开销后，纯计算上 DuckDB 快约 10x（嵌入式 + 向量化）。
(2026-07-13)

### Q: SegmentTree 是什么结构？
有序数组 + 惰性加载。`vector<unique_ptr<SegmentNode<T>>> nodes`，每个节点存 `row_start` 和 `shared_ptr<ColumnSegment>`。按行号二分查找定位 Segment。持久化数据用 `SUPPORTS_LAZY_LOADING=true` 的模板参数，Segment 在查询时按需从磁盘加载。
(2026-07-13)

### Q: SegmentTree 中的数据未写盘能查询吗？
能。`DataTable::Scan()` 先扫持久化 RowGroup，再扫 `LocalStorage`（事务本地 transient Segment）。同一事务 INSERT 后立即可 SELECT 到，其他事务看不到（MVCC 隔离）。
(2026-07-13)

### Q: 1GB 内存算 10GB 数据的 AVG 会爆内存吗？
全局聚合（无 GROUP BY）：不会。DuckDB 和 ClickHouse 都是流式块处理，逐块累加 SUM/COUNT，内存 O(1)。带 GROUP BY：取决于分组数，超内存时 DuckDB 用 RadixPartitionedHashTable 溢出磁盘。
(2026-07-13)

### Q: DataChunk 是行存还是列存？
列存。`vector<Vector> data`，每个 Vector 是一列的值。DuckDB 全链路列存：DataChunk → ColumnDataCollection → ColumnSegment → DataBlock。
(2026-07-13)

### Q: 压缩算法是怎么选出来的？
Checkpoint 时 `ColumnDataCheckpointer` 扫描所有 Segment，对所有候选算法并行采样：init_analyze → analyze(逐 Vector) → final_analyze(返回预估压缩字节数)。score 最低者胜出。每列独立选，每次 Checkpoint 重选。
(2026-07-13)

### Q: DictFSST 的 FSST 阶段在干什么？
字典去重后的字符串内容再做子串级压缩。扫描所有字典条目找高频子串（最高 255 个），分配单字节码，逐条替换。例如 "hello world" 变成 [0x01]" "[0x02]。符号表存子串→字节码的映射，查询时反向解压。
(2026-07-13)

### Q: reference_wrapper 为什么能放进 vector 而原生引用不行？
原生引用不是对象——不能重新绑定、不满足 CopyAssignable。`reference_wrapper` 是普通类，内部包一个 T* 指针，默认构造 + 拷贝构造 + 拷贝赋值全有。容器存的实际上是这个 wrapper 对象（8 字节指针）。
(2026-07-13)

### Q: DuckDB 的 BufferPool 有 LRU 吗？
有近似 LRU。三级优先级队列（BLOCK → MANAGED_BUFFER → TINY_BUFFER），每次 Block 被 unpin 时递增序列号 + 记录时间戳入队。驱逐时按优先级出队，死节点（weak_ptr 过期或序列号不匹配）跳过。另有 PurgeAgedBlocks 基于 LRUTimestamp 做年龄驱逐。
(2026-07-13)

### Q: ~SegmentLock() 如果手写，mutex.unlock 还会自动调吗？
会。C++ 规则：无论是否手写析构函数体，编译器都会在函数体执行完后自动调用所有非静态成员变量的析构。释放锁的是 `unique_lock<mutex>::~unique_lock()`。
(2026-07-13)

<!-- 以下继续记录 -->
