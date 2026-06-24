### Q: 如何设计医院排队叫号系统，30min system design 格式？
核心设计：以 Redis Sorted Set 为队列引擎，score = priority * 10^15 + timestamp_ns 编码优先级与 FIFO 顺序，Lua 脚本保证原子出队，WebSocket 实时推送大屏，PostgreSQL 兜底持久化与审计日志。过号 3 分钟超时自动跳过，3-10 分钟可重排同优先级队尾，超 10 分钟降权排队尾。关键风险点：score 乘数必须远大于时间戳（毫秒 10^12 级，故用 10^15），否则优先级被时间戳淹没。(2026-06-24)
<!-- 以下继续记录 -->
