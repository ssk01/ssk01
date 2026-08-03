# v4-grad — 独立梯度子图 (dl-from-zero 系列第 4 版)

dl-from-zero 系列第 4 迭代: **独立的梯度子图** —— 对应 TF `gradients.py`。

## 核心变化

之前(1~3 版)反向是**手写 per-op backward + Session 单独一趟逆拓扑遍历**。第 4 版改成:
**梯度计算本身变成图里的节点**(`graph/gradients.h` 的 `GradientBuilder`), 和正向统一对待。

```
前向图:  x y a b -> mul -> add -> sub -> square -> mean(loss)
                        |________梯度子图________|
   grad_seed(1.0) -> mean_grad -> ... -> reduce_sum -> mul -> reduce_sum
                                                      \         \
   sgd_step(a, grad_a)  sgd_step(b, grad_b)
```

因为梯度子图在**同一张图**里:
- 训练 fetch `sgd_step`(它消费 `grad_of[var]`)→ 梯度子图自然可达、随正向一起执行
- 推理 `prune` 到 `y_pred` → 梯度子图/loss/sgd 全部不可达, 自动被剪掉
- 不需要之前那套独立 backward 机制了

## 各 op 的梯度如何生成

| 前向 op | 梯度贡献(节点) |
|---|---|
| ADD | 梯度原样透传给两个输入 |
| MUL | `grad * 另一输入` |
| SUB | `grad` / `grad * (-1)` |
| SQUARE | `grad * 2 * x` |
| MEAN | `mean_grad` 专用 kernel(标量→广播成输入形状) |
| SIGMOID | `grad * p * (1-p)` |
| LOG | `grad * recip(x)` |
| MATMUL | `matmul_grad_a` / `matmul_grad_b` 专用 kernel(对应 TF MatMulGrad) |

**标量变量归约**: 标量变量会被广播到整批, 其梯度是 `[N]` 向量, 需 `reduce_sum` 成标量
(对应 TF 梯度里 shape-match 的 reduce; 旧实现的 `tensor_add_to` 是隐式求和, 子图改为显式 `REDUCE_SUM` 节点)。

## 顺手改进

- **RunState 用稠密数组**: `outputs[ node->id ]` 纯下标访问, 不再用 `unordered_map` 哈希
  (对应 TF executor 编译期定槽 + `vector<Entry>` 运行时数组下标); 每个节点创建时拿单调递增 `id`
- `has_value` 标记(对应 TF `Entry.has_value`): apply 优化器前检查该轮是否真算出了梯度

## 运行

```bash
make && ./build/train          # 训练(梯度子图)+ prune + 并发推理
make pybind && python3 demo_linear.py
python3 credit_card_fraud.py   # 逻辑回归, 54 节点(含梯度子图) -> 6 推理节点
python3 demo_concurrent.py
```

## 结果

- 线性回归收敛 a=1.97 b=2.99, prune 22 节点 -> 5 节点
- 欺诈 AUC 0.936, prune 54 节点 -> 6 节点(`X w b matmul add sigmoid`)
- 梯度子图节点在 `node_names` 里可见: `grad_seed, mean_grad, c2, mul, mul, reduce_sum, ...`

## 与 TF 的对应

| 我们 | TF |
|---|---|
| `gradients.h` GradientBuilder | `gradients.py` BackpropBuilder |
| `mean_grad` / `matmul_grad_a/b` 专用 kernel | `MeanGrad` / `MatMulGrad` 注册的梯度 op |
| `reduce_sum`(标量归约) | 梯度 shape-match 的 reduce |
| 梯度节点与正向同图, prune 自动剪 | 同 |
