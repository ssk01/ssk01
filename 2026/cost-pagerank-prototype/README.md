# COST、PageRank 与连通分量：Mac 本地原型

首先打开 `cost_pagerank.ipynb`。已保存执行输出，包含中文解释、数学模型、四节点 PageRank 演示、串行与多进程实测图。

## 文件

- `cost-paper-zh.md`：COST 论文完整中文译文，保留表格、代码、原图与参考文献；图像在 `assets/cost-paper/`。

- `cost_pagerank.ipynb`：主要阅读入口，参数可修改。
- `pagerank_demo.py`：PageRank 实现及 macOS spawn 安全的子进程函数。
- `run_notebook.py`：无需 Jupyter 的顺序执行器，捕获打印和图像，更新 notebook。
- `results.json`：最近一次执行的原始计时、配置和版本信息。

## 本地运行

在此目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_notebook.py
```

若现有 Python 已有 requirements.txt 中的依赖（包括 Numba），可直接运行最后一行。执行器会运行 notebook 中的本地代码，只对你信任的 notebook 使用。

交互编辑（可选）：

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --prefix .venv --name cost-pagerank --display-name 'COST PageRank'
jupyter lab cost_pagerank.ipynb
```

从项目目录启动，选刚安装的 Python 内核，顺序执行所有单元。修改 `N`、`DEGREE`、`ITERATIONS` 可改变真实负载；修改 `EXTRA_A` 等可改变数学模型。模型数字和实测数字分开呈现。普通 notebook 运行使用 plt.show；轻量执行器注入等价的图像捕获函数。

默认 8000 节点、127984 条边、20 轮迭代，只启动至多 4 个子进程。避免初次运行就把 N 改得特别大：朴素并行实现每轮发送边片段并返回稠密数组，其内存和通信成本随规模增长。

计时包含预热、三次重复中位数；图构建与进程启动单列。未绑定 CPU 核。环境变量限制常见原生线程池；threadpoolctl 可额外限制已加载线程池。CSR 稀疏矩阵向量计算作为单线程基线。Python 循环与 CSR 的差距混合了解释器、表示和数据访问等因素。

正确性：四节点结果与独立线性方程解比较；所有大图实现比较同样 20 轮的结果，并检查非负、概率和。这里不是原论文系统性能复现，也不测量 DuckDB。

来源：
- https://www.usenix.org/system/files/conference/hotos15/hotos15-paper-mcsherry.pdf
- https://ir.cwi.nl/pub/28800/28800.pdf

## 并行瓶颈诊断（Notebook 第 6.1 节）

`diagnose_parallel.py` 比较逐轮传图、常驻共享图、共享排名与输出、CSR 按行分区线程执行四种设计；覆盖 8000、80000、400000 节点，每节点约 16 条出边。各配置预热一次，计时五次，固定 20 轮，校验结果一致。大型朴素版本需反复传输 GB 级数据，整套实验可能需要数分钟。

```bash
python diagnose_parallel.py
python run_notebook.py
```

完整计时保存在 `diagnosis_results.json`。Notebook 只读取诊断结果，不会默认重复运行这套实验。横轴是工作线程/子进程数，不表示独占核心；不含准备和启动的查询耗时与端到端耗时不同。拆分计时中的提交/等待包含计算，不能直接解释为纯通信时间。CSR 使用线程避免跨进程序列化；原生稀疏计算允许线程并行执行，这不是把 Python 逐边循环放进线程池。

## COST 初衷、作者实现与连通分量

COST 衡量超过实现得当的单线程程序所需的配置。PageRank notebook 开头与第 6.4 节明确区分自身加速比和相对同一个单线程 CSR 基线的收益；已测配置只覆盖 1、2、4 个 worker。

- `author_sources.md`：作者源码固定提交、源码链接、实际编译运行范围与教学移植的区别；原始源码摘录与许可证在 `references/`。
- `paper_algorithms.py`：Numba 编译的 PageRank、Hilbert 编码、原地标签传播、按 rank 合并的并查集与同步并行标签传播。
- `benchmark_paper.py`：重新测量布局和连通分量实验，独立 SciPy 正确性校验，结果写入 `layout_results.json`、`components_results.json`。JIT 预热不计入计时，重复五次取中位数。
- `cost_pagerank.ipynb` 第 5.1 节：同一个 PageRank 内核只改变边遍历顺序，展示 Hilbert 曲线、加速比和预处理摊销；内存数组实验不包含原作者磁盘压缩布局或大页优化。
- `cost_connected_components.ipynb`：从标签传播过程图开始，比较不同边顺序、算法工作量与耗时，再将并行配置和有效单线程基线比较。保留标签传播在部分图上比并查集更快的结果。
- `author_smoke_results.json`：原始 Rust 程序的本地执行检查记录。

两个 notebook 均保存了执行输出。常规阅读只加载上述大型实验的结果，不会自动重测：

```bash
python run_notebook.py cost_pagerank.ipynb
python run_notebook.py cost_connected_components.ipynb
# 可选：重新测量，然后重新执行两个 notebook 更新图表
python benchmark_paper.py
```

不同实验的图和 PageRank 初始化语义有区别，耗时只在各自匹配的实验组内比较。原始概率归一化 demo 与新加入的作者语义移植在 notebook 中单独解释。
