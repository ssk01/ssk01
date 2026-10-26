# COST 与 PageRank：Mac 本地原型

首先打开 `cost_pagerank.ipynb`。已保存执行输出，包含中文解释、数学模型、四节点 PageRank 演示、串行与多进程实测图。

## 文件

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

若现有 Python 已有 NumPy、SciPy、Matplotlib，可直接运行最后一行。执行器会运行 notebook 中的本地代码，只对你信任的 notebook 使用。

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
