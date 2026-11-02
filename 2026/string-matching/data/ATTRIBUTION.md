# 数据来源

正文来自 [OpenIndex / open-index/hacker-news](https://huggingface.co/datasets/open-index/hacker-news) 汇集的 Hacker News 公开评论。数据集卡标注数据库许可为 **ODC-BY**；评论内容由原作者撰写，数据库许可不另行赋予单条评论的内容权利。

本样本从已有的非空评论正文导出中，依次取不超过 16 MiB 的完整行。正文原始 UTF-8 字节、HTML 标签和实体均保留；没有为性能实验注入或修改文字。样本是导出顺序的一段，并非随机抽样。该导出没有保留 item id，`trace.json` 使用本样本内的零基行号与字节范围定位原文。

`comments.bin.gz` 解压后为各行正文直接拼接；`lengths.bin.gz` 解压后为逐行 little-endian uint32 字节长度。`manifest.json` 记录行数、正文长度和 SHA-256，运行脚本会校验。

原始样本不随仓库提交。重跑已记录的基准需将本次样本的两个压缩文件放在此目录，校验值须与 `manifest.json` 一致。公开来源链接用于说明来源，不能仅凭链接恢复这份未保留 item id 的导出片段。无需样本即可用 `python3 run.py --trace-only` 演示保留的评论，或用 `--plots-only` 重画已有结果。
