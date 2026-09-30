# Conflict 清理与参数实验：运行和保存方案

版本：clean-study-v1。2026-09-30 讨论后实施。旧实验已查看过测试结果，本轮属于后续探索性研究。

## 实验范围

- 清理：从训练集中删除含 conflict 的 **5 条完整样本**（5 条 conflict 标注及另外 2 条 positive 标注）。原始数据保留；开发集/测试集逐字节不变。NULL 及原来的显式 aspect 筛选规则保留。
- A：in-domain 原始/清理后 × Standard/Weighted × 5 种子 = 20 次。2e-5、5 轮、warmup 0、无梯度裁剪。
- B：in-domain + 7 个 LODO × 两种 loss × 2e-5/3e-5/5e-5 × 种子 13/42 = 96 次。30 轮、10% warmup、梯度裁剪 1.0。仅源领域 train/dev。
- 每个条件独立用两种 loss、两个种子的最佳开发集完整三元组 micro-F1 均值选共同学习率；并列选更低学习率。不得把其他 fold 或 in-domain 的结果用于该 LODO fold 选参数。
- C：复用 B 选中的 32 个模型，再训练种子 123/2024/777 的 48 次，得到 80 个最终结果。最终测试只评估开发集选出的 checkpoint。
- 基础预算共 164 次完整训练，另有两次 64 条 train / 32 条 dev 的小型 GPU 验证。Aspect-only、Category-only、1e-4 和 40 轮扩展均未加入本轮。
- 不依赖早停，保存 30 轮完整曲线；30 轮线性调度中的第 8 轮不等同于旧 8 轮调度。若开发集最佳点仍在末轮，另行版本化扩展预算。
- 清理同时改变样本及情感输出词表，不能将结果变化全部归因于一个巨大类别权重。五个配对种子报告均值、标准差、逐种子差异；搜索种子 13/42 的复用会在报告中注明。

## 文件保存与下载

每次运行保存一个 `best_model.pt`；运行期间还滚动保存一个 `resume.pt`（包含当前权重、AdamW 状态、调度器、最佳状态及随机数状态），成功完成训练后自动移除恢复文件。中断后从上一完整 epoch 恢复，未完成 epoch 重做。每轮的指标写入一个小 `history.json`，不会每轮保存独立模型。

| 内容 | Cluster 保存 | 本地下载 |
|---|---|---|
| 结果、30 轮曲线、配置、各类别指标 | 保存全部 | 包含 |
| 开发集和测试集逐条预测 | gzip 压缩保存 | 包含 |
| 数据清理清单、数据哈希、原始小数据快照 | 保存一份 | 包含 |
| 实际运行代码快照、模型来源哈希、环境版本 | 保存 | 包含 |
| 预训练 BERT / tokenizer | 复用共享目录；tokenizer 小快照一份 | 仅 tokenizer/config 小快照 |
| 调参最佳模型 | 选参前保留 96 个；选参后保留选中的 32 个 | 不包含 |
| 最终模型 | 评估及结果导出成功后，默认保留每条件/方法按 dev F1 选出的 1 个，共 16 个 | 不包含 |
| 优化器和续跑文件 | 仅未完成训练需要 | 不包含 |

16 个模型预计约 7 GB，留在 cluster。结果 ZIP 不含任何模型权重；现有预测规模下通常比全量 artifacts 小很多，最终包大小会实际统计。预测文件可以重新计算指标、做错误分析和画图；对新文本推理仍需要保留的模型，或重新训练。

这是最终保留量。选参前仍需暂存候选权重；默认两个并发任务时，连同滚动续跑文件和原子写入临时文件，建议为新实验预留约 65–70 GB 的 cluster 空间。提高并发数会增加峰值占用。轻量下载不会降低这一阶段的临时存储需求。

只允许整理本轮带 study.json 标记的目录中的明确 checkpoint。旧目录、旧模型完全不参与清理。未完成任务保留恢复文件。整理前核对预测和指标哈希；已整理模型的任务保留完成标记，重跑不会误认为缺失模型而重新训练。

## 上传与提交

本地生成 `bertfinetune-clean-v1.tar.gz`，不包含旧结果或预训练大模型。解压到 `/data/$USER`，使用新的 `bertfinetune-clean-v1` 目录。共享环境和 BERT 默认复用 `/data/$USER/bertfinetune`；若现有目录不同，设置 `CS760_SHARED_PROJECT`，也可以分别设置 `CS760_PYTHON` 和 `BERT_MODEL_PATH`。

在 cluster 的新项目目录执行：

```bash
sha256sum -c cluster/transfer_manifest.sha256
bash cluster/submit_clean_study.sh
```

以上第二条只显示任务图。查看账号 GPU 配额、最长任务时限后提交：

```bash
CS760_CONCURRENCY=2 CS760_TRAIN_TIME=04:00:00 bash cluster/submit_clean_study.sh --submit
```

脚本按依赖提交：

1. `smoke`：两个小训练任务，验证失败则后续不启动。
2. `main`：116 个任务（A 的 20 次与 B 的 96 次交错排列），整个数组最多同时占用两个 GPU。
3. `select`：CPU 作业，在每个 fold 内选参数，核对选中 checkpoint 后删除 64 个未选中的新搜索模型。
4. `completion`：80 个任务（32 个只做评估，48 个新增训练）。
5. `finalize`：检查 100 个结果、汇总、导出 ZIP、整理到 16 个模型、刷新 ZIP。

GPU 任务默认 1 GPU / 4 CPU / 32 GB RAM；4 小时是可调整的申请时限，未承诺足够或符合当前账号策略。提交前运行小任务实测更合适。阵列默认使用集群分配的 GPU；A100 是否需要额外型号/分区参数须按账号实际配置确认。所有训练经 Slurm，登录节点只做准备/提交。

如果需要先单独验证：

```bash
sbatch --array=0-1%2 cluster/clean_study_gpu.sbatch smoke
```

检查小任务日志后再使用总提交脚本；总提交中的 smoke 会跳过已完成任务。脚本记录提交 job IDs，重复总提交会拒绝，避免重复花费 GPU。某索引失败时，查看报错后可直接重提对应阶段/索引，例如：

```bash
sbatch --array=17 cluster/clean_study_gpu.sbatch main
```

失败会阻断依赖作业；修复重提后，需要按 `submission_jobs.txt` 检查并取消/重建仍在等待旧失败依赖的后续作业，不能假定自动解除依赖。

## 最后只下载两个小文件

完成后位于新项目目录：

```text
artifacts/clean_study_results_light.zip
artifacts/clean_study_results_light.zip.json
```

第二个文件记录真实大小和 SHA256。ZIP 包含 CSV、汇总 JSON、所有训练曲线、压缩预测、配置/环境/数据审计和代码快照，可用本地工具完成报告。不要将整个 `artifacts` 或 `model_cache` 目录下载。

## 已有实验也可以只导出小结果

在含本脚本的新项目目录，指定旧结果路径：

```bash
python run_clean_study.py export --legacy-root /path/to/old/artifacts --destination /path/to/old-results-light.zip
```

这项操作只读取旧指标/预测，完全不读取或删除旧 checkpoint。请使用实际 Python 环境路径替换 `python`。

## 手动查看本轮清理计划

```bash
python run_clean_study.py prune --scope unselected
python run_clean_study.py prune --scope archive
```

默认只列出计划。`--apply` 才删除明确列出的新 checkpoint。若最终也不需要在 cluster 保留代表模型，可在结果包已下载并校验后使用 `--scope archive --keep-models none --apply`，之后重新导出以同步记录。
