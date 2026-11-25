# CH-SAC v2 二维单翼型网格实验

CH-SAC v2 从给定的翼型几何构建 C-H 拓扑。策略读取 18 维几何特征和 20 维网格与搜索状态，在 5 步内更新 7 个连续网格控制量，调节壁面和尾迹的节点分配。SAC 训练目标使用网格几何质量，SU2 求解用于训练后的气动验证。

## 发布内容

| 内容 | 文件 |
| --- | --- |
| 环境与模型 | [ch_env.py](ch_env.py)、[ch_sac_v2_model.py](ch_sac_v2_model.py) |
| 数据准备、训练与基线 | [prepare_dataset.py](prepare_dataset.py)、[train_ch_sac_v2.py](train_ch_sac_v2.py)、[run_baselines.py](run_baselines.py) |
| 评测与绘图 | [eval_and_cfd.py](eval_and_cfd.py)、[cfd_support.py](cfd_support.py)、[plot_ch_sac_v2.py](plot_ch_sac_v2.py) |
| 权重与训练历史 | [seeds](seeds) 下 seed 42–46 各一组 `best.pt`、`last.pt` 和 `history.csv`。检查点包含 Actor、双 Q Critic、目标 Critic 和熵温度。监督基线权重为 [supervised_mlp.pt](supervised_mlp.pt)。 |
| 测试数据与清单 | [测试网格缓存](cache)、[NACA-24 翼型坐标](reference/naca24/samples.jsonl)、[数据清单](dataset_manifest.json) |
| 统计和图像 | [训练概要](training_summary.json)、[基线概要](baselines_summary.json)、[测试概要](eval_summary.json)、[逐样本结果](eval_per_case.json)、[训练曲线](fig1_training_curves.png)、[网格质量测试](fig2_benchmark_comparison.png)、[网格与 CFD](fig3_mesh_and_cfd.png) |
| CFD 验证 | [六个算例的网格、配置、残差和表面解](cfd_verification)、[CFD 汇总](cfd_verification_summary.json) |

UIUC 划分包含 192 个训练、32 个验证和 64 个留出测试翼型。另有 24 个解析 NACA 四位数翼型。每个留出翼型在两档单元预算下评测。NACA 两档分别为 1,536 和 5,120 个四边形。UIUC 粗档含 36 个 1,536 单元网格和 28 个 1,588 单元钝尾缘网格，细档为 5,120 单元。UIUC 坐标和划分沿用仓库的 [UIUC v1 数据](../../../data/processed/uiuc_v1) 和 [第三方归属说明](../../../THIRD_PARTY_DATA.md)。

## 复现

以下命令从仓库根目录运行。先在虚拟环境安装 Python 依赖。

```sh
python -m pip install -e ".[ch-sac]"
```

使用发布的五组验证集最佳权重和留出缓存执行确定性评测。绘图命令按发布归档重新生成三张图。

```sh
python experiments/results/ch_sac_v2_20261125/eval_and_cfd.py --skip-cfd --eval-out repro_eval
python experiments/results/ch_sac_v2_20261125/plot_ch_sac_v2.py
```

在单独检出的分支副本中执行以下命令可重建缓存并重新训练。数据准备会更新数据清单和缓存，重新训练会更新同名种子目录和训练统计。

```sh
python experiments/results/ch_sac_v2_20261125/prepare_dataset.py
python experiments/results/ch_sac_v2_20261125/train_ch_sac_v2.py --device cuda
```

默认每个种子运行 60 epoch，每个 epoch 80 个 episode，共 24,000 个在线步，另有 1,000 个预热步。批大小为 256，回放容量为 50,000，学习率为 0.0003，折扣因子为 0.95，软更新系数为 0.005。种子 45 是按最佳验证得分排序选出的中位种子，用于 CFD。可运行 `run_baselines.py` 重算默认网格、全局优化、局部搜索和监督 MLP 基线。

使用本地 SU2 8.6 的 `SU2_CFD` 可复跑六个算例。执行前将 `/path/to/SU2_CFD` 替换为可执行文件路径。复跑写入新目录，原始报告和网格保留在 `cfd_verification`。

```sh
python experiments/results/ch_sac_v2_20261125/eval_and_cfd.py --cfd-only --su2 /path/to/SU2_CFD --cfd-out repro_cfd
```

CFD 工况采用定常 RANS-SA，马赫数 0.15，迎角 10°，雷诺数 600 万。图 3 中带星号的 4415 和 6423 L2Roe 算例达到迭代上限时仍未满足联合收敛条件，所列升力和阻力系数取末步值。
