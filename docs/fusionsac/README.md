# FusionSAC 模型训练说明与科研插图

FusionSAC 使用带合法动作掩码的离散 SAC 为二维尖尾缘翼型生成分块拓扑，再通过 TFI 生成两档结构化网格。模型训练说明梳理状态和动作建模、行为克隆、训练设置、正式评估与 SU2 求解。

- A4 六页训练说明　[Word](<FusionSAC 模型训练说明 V1.0.docx>) · [PDF](<FusionSAC 模型训练说明 V1.0.pdf>)
- [18 组科研插图总览](figures/科研插图总览.pdf)
- [图件引用与实验数据来源](图件引用与实验数据来源清单.md) · [文件与 SHA-256 清单](figure_manifest.json)

## 单组图件

| 编号 | 图题 | 文件 |
|---|---|---|
| 01 | FusionSAC 总体研究框架 | [PNG](figures/01_方法原理与模型架构/01_framework.png) · [PDF](figures/01_方法原理与模型架构/01_framework.pdf) · [SVG](figures/01_方法原理与模型架构/01_framework.svg) · [draw.io](figures/drawio/01_framework.drawio) |
| 02 | 翼型预处理与初始计算域 | [PNG](figures/01_方法原理与模型架构/02_geometry.png) · [PDF](figures/01_方法原理与模型架构/02_geometry.pdf) · [SVG](figures/01_方法原理与模型架构/02_geometry.svg) |
| 03 | 自动分块 MDP | [PNG](figures/01_方法原理与模型架构/03_mdp.png) · [PDF](figures/01_方法原理与模型架构/03_mdp.pdf) · [SVG](figures/01_方法原理与模型架构/03_mdp.svg) |
| 04 | BlockGraph 与拓扑操作 | [PNG](figures/01_方法原理与模型架构/04_blockgraph.png) · [PDF](figures/01_方法原理与模型架构/04_blockgraph.pdf) · [SVG](figures/01_方法原理与模型架构/04_blockgraph.svg) |
| 05 | FusionSAC 离散候选网络 | [PNG](figures/01_方法原理与模型架构/05_network.png) · [PDF](figures/01_方法原理与模型架构/05_network.pdf) · [SVG](figures/01_方法原理与模型架构/05_network.svg) · [draw.io](figures/drawio/05_network.drawio) |
| 06 | BC 预训练与离散 SAC 训练 | [PNG](figures/01_方法原理与模型架构/06_bc_sac.png) · [PDF](figures/01_方法原理与模型架构/06_bc_sac.pdf) · [SVG](figures/01_方法原理与模型架构/06_bc_sac.svg) · [draw.io](figures/drawio/06_bc_sac.drawio) |
| 07 | 多随机种子开发验证曲线 | [PNG](figures/02_模型训练与性能分析/07_learning_curves.png) · [PDF](figures/02_模型训练与性能分析/07_learning_curves.pdf) · [SVG](figures/02_模型训练与性能分析/07_learning_curves.svg) |
| 08 | 训练内部指标 | [PNG](figures/02_模型训练与性能分析/08_training_metrics.png) · [PDF](figures/02_模型训练与性能分析/08_training_metrics.pdf) · [SVG](figures/02_模型训练与性能分析/08_training_metrics.svg) |
| 09 | 独立测试算法对比 | [PNG](figures/02_模型训练与性能分析/09_algorithm_comparison.png) · [PDF](figures/02_模型训练与性能分析/09_algorithm_comparison.pdf) · [SVG](figures/02_模型训练与性能分析/09_algorithm_comparison.svg) |
| 10 | 消融与样本效率 | [PNG](figures/02_模型训练与性能分析/10_ablation_efficiency.png) · [PDF](figures/02_模型训练与性能分析/10_ablation_efficiency.pdf) · [SVG](figures/02_模型训练与性能分析/10_ablation_efficiency.svg) |
| 11 | 初始化后的策略退化与恢复 | [PNG](figures/02_模型训练与性能分析/11_policy_regression.png) · [PDF](figures/02_模型训练与性能分析/11_policy_regression.pdf) · [SVG](figures/02_模型训练与性能分析/11_policy_regression.svg) |
| 12 | 真实策略逐步自动分块 | [PNG](figures/03_自动分块与网格生成/12_trajectory.png) · [PDF](figures/03_自动分块与网格生成/12_trajectory.pdf) · [SVG](figures/03_自动分块与网格生成/12_trajectory.svg) |
| 13 | 典型翼型与终图对比 | [PNG](figures/03_自动分块与网格生成/13_topology_comparison.png) · [PDF](figures/03_自动分块与网格生成/13_topology_comparison.pdf) · [SVG](figures/03_自动分块与网格生成/13_topology_comparison.svg) |
| 14 | 两档 TFI 网格与局部细节 | [PNG](figures/03_自动分块与网格生成/14_tfi_meshes.png) · [PDF](figures/03_自动分块与网格生成/14_tfi_meshes.pdf) · [SVG](figures/03_自动分块与网格生成/14_tfi_meshes.svg) |
| 15 | 网格质量与配对评价 | [PNG](figures/03_自动分块与网格生成/15_mesh_quality.png) · [PDF](figures/03_自动分块与网格生成/15_mesh_quality.pdf) · [SVG](figures/03_自动分块与网格生成/15_mesh_quality.svg) |
| 16 | 网格质量奖励截断与饱和 | [PNG](figures/04_CFD验证与局限性分析/16_reward_saturation.png) · [PDF](figures/04_CFD验证与局限性分析/16_reward_saturation.pdf) · [SVG](figures/04_CFD验证与局限性分析/16_reward_saturation.svg) |
| 17 | 真实失败算例与发生频率 | [PNG](figures/04_CFD验证与局限性分析/17_failures.png) · [PDF](figures/04_CFD验证与局限性分析/17_failures.pdf) · [SVG](figures/04_CFD验证与局限性分析/17_failures.svg) |
| 18 | SU2 求解与联合收敛检查 | [PNG](figures/04_CFD验证与局限性分析/18_su2.png) · [PDF](figures/04_CFD验证与局限性分析/18_su2.pdf) · [SVG](figures/04_CFD验证与局限性分析/18_su2.svg) |

## 结果适用范围

正式独立评估覆盖同一冻结家族内的 90 个尖尾缘翼型。E7、E8 的最佳权重均选自在线第 0 步。六个 SU2 工况均未达到联合收敛标准。完整证据见[正式实验归档](../../experiments/results/airfoil_fusion_20261128/README.md)。
