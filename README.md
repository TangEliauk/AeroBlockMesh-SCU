# AeroBlockMesh-SCU

面向 CFD 的智能多块结构化网格生成研究项目。当前实现涵盖翼型自动分块、TFI 结构化制网与 SU2 验证。

本仓库按照 SmartCFD-Mesh 的模块结构组织，保留 GitHub 仓库名称 `AeroBlockMesh-SCU`，Python 包名为 `smartcfd`。

## 当前实验 FusionSAC

FusionSAC 用带合法动作掩码的离散 SAC 选择切割、共享边插点和 STOP，自动生成翼型分块拓扑。通过验证的终图用于生成 1,536 和 5,120 单元两档 TFI 网格。仓库内提供正式评估、封存权重和从训练到 SU2 的完整复现流程。

- [FusionSAC 模型训练说明与 18 组科研插图](docs/fusionsac/README.md)
- [从 tgx 复现训练、评估和 CFD](docs/airfoil_fusion.md)
- [正式结果、封存权重和单例复算](experiments/results/airfoil_fusion_20261128/README.md)
- [本地方案与 GitHub CH-SAC v2 的实现对照](docs/airfoil_fusion_audit.md)
- [CPU 几何检查与验证耗时](docs/airfoil_fusion_performance.md)

CH-SAC v2 提供固定 C–H 拓扑的连续调参对照。对应的模型、权重、评估结果和复现命令随原发布保留。[CH-SAC v2 实验说明](experiments/results/ch_sac_v2_20261125/README.md)

## 公开数据与开发入口

`scripts/` 提供数据准备、训练、制网、评估、CFD 和绘图入口。公开数据脚本和回归测试分别位于 `scripts/data/`、`tests/data/`。数据来源与固定划分见 [公开数据说明](docs/public_data_v1.md)。

已补入四组 [多块分块终态参考](data/references/multiblock_v1/README.md)：十二块通道、四块多段翼型、DLR-F6 ICEM 工程、M6 四块网格及配对文件。原件与来源哈希保留，实际清洗的块、接口、边界记录另存于 [block_topology_v1](data/processed/block_topology_v1/README.md)。这些工程参考尚未转化为训练动作轨迹。

## 目录结构

```text
AeroBlockMesh-SCU/
├── README.md
├── LICENSE                         # 许可证待选择
├── CONTRIBUTING.md
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── THIRD_PARTY_DATA.md              # 已有第三方数据归属
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/                        # 通用流程与实验配置
│   ├── geometry.yaml
│   ├── partition.yaml
│   ├── rl.yaml
│   ├── meshing.yaml
│   ├── evaluation.yaml
│   └── cfd.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── uiuc_v1/                # 已有清洗后几何与质量记录
│   │   └── block_topology_v1/      # 多块终态清洗产物
│   ├── expert/
│   ├── splits/
│   │   └── uiuc_v1/                # 已有固定数据划分
│   ├── references/
│   │   ├── public_v1/              # 已有实验参照与元数据
│   │   └── multiblock_v1/          # 新增四组终态原件及清单
│   ├── sources/
│   │   └── public_v1/              # 已有下载来源清单
│   ├── public_v1_manifest.json
│   ├── README.md
│   └── README_cleaned.md
├── src/
│   └── smartcfd/
│       ├── __init__.py
│       ├── core/
│       │   ├── types.py
│       │   └── io.py
│       ├── geometry/
│       │   ├── airfoil.py
│       │   ├── geometry2d.py
│       │   ├── geometry3d.py
│       │   ├── domain.py
│       │   └── preprocessing.py
│       ├── partition/
│       │   ├── blocks.py
│       │   ├── topology.py
│       │   ├── actions.py
│       │   ├── constraints.py
│       │   └── transition.py
│       ├── rl/
│       │   ├── env.py
│       │   ├── state.py
│       │   ├── reward.py
│       │   ├── agents/
│       │   │   ├── policy.py
│       │   │   └── sac.py
│       │   ├── imitation/
│       │   │   ├── dataset.py
│       │   │   └── pretrain.py
│       │   ├── replay_buffer.py
│       │   └── trainer.py
│       ├── meshing/
│       │   ├── structured.py
│       │   ├── tfi.py
│       │   ├── mapping.py
│       │   ├── models/
│       │   │   ├── kanmeshnet.py
│       │   │   ├── meshnet3d.py
│       │   │   └── multimeshnet.py
│       │   ├── interfaces/
│       │   │   ├── matching.py
│       │   │   └── connectivity.py
│       │   └── generator.py
│       ├── evaluation/
│       │   ├── partition_quality.py
│       │   ├── mesh_quality.py
│       │   ├── jacobian.py
│       │   ├── orthogonality.py
│       │   ├── boundary_layer.py
│       │   └── metrics.py
│       ├── cfd/
│       │   ├── solver.py
│       │   ├── boundary_conditions.py
│       │   ├── convergence.py
│       │   └── validation.py
│       ├── baselines/
│       │   ├── rule_partition.py
│       │   ├── template_partition.py
│       │   └── traditional_meshing.py
│       └── pipeline/
│           └── workflow.py
├── scripts/
│   ├── data/                       # 已有下载、清洗、划分与验证脚本
│   ├── prepare_data.py
│   ├── train_rl.py
│   ├── pretrain_imitation.py
│   ├── generate_mesh.py
│   ├── evaluate.py
│   ├── run_cfd.py
│   └── visualize.py
├── experiments/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── tests/
│   ├── data/                       # 已有 21 个数据回归测试
│   ├── test_geometry.py
│   ├── test_partition.py
│   ├── test_rl_env.py
│   ├── test_meshing.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
└── docs/
    ├── architecture.md
    ├── interfaces.md
    ├── methodology.md
    ├── datasets.md
    ├── experiments.md
    ├── references.md
    └── public_data_v1.md            # 已有公开数据使用与复现说明
```

各 Python 子包还包含 `__init__.py`，空数据与实验目录使用 `.gitkeep` 保留。

## 开发环境

需要 Python 3.10 或更高版本。在仓库根目录运行以下命令（PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
python scripts/generate_mesh.py --help
```

基础包使用 NumPy、PyYAML 和 Gymnasium。`data` 提供多块清洗使用的 h5py。翼型融合使用 Python 3.11 和 `fusion` 依赖。配置中的路径相对于仓库根目录。

## 运行入口

| 脚本 | 用途 | 配置 |
| --- | --- | --- |
| `scripts/prepare_data.py` | 准备数据 | `configs/geometry.yaml` |
| `scripts/train_rl.py` | 训练分块策略 | `configs/rl.yaml` |
| `scripts/pretrain_imitation.py` | 专家示范预训练 | `configs/rl.yaml` |
| `scripts/generate_mesh.py` | 生成结构化网格 | `configs/meshing.yaml` |
| `scripts/evaluate.py` | 评价分块与网格 | `configs/evaluation.yaml` |
| `scripts/run_cfd.py` | CFD 验证 | `configs/cfd.yaml` |
| `scripts/visualize.py` | 可视化 | `configs/meshing.yaml` |

各脚本的 `--help` 显示相应参数。实验专用入口和冻结协议见 [翼型融合复现](docs/airfoil_fusion.md)。

## 开发检查

```powershell
python -m build
python -m compileall -q src scripts tests
python -m unittest discover -s tests/data -p "test_*.py" -v
python scripts/data/verify_public_data.py
python scripts/data/verify_multiblock_data.py
python scripts/data/clean_cgns_multiblock.py --check
python scripts/data/clean_engineering_multiblock.py --check
```

GitHub Actions 在 Python 3.10、3.12 和 3.14 上检查安装、构建、数据回归、冻结版本哈希及多块清洗产物。Python 3.11 作业检查模块导入、脚本入口、融合回归和正式权重的单例复算。

模块边界见 [架构说明](docs/architecture.md)，接口草案见 [接口说明](docs/interfaces.md)，数据约定见 [数据管理](data/README.md)，开发约定见 [参与开发](CONTRIBUTING.md)。

## 许可证

许可证尚未选择；`LICENSE` 当前是待确定的占位文件，不构成开源许可。

## 二维多块参考补充

新增 [三组二维辅助终态参考](data/references/multiblock_2d_v1/README.md)：TMR 四块后向台阶、CFL3D 三块喷射器和四块带腔 Hump。官方原件与配套输入保留，二维边界、投影与连接检查另存于 [block_topology_2d_v1](data/processed/block_topology_2d_v1/README.md)，不改写既有四组或冻结的 `public_v1`。

七组资料的维度、可参考信息、用途与定位见 [对照指南](docs/multiblock_reference_guide.md)和[机器可读用途表](data/references/multiblock_case_catalog.json)。新增三组用于基础约束、内流和复杂二维案例的辅助研究，不等同于翼型专家轨迹，也不表示分块、训练或 CFD 算法已接入。
