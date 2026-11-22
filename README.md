# AeroBlockMesh-SCU

面向 CFD 的智能多块结构化网格生成研究项目，计划结合强化学习自动分块与无监督学习网格生成。

本仓库按照 SmartCFD-Mesh 的模块结构组织，保留 GitHub 仓库名称 `AeroBlockMesh-SCU`，Python 包名为 `smartcfd`。

## 当前状态

当前版本在已有公开数据处理与验证流程基础上补齐项目目录骨架、配置草案、模块职责说明和开发工具配置。`src/smartcfd` 中的几何、分块、强化学习、网格生成、质量评价及 CFD 模块尚未实现；`tests/` 顶层六个文件为相应的测试预留模块。

`scripts/` 顶层七个预留入口支持 `--help`，执行具体任务时会明确报告“功能尚未实现”。已有可执行的数据脚本和回归测试分别位于 `scripts/data/`、`tests/data/`，公开数据及其来源与固定划分见 [公开数据说明](docs/public_data_v1.md)。

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
├── configs/                        # 配置草案，尚未接入算法
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

基础包没有第三方运行依赖；多块清洗工具通过可选的 `data` 依赖使用 NumPy/h5py。配置中的路径约定相对于仓库根目录。

## 运行入口

| 脚本 | 计划用途 | 默认配置草案 |
| --- | --- | --- |
| `scripts/prepare_data.py` | 准备数据 | `configs/geometry.yaml` |
| `scripts/train_rl.py` | 训练分块策略 | `configs/rl.yaml` |
| `scripts/pretrain_imitation.py` | 专家示范预训练 | `configs/rl.yaml` |
| `scripts/generate_mesh.py` | 生成结构化网格 | `configs/meshing.yaml` |
| `scripts/evaluate.py` | 评价分块与网格 | `configs/evaluation.yaml` |
| `scripts/run_cfd.py` | CFD 验证 | `configs/cfd.yaml` |
| `scripts/visualize.py` | 可视化 | `configs/meshing.yaml` |

上述配置参数仅作为入口预留，目前不会读取配置或执行算法。

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

GitHub Actions 检查 Python 3.10、3.12 和 3.14 下的安装、依赖一致性、源码与 wheel 构建、模块导入及入口帮助，并运行数据回归测试、冻结版本哈希核验和多块清洗产物重现检查。顶层算法测试需要随实现补充。

模块边界见 [架构说明](docs/architecture.md)，接口草案见 [接口说明](docs/interfaces.md)，数据约定见 [数据管理](data/README.md)，开发约定见 [参与开发](CONTRIBUTING.md)。

## 许可证

许可证尚未选择；`LICENSE` 当前是待确定的占位文件，不构成开源许可。
