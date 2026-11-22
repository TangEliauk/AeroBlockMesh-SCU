# AeroBlockMesh-SCU

面向 CFD 的智能多块结构化网格生成研究项目，计划结合强化学习自动分块与无监督学习网格生成。

本仓库按照 SmartCFD-Mesh 的模块结构组织，保留 GitHub 仓库名称 `AeroBlockMesh-SCU`，Python 包名为 `smartcfd`。

## 当前状态

仓库包含已冻结的公开数据处理流程和一个可运行的 **二维 NACA 四位数翼型原型**。原型使用四块共形 O 型网格，支持固定参数生成和 SAC 连续参数优化。可训练策略、评价解析翼型留出集、导出 PLOT3D 网格及块间接口，并打开离线演示。

原型使用解析翼型和搜索生成的调参示范。公开 UIUC 几何、家族划分和 CFD 参照资料继续按各自数据版本管理。自由拓扑分块、基于 UIUC 任意轮廓的策略训练、学习型块内网格生成和 CFD 求解保留为研究路线。[公开数据说明](docs/public_data_v1.md) · [二维实验与结果](docs/experiments.md)

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
├── configs/                        # 通用流程配置草案
│   ├── geometry.yaml
│   ├── partition.yaml
│   ├── rl.yaml
│   ├── meshing.yaml
│   ├── evaluation.yaml
│   └── cfd.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── uiuc_v1/                # 已有清洗后几何与质量记录
│   ├── expert/
│   ├── splits/
│   │   └── uiuc_v1/                # 已有固定数据划分
│   ├── references/
│   │   └── public_v1/              # 已有实验参照与元数据
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
│   ├── sac_2d/demo/               # 二维原型的离线演示
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
python -m pip install -e ".[dev]"
python scripts/generate_mesh.py --help
```

二维原型使用 NumPy。配置中的路径约定相对于仓库根目录。

## 运行二维原型

```powershell
python scripts/generate_mesh.py --naca 2412 --method sac --fine
python scripts/evaluate.py --naca-2d
python scripts/train_rl.py --naca-2d --steps 24000 --seed 1729
python scripts/visualize.py --demo
```

生成和评估命令可以直接加载内置的演示策略，也可通过 `--checkpoint` 指定训练检查点。用 `--method fixed` 生成参考参数网格，用 `--output` 指定独立的本地产物目录。训练入口自动完成示范搜索、行为克隆预训练和 SAC 训练。也可在浏览器直接打开 [离线演示](experiments/sac_2d/demo/index.html)。

更一般的流程入口及 YAML 保留在仓库中。使用 `--help` 查看各脚本的现有参数。

## 开发检查

```powershell
python -m build
python -m compileall -q src scripts tests
python -m unittest discover -s tests/data -p "test_*.py" -v
```

GitHub Actions 检查 Python 3.10、3.12 和 3.14 下的安装、依赖一致性、源码与 wheel 构建、模块导入及入口帮助。开发时用 `python -m pytest` 运行数据和算法测试。

模块边界见 [架构说明](docs/architecture.md)，接口草案见 [接口说明](docs/interfaces.md)，数据约定见 [数据管理](data/README.md)，开发约定见 [参与开发](CONTRIBUTING.md)。

## 许可证

许可证尚未选择；`LICENSE` 当前是待确定的占位文件，不构成开源许可。
