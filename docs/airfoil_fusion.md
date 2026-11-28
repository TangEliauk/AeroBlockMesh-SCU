# 翼型自动分块融合实验

融合实验使用 C 型外流域与固定近壁保护带。策略在所有未保护块上选择候选切割或共享边插点。全图通过四边形终态验证后可选择 STOP。终图用于 TFI 制网和 SU2 导出。

[基线证据与融合审计](airfoil_fusion_audit.md) · [封存正式结果](../experiments/results/airfoil_fusion_20261128/README.md)

## 从 GitHub 复现

使用 Python 3.11、Git 和常规 64 位 CPU。由完整 Git 检出运行，Git 对象用于读取固定 CH-SAC v2 提交中的对照证据。A 方案参考、训练和验证所需的翼型坐标及划分随仓库提供。Excel 由 XlsxWriter 生成。

### 安装

PowerShell

```powershell
git clone --branch tgx git@github.com:TangEliauk/AeroBlockMesh-SCU.git
Set-Location AeroBlockMesh-SCU
py -3.11 -m venv .venv
$fusionPython = '.\.venv\Scripts\python.exe'
& $fusionPython -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
& $fusionPython -m pip install -e '.[dev,data,fusion]' numpy==2.3.1 PyYAML==6.0.2
& $fusionPython -m pip check
& $fusionPython scripts/verify_airfoil_fusion_release.py --replay
```

Linux shell

```sh
git clone --branch tgx git@github.com:TangEliauk/AeroBlockMesh-SCU.git
cd AeroBlockMesh-SCU
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,data,fusion]' numpy==2.3.1 PyYAML==6.0.2
python -m pip check
python scripts/verify_airfoil_fusion_release.py --replay
```

### 求解器和训练

安装 SU2 并将 `AEROBLOCK_SU2` 指向解包后的 `SU2_CFD`，也可把可执行文件所在目录加入 `PATH`。原正式运行使用官方 v8.6.0 发布页的 Windows OpenMP ZIP，资产文件名为 `SU2-v8.5.1-win64-omp.zip`，SHA-256 为 `4bc28b656def415b4d22c8274af2473dd719aa5ebfa6bc1c95c08b1bac26508c`。Linux 可从同一发布页安装匹配平台的版本。每次 CFD 输出会记录实际可执行文件哈希。[SU2 发布页](https://github.com/su2code/SU2/releases/tag/v8.6.0)

在仓库根目录运行以下命令。将第一行替换为实际 SU2 安装路径。

```powershell
$env:AEROBLOCK_SU2 = 'C:\path\to\SU2_CFD.exe'
& $fusionPython -B -m pytest -q tests/test_airfoil_fusion.py tests/test_airfoil_study_report.py
& $fusionPython -B -u scripts/run_airfoil_fusion_accelerated.py --profile smoke --output experiments/runs/airfoil_fusion_smoke_repro --validation-workers 1 --cache-geometry
& $fusionPython -B -u scripts/run_airfoil_fusion_accelerated.py --profile formal --smoke-evidence experiments/runs/airfoil_fusion_smoke_repro --output experiments/runs/airfoil_fusion_formal_v2 --validation-workers 1 --cache-geometry
```

```sh
export AEROBLOCK_SU2=/path/to/SU2_CFD
python -B -m pytest -q tests/test_airfoil_fusion.py tests/test_airfoil_study_report.py
python -B -u scripts/run_airfoil_fusion_accelerated.py --profile smoke --output experiments/runs/airfoil_fusion_smoke_repro --validation-workers 1 --cache-geometry
python -B -u scripts/run_airfoil_fusion_accelerated.py --profile formal --smoke-evidence experiments/runs/airfoil_fusion_smoke_repro --output experiments/runs/airfoil_fusion_formal_v2 --validation-workers 1 --cache-geometry
```

先检查冒烟结果目录的 `audit.json` 中 `passed` 为 `true`。正式实验的耗时随机器与策略轨迹变化，跨平台重训的浮点轨迹可能有差异。发布校验命令核对封存字节，并使用数值容差复算一条封存权重的测试算例。

运行目录由程序创建。续跑相同目录时会核对冻结源码、配置及数据，再从最新的 `resume.pt` 恢复模型、优化器、经验池与随机状态。创建输出目录中的 `STOP_REQUESTED` 可申请队列在保存检查点后停止，移除标记后按原命令续跑。训练队列日志位于 `queue_logs`，测试逐例凭证位于 `evaluation/case_receipts`。

单进程验证会复用已核验轨迹和 E8 的相同曲边验证请求。内存充足时可尝试 `--validation-workers 2`。优化机制与测量见 [训练耗时诊断](airfoil_fusion_performance.md)。

可用相同输入另行执行 `--stage report` 或 `--stage audit`。它们读取对应运行目录中已生成的评估和求解结果。

## 实验与结果

冒烟使用 32 个训练和 8 个开发验证翼型、seed 42，SAC 每臂 40 步。正式矩阵冻结 128 个训练和 32 个验证翼型，seed 42–46，在五个 SAC 臂上各执行至少 4,000 步。若 E6、E7 或 E8 有至少两个种子在第 4,000 步比第 3,000 步多完成至少一个验证翼型，则全部 SAC 臂统一延长到 8,000 步。最优权重与末轮权重均进入独立测试。

| 方法 | 对照含义 |
|---|---|
| E1–E3 | 固定规则、合法动作随机、即时贪心 |
| E4 | 原始 FIFO 离散 SAC |
| E4a | 全局动作、显式 STOP 与 A 的奖励及固定温度 |
| E5 | 融合候选网络的行为克隆 |
| E6 | 从零训练融合版 SAC |
| E7 | BC 初始化、Critic 预热及融合 SAC |
| E8 | E7 加保守筛选 |
| E9 | 固定提交归档的 CH-SAC v2 模板调参参考 |

正式评估读取冻结 NACA 家族组中满足静态尖尾缘资格的样本。物理初始化、分块或制网失败逐例记录。按固定目标比较 1,536 与 5,120 单元的网格。等预算配对要求各方法在目标的 ±5% 内且实际单元数之差在目标的 2% 内。检查、作图和报告使用真实记录。CFD 的 RANS 阶段分别报告进程结果与联合收敛判断。

UIUC 冻结留出集中的 n0012 条目不满足既有的尖尾缘资格。NACA0012 CFD 使用同一 CH-SAC v2 提交发布的解析闭合坐标。该算例只在训练和测试锁定后执行，保留坐标来源的哈希及选中的权重编号。

成功运行的目录包含逐例动作与终图、权重、配对统计、求解器日志、技术报告、Excel 和十类图。audit.json 核对冻结清单、训练交互、权重身份、评估分母、网格导出、求解历史与交付文件。

`report/evaluation_metrics.json` 保存逐种子的 UV 终图合法性、成功分块编辑数与外域块数、两档制网指标和开发验证样本效率。图 7 中的缩放雅可比分布来自代表翼型复跑后核对过的实际粗档网格。

Excel 工作簿包含两档预算的完整逐例指标和带公式的汇总。正式归档保留原运行生成的工作簿，新运行使用仓库中的 Python 导出器。
