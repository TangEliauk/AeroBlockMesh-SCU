# 清洗后公开数据集 v1

本数据版本服务于CFD多块结构化网格智能分块研究。提供规范化翼型几何、固定的家族分组划分、实验验证表以及完整来源和处理记录。**几何数据不包含专家动作、块拓扑或SAC经验回放；还需要实现分块环境，并按研究路线采集专家示范。**

## 数据结果

| 数据 | 结果 | 位置 |
|---|---|---|
| UIUC原始坐标 | 1665个DAT，原始文件保持不变 | 按来源清单下载，原始数据不入Git |
| 清洗后翼型 | 1607个精确不同的规范化采样轮廓；对应1613个源文件 | [samples.jsonl](../data/processed/uiuc_v1/samples.jsonl) |
| 重复与隔离 | 合并6个规范化精确重复源，保留别名；隔离52个待审源并记录原因 | [处理清单](../data/processed/uiuc_v1/) |
| 固定划分 | train 1277，val 160，test 170；共186个连通家族组 | [划分目录](../data/splits/uiuc_v1/) |
| 实验参照 | NACA0012测力53行、上表面压力73行；M6两个工况各271点，共668行 | [实验CSV与工况元数据](../data/references/public_v1/) |
| 多块工程参考 | CGNS、DLR-F6、CRM、M6、ICEM等，仅记录来源、SHA256和用途 | [参考数据清单](../data/references/public_v1/reference_manifest.json) |

坐标仅做平移、统一弦长尺度、精确重复点移除和绕序规范化，不平滑、不重采样、不插补。多边形为逆时针，首点不在末端重复，闭合边隐式存在；原始尾缘间隙和闭合处理标志逐源记录。筛查阈值是常规单段、弦向基本对齐翼型的保守操作规则，隔离不等于原始外形错误。详见[几何说明](../data/processed/uiuc_v1/README.md)、[清洗QA](../data/processed/uiuc_v1/qa_report.json)和[划分报告](../data/splits/uiuc_v1/split_report.json)。

同名家族、可识别的派生名及检测到的近重复进入同一连通组。NACA连通家族留作最终test，供计划中的NACA0012基准独立评价；不能再用这部分数据选择奖励、超参数或检查点。名称规则和129截面相似性筛查不能证明发现了全部未标注的几何家族，完整边界与阈值均已记录。

## 读取和核验

在仓库根目录使用Python 3.10或更新版本，无第三方运行依赖：

```shell
python scripts/data/verify_public_data.py
python -m unittest discover -s tests/data -p "test_*.py" -v
```

每条`sample_id`在`train.json`、`val.json`或`test.json`中恰好出现一次。`source_files`保留原始哈希、标题、坐标变换与质量标志，`aliases`保留合并的来源名。读取示例：

```python
import json
from pathlib import Path

train_ids = set(json.loads(Path("data/splits/uiuc_v1/train.json").read_text()))
with Path("data/processed/uiuc_v1/samples.jsonl").open(encoding="utf-8") as stream:
    train_geometries = [row for line in stream if (row := json.loads(line))["sample_id"] in train_ids]
```

## 从公开原始文件复现

```shell
python scripts/data/download_public_sources.py --extract-uiuc
python scripts/data/clean_uiuc.py --input data/raw/public_v1/uiuc/coordinates --output data/processed/uiuc_rebuilt
python scripts/data/split_uiuc.py --input data/processed/uiuc_rebuilt/samples.jsonl --output data/splits/uiuc_rebuilt --config data/splits/uiuc_v1/split_config.json --raw-root data/raw/public_v1/uiuc/coordinates
python scripts/data/prepare_reference_data.py --input-root data/raw/public_v1/experiments --output-root data/references/public_rebuilt
```

建议先输出到`*_rebuilt`目录比较，不覆盖冻结的v1；生成时间和是否执行在线来源核验的字段可能不同，几何JSONL、ID划分和实验CSV应可复现。下载器按冻结SHA256验证，已存在且匹配的原始文件跳过，不覆盖不匹配文件。原始坐标留在本地；新增`data/raw/`建议保持忽略。

## 工况和归属

Ladson测力按三种粗糙带分组；Gregory压力属于不同雷诺数的数字化上表面数据，三个超出单位弦长区间的原始测点保留并标记；M6两个工况分表保存。不得把不同实验条件拼成同一真值。标准化保留全部原始数值观测，不把真实极值自动删除。

大体积多块网格保持原始科学数据，未改动节点或单元；仓库提供可追溯的下载清单，未将其打包为SAC标签。CGNS/ADF与HDF5需用相应读取器。数据质量检查不等于CFD验证，本次未执行SAC训练、网格生成或求解器复算。

第三方数据保留各自归属，见[THIRD_PARTY_DATA.md](../THIRD_PARTY_DATA.md)。UIUC未核实统一数据许可，不套用本项目代码许可；TMR来源声明和原实验归属分别记录。
