# 分块终态清洗产物 v1

本目录包含四组源案例的确定性 JSON 产物，用于后续块拓扑读取、接口核验和工程结果对照。原件放在 [multiblock_v1](../../references/multiblock_v1/README.md)，不会就地修改。

| 文件 | 实际清洗内容 |
| --- | --- |
| `cgns_channel12.json` | 12 个结构区、20 组接口、32 个边界；尺寸、坐标摘要、Family、范围及索引变换 |
| `cgns_airfoil4.json` | 4 个结构区、8 组接口、20 个边界；第三方向的两层点平面保留，不擅自降维 |
| `dlr_f6_icem.json` | 三份 BLK 中可解析的点、节点与 SUPERBLOCKS 等源记录及引用检查；保留未知属性，不推断最终活动块集合 |
| `onera_m6.json` | PLOT3D 4 块、316,932 点的坐标摘要与 GMAN 声明的 24 个面、6 组双向接口 |

JSON 包含源路径、源 SHA-256 和验证范围；不同格式的专有字段明确保留，不声称形成覆盖全部工程语义的稳定训练 API。CGNS 使用原来的 1-based 索引；其他格式的编号也沿用源约定，不混用。原始坐标保存在原件中，JSON 不重复存储整套网格。

清洗包括数值有限性、尺寸/计数、引用有效性以及可解析接口的检查。CGNS 根据原索引变换比较接口节点；ICEM 保留不能可靠解释的专有语义。M6 的 `units feet` 是 GMAN 指令声明，不能据此自动换算 PLOT3D 坐标。未知单位保持未知。

两个 CGNS 案例的双向接口范围与变换均互逆，逐点 XYZ 差为零。M6 的 6 组接口分别找到唯一的面索引映射并逐点匹配；这些映射是通过源坐标验证得到的候选，不冒充 GMAN 原文件中显式提供的变换。

ICEM 每份 BLK 有 77 个 PREPOINTS、548 个 NODES 和 374 条 SUPERBLOCKS 源记录，不能将后者直接称为 374 个活动流体块。其中 24 条较高阶记录引用了 28 个 NODES 段未显式定义的中间节点编号；已定义的粗角点经过引用检查，未解析的中间编号单独保留，未插值补点。相关坐标摘要仅覆盖可解析节点，产物明确标记解析不完整。

清洗没有平滑、重采样、删除共享接口节点或转换尺度；也未计算单元 Jacobian、认证网格质量、运行 CFD，或生成专家动作轨迹。三个 ICEM 密度文件属于同一几何，不是三个独立外形。M6 等工程参考不自动纳入 UIUC 训练划分。

在仓库根目录复现或只读检查：

```shell
python -m pip install -e ".[dev,data]"
python scripts/data/clean_cgns_multiblock.py
python scripts/data/clean_engineering_multiblock.py
python scripts/data/clean_cgns_multiblock.py --check
python scripts/data/clean_engineering_multiblock.py --check
python scripts/data/verify_multiblock_data.py
```

读取示例：

```python
import json
from pathlib import Path

case = json.loads(Path("data/processed/block_topology_v1/cgns_channel12.json").read_text(encoding="utf-8"))
print(case["case_id"])
```

新的发布清单同时锁定原件和清洗 JSON 的字节数及哈希，CI 重跑清洗并与已提交结果比较，防止解析程序变化却未更新产物。
