# 二维分块终态清洗产物 v1

本目录保存三组辅助案例的确定性二维参考 JSON，源文件保存在 [multiblock_2d_v1](../../references/multiblock_2d_v1/README.md)。
独立[发布清单](../../references/multiblock_2d_v1/manifest.json)锁定原件与派生文件，既有四组 `block_topology_v1` 不变。

| 文件 | 原始表示与参考内容 |
| --- | --- |
| `backstep4.json` | 四块真正二维 PLOT3D；块边界折线、源索引与坐标匹配得到的接口候选；NMF 原件另行保留 |
| `ejector3.json` | 三块薄层 PLOT3D；二维平面、边界折线与连续子边坐标匹配结果；配套求解输入另行保留 |
| `hump4.json` | 四块带腔 Hump 薄层 PLOT3D；曲壁、缝口和腔体周边的块边界与局部接口参考 |

每份 JSON 记录 `case_id`、源文件及哈希、`blocks`、`interfaces`、`boundaries`、`projection`、`checks` 和 `limitations`。
不同编号体系沿用原件约定并显式注明，不将源索引、数组偏移和二维边索引混用。
边界折线用于查看终态分块，原始完整网格坐标仍在参考目录；不在 JSON 中复制整套体网格。

## 本次清洗结果

| 案例 | 块数 | 去重接口候选 | 比较节点对 | 未分类边界段 |
| --- | ---: | ---: | ---: | ---: |
| `backstep4` | 4 | 3 | 123 | 11 |
| `ejector3` | 3 | 2 | 112 | 10 |
| `hump4` | 4 | 3 | 459 | 12 |

坐标数量与尺寸一致，全部数值有限；接口映射均有互逆邻接记录，逐点最大坐标差均为 0，绝对容差为 `1e-10`。
比较节点对按去重接口统计，不把双向邻接重复计数；未匹配边界段保留为未分类，不能自动称为壁面或远场。

## 坐标与连接检查

后台阶二维网格无需删除坐标方向；配套 NMF 属于薄层三维版本，不能不经映射直接套用其面编号与索引。
喷射器和 Hump 的二维表达从薄层文件选取平面，先检查两平面的面内坐标一致性，记录所选源坐标轴、保留平面与误差。
本次后台阶保留原 `X,Y`；喷射器和 Hump 选择源逻辑 `I=1`、保留 `J,K` 网格的物理 `X,Z`，两层 `Y` 分别为 `0,-1`，面内坐标最大差为 0。
投影不归一化、缩放、平滑或重采样原坐标；所有原件保持原字节，未知单位不会自动补齐。

接口按边界坐标寻找连续子边匹配并逐点核验，保留反向索引和局部边区间；这些映射属于坐标推断候选，不冒充源文件显式给出的变换。
不能只因两块相邻或边端点一致就认为整条接口已通过，实际接口数、比较点数和容差结果以各 JSON 的 `checks` 为准。
清洗器没有自动解析 NMF、CFL3D 求解输入的物理边界类型及其他全部语义；这些文件保留原件供独立对照，不构成完整的求解器配置转换器。
当前仅支持不同块之间的点匹配接口，未提供周期、自连接、非匹配或 overset 接口支持；坐标匹配也不证明各块内部互不重叠。

## 复现和使用

在仓库根目录运行：

```shell
python scripts/data/clean_2d_multiblock.py
python scripts/data/clean_2d_multiblock.py --check
python scripts/data/verify_2d_multiblock_data.py
```

`--check` 只读重算；更新脚本导致产物变化时，应审查科学含义后再更新独立清单的派生哈希。
读取示例：

```python
import json
from pathlib import Path

case = json.loads(Path("data/processed/block_topology_2d_v1/backstep4.json").read_text(encoding="utf-8"))
print(case["case_id"], len(case["blocks"]))
```

这些输出是分块终态参考，不是专家动作轨迹、SAC 回放或统一的强化学习环境接口。
本次不运行策略训练、人工效率对照、CFD 求解或单元质量认证；坐标与接口检查不能替代这些工作。
三组均为辅助案例，定位和七组间的区别见 [参考指南](../../../docs/multiblock_reference_guide.md)与[机器可读用途表](../../references/multiblock_case_catalog.json)。
