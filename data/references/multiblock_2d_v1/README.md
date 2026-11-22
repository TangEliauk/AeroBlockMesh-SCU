# 二维多块终态参考补充 v1

本目录保存三组二维辅助案例的 11 个官方原文件或归档成员，共 **31,757,575 字节**。
它们补充既有 [四组工程参考](../multiblock_v1/README.md)，不改写原 `public_v1`、`multiblock_v1` 或 UIUC 家族划分。
原件保留原始字节，二维边界与连接的派生产物单独存放在 [block_topology_2d_v1](../../processed/block_topology_2d_v1/README.md)。

| 案例 | 本目录原件 | 来源与使用定位 |
| --- | --- | --- |
| `backstep4` | `backstep/backstep5_4levdn.p2dfmt.gz`、配套 `.nmf` | [TMR 后向台阶](https://tmbwg.github.io/turbmodels/backstep_grids.html)；最粗一级四区二维网格，用于非凸域、角点和接口基础测试 |
| `ejector3` | `ejector/nozzle.p3dfmt`、`nozzle.inp`、`split.inp` | [CFL3D Test Cases](https://nasa.github.io/CFL3D/Cfl3dv6/cfl3dv6_testcases.html) 的三块点匹配喷射器；二维内流补充 |
| `hump4` | `hump/hump2_e_newtop.fmt`、三个 `hump2_e_oscillate_newtop.inp_*`、`README`、`split.inp` | 同一 CFL3D 官方页的四块带腔 Hump；曲壁、缝口和腔体的复杂二维辅助案例 |

[manifest.json](manifest.json)锁定每个原件和派生文件的相对路径、字节数、SHA-256、官方来源与提取方式，并记录案例用途和限制。
[七组机器可读用途表](../multiblock_case_catalog.json)与[参考指南](../../../docs/multiblock_reference_guide.md)说明本补充集在课题中的位置。
文件规模按展开后的成员统计：Hump 网格为 30,668,830 字节，其下载归档为 3,611,986 字节，二者含义不同。
源归档整体并未作为新独立案例重复入库，所选归档成员不做文本换行或数值改写。

本机已按原件声明尺寸核对坐标标量数与有限性，二维节点规模如下；薄层文件的平面一致性和接口映射另由清洗器检查：

| 案例 | 源块尺寸 | 二维节点数 |
| --- | --- | ---: |
| `backstep4` | `33×33`、`13×33`、`49×57`、`17×57` | 5,280 |
| `ejector3` | `2×31×41`、`2×31×71`、`2×101×121` | 15,693；原薄层共 31,386 点 |
| `hump4` | `2×793×217`、`2×161×121`、`2×65×121`、`2×49×217` | 210,060；原薄层共 420,120 点 |

## 获取和复现

已提交原件可直接离线清洗与检查。在仓库根目录执行：

```shell
python scripts/data/clean_2d_multiblock.py
python scripts/data/clean_2d_multiblock.py --check
python scripts/data/verify_2d_multiblock_data.py
```

`--check` 只读重算并比较已提交产物。下载器提供单独的来源复核入口，按原始资产 ID 选择：

```shell
python scripts/data/download_2d_reference_sources.py --list
python scripts/data/download_2d_reference_sources.py --ids backstep4_grid backstep4_nmf --output-root data/raw/2d_source_rebuilt
```

替代输出根下仍保存完整 `data/references/multiblock_2d_v1/...` 路径，便于与本目录比较。
后台阶网格从官方 ZIP 以有界 HTTP Range 获取；CFL3D 从固定哈希的 tar 归档取指定成员，不执行归档中的程序。
现有文件与清单不符时保留并报错，不能通过覆盖原件或修改历史清单来消除差异。

## 维度、归属与使用范围

后台阶 `.p2dfmt` 是真正二维坐标，`.nmf` 却来自两平面三维版本；面编号和索引的投影需明确记录和验证。
喷射器与 Hump 是二维问题的两平面文件表示，选择二维平面不改变原始文件，投影检查见派生 JSON。
同一几何的密度级别和格式变体不能跨划分计作独立外形；本集不分配到 UIUC 的 train/val/test。

这些是辅助分块终态参考，不含原始人工动作、人工耗时、可直接训练的专家轨迹或本项目 CFD 结果。
TMR 来源按 [TMBWG 仓库 CC0 声明](https://github.com/TMBWG/turbmodels/blob/main/LICENSE)记录；CFL3D 案例按官方公开测试包及原归属记录，未确认各样例独立的统一数据许可。
CFL3D 软件许可不自动用于所有样例网格；项目没有为第三方文件重新授权，详见本补充集的 [PROVENANCE.md](PROVENANCE.md)。历史 [THIRD_PARTY_DATA.md](../../../THIRD_PARTY_DATA.md) 保持冻结原文。
