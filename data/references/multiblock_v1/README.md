# 多块分块终态参考 v1

本版本补入四组分块终态参考，包含 5 个原始文件，共 17,246,609 字节。原件逐字节保留，清洗后的拓扑、边界与检查结果另存于 [block_topology_v1](../../processed/block_topology_v1/README.md)。它们是少量工程样例，不是专家动作轨迹或 SAC 经验回放。

| 案例 | 原件 | 用途与解释边界 |
| --- | --- | --- |
| 十二块通道 | `channel12/sqnz_s.hdf.cgns.gz` | 结构化块、边界与一对一接口核验 |
| 四块多段翼型 | `airfoil4/multi_hdf5.cgns.gz` | 准二维气动结构网格；保留第三方向两层点平面 |
| DLR-F6 ICEM | `dlr_f6_icem/dlr_f6_wingbody_icem_REV2.zip` | 一个 TIN 与粗、中、细三份 BLK；同一几何的密度变体，部分专有字段保留未解释 |
| ONERA M6 | `onera_m6/m6wing.x.fmt`、`onera_m6/gman.a.com` | 四块 PLOT3D 网格与配套边界/连接命令；未执行 GMAN 或 CFD 求解器 |

本目录的 [manifest.json](manifest.json) 是新增实体和派生文件的发布清单，记录相对路径、字节数、SHA-256 和来源。原 [public_v1 清单](../public_v1/reference_manifest.json) 保持历史快照，里面的 `reference_only_not_bundled` 描述的是当时发布范围；新增实体以本目录清单为准。约 55 MB 的 DLR-F6 CGNS 和约 115 MB 的 CRM 仍只保留官方下载引用。

在仓库根目录安装清洗工具并复现：

```shell
python -m pip install -e ".[dev,data]"
python scripts/data/clean_cgns_multiblock.py
python scripts/data/clean_engineering_multiblock.py
python scripts/data/verify_multiblock_data.py
```

检查提交产物是否可重现，使用 `--check`，不会覆盖文件：

```shell
python scripts/data/clean_cgns_multiblock.py --check
python scripts/data/clean_engineering_multiblock.py --check
python -m unittest discover -s tests/data -p "test_*.py" -v
```

清洗不改节点、不平滑、不重采样、不合并本来就需要分开的接口节点，也不更改几何单位。输出只包含已解析且可检查的字段，未知的格式语义与未执行的物理验证会明确记录。来源与署名说明见 [PROVENANCE.md](PROVENANCE.md)。
