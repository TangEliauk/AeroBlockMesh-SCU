# 数据目录

仓库已包含公开数据 v1：规范化 UIUC 翼型几何、固定的训练/验证/测试划分，以及标准化的实验参照表。数据范围、清洗结果、读取示例与复现命令见 [清洗后公开数据集 v1](../docs/public_data_v1.md) 和 [README_cleaned.md](README_cleaned.md)。几何数据尚不包含专家动作、分块拓扑或强化学习经验回放。

另补入四组 [多块分块终态参考](references/multiblock_v1/README.md)：CGNS 十二块通道、四块多段翼型、DLR-F6 ICEM 工程和 ONERA M6 四块网格与配对文件。5 个原件保留原始字节，清洗后的拓扑、边界及核验记录单独保存，不作为专家操作轨迹。

| 目录 | 用途 |
| --- | --- |
| `raw/` | 保留原始几何、原始参考数据及其来源信息 |
| `processed/uiuc_v1/` | 已纳入版本管理的规范化翼型几何、源文件映射与清洗 QA |
| `processed/block_topology_v1/` | 四组终态参考的清洗 JSON、源哈希与解析范围 |
| `references/public_v1/` | 已纳入版本管理的实验参照表、工况元数据及外部参考资产清单 |
| `references/multiblock_v1/` | 四组终态参考的原件、来源与新增发布清单 |
| `expert/` | 预留专家分块示范，尚未提供可用示范集 |
| `splits/uiuc_v1/` | 已纳入版本管理的固定家族分组划分、配置与报告 |

公开数据 v1 和多块终态 v1 的上述发布产物保留在 Git 中；`raw/`、`expert/` 及其他生成的处理结果默认忽略。`.gitkeep` 只用于跟踪空目录。大型 DLR-F6 CGNS、CRM 等仍按来源清单获取，不在轻量划分目录中存放数据副本。

`scripts/data/` 已提供下载、清洗、划分、实验数据整理与核验脚本；`tests/data/` 包含数据回归测试。多块清洗器需要可选的 `data` 依赖。可在仓库根目录离线核验已提交数据并运行测试：

```shell
python scripts/data/verify_public_data.py
python scripts/data/verify_multiblock_data.py
python scripts/data/clean_cgns_multiblock.py --check
python scripts/data/clean_engineering_multiblock.py --check
python -m unittest discover -s tests/data -p "test_*.py" -v
```

添加数据前，应确认来源与使用许可，记录数据版本、校验值、单位、坐标系及预处理参数。原始文件不要就地修改；相同几何及其派生样本应保持在同一数据划分内。

详细的数据隔离与追溯约定见 [docs/datasets.md](../docs/datasets.md)。第三方数据归属及许可记录见 [THIRD_PARTY_DATA.md](../THIRD_PARTY_DATA.md)，不能由项目代码许可替代。`src/smartcfd` 中的分块、训练、网格生成和 CFD 算法仍为占位模块。

## 二维辅助参考补充

| 目录或文件 | 用途 |
| --- | --- |
| `references/multiblock_2d_v1/` | 三组二维辅助案例的 11 个原件、配套输入及独立来源清单 |
| `processed/block_topology_2d_v1/` | 后台阶、喷射器与带腔 Hump 的二维边界、投影和接口核验 JSON |
| `references/multiblock_case_catalog.json` | 七组终态参考的维度、参考信息、用途、定位和限制 |

新增目录与其原件、派生文件独立版本管理，不改写旧数据版本；获取与复现见 [二维参考原件](references/multiblock_2d_v1/README.md)、[派生产物](processed/block_topology_2d_v1/README.md)和[七组使用指南](../docs/multiblock_reference_guide.md)。它们不进入 UIUC 划分，也不是专家操作轨迹。
