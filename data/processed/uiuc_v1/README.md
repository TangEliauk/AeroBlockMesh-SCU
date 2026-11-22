# UIUC 清洗后二维训练几何

这是从 UIUC 官方坐标整包得到的**几何子集**，用于几何编码、生成环境输入和家族外测试。它不含专家动作、多块拓扑、块间连接或 CFD 真值，不能直接作为行为克隆示范。

官方来源：[UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html)。输入整包为 `coord_seligFmt.zip`，SHA256 为 `5a17e5400322d7d1b0171554cee3764bcf34aa2cc349e0e4e95aeafe5062fda7`。第三方翼型数据的归属保持不变；项目代码许可证不为 UIUC 数据重新授权。来源页未确认统一数据再分发许可证，使用时应保留作者和来源信息。

## 本次处理结果

从 1,665 个原始 DAT 文件中，1,613 个文件通过本流水线的保守几何筛查。精确规范化去重合并 6 个别名文件，得到 **1,607 个独立采样轮廓**；52 个文件进入隔离清单，未被静默修复或丢弃。这里的“独立”只指精确轮廓不同，不等于统计独立或不存在近重复。最终统计、阈值和拒绝原因以 `summary.json` 为准。

| 文件 | 内容 |
|---|---|
| `samples.jsonl` | 每行一个规范化几何、全部来源引用和别名 |
| `accepted_manifest.jsonl` | 每个通过检查的原始文件，及其对应 sample_id |
| `rejected_manifest.jsonl` | 每个隔离文件的原始哈希、原因和触发量 |
| `duplicate_manifest.jsonl` | 精确重复来源及保留的代表文件 |
| `summary.json` | 计数、配置、判定边界和来源归属 |
| `scripts/clean_uiuc.py` | 仅依赖 Python 标准库的清洗 CLI |
| `tests/test_clean_uiuc.py` | 数据损坏、重复、坐标变换、保形和源文件不变性的回归测试 |

## 坐标与来源字段

`polygon` 是二维数值数组，方向为逆时针，从 `(x,y)` 字典序最小点开始，不重复存放末端首点；`closure="implicit_last_to_first"` 表示最后一点到第一点的边隐式存在。

对全部坐标只做平移与统一正比例缩放：

```text
chord = xmax - xmin
x_clean = (x_raw - xmin) / chord
y_clean = (y_raw - y_LE) / chord
```

`y_LE` 取最小 x 处的最小 y。**不旋转、不平滑、不重采样、不补点**。重复的连续坐标、重复末端首点只按精确相等移除。保留钝尾缘两端；接受的开口轮廓必须在尾缘附近结束，并满足间隙限制，其隐式闭合段不增加新坐标。

`sample_id` 由 `canonical_sha256` 前 16 位组成。原始十进制数先解析为精确 Fraction，进行规范化、统一绕序和起点后计算完整哈希。浮点 polygon 只负责便捷取用；不会用舍入后的近似坐标判定重复。增加共线点或微小坐标差异仍会保留为不同样本；家族与近重复划分应在下一步处理。

`source_files` 保存相对输入目录的路径、原文件名、原字节 SHA256、标题、输入格式、完整变换参数和处理标志。`aliases` 保存来源文件 stem。所有可发布记录均不包含本机绝对路径或网页 HTML。

## 筛查范围与可配置阈值

流水线支持标题、`#` 注释、DOS EOF、D 指数和可识别的 Lednicer 双表面点数头；若点数头存在，必须与数据行数一致，前缘连接也必须在配置容差内。数值行损坏或出现非有限数时，文件进入隔离清单。

| 判定项 | 默认值（规范化弦长单位） |
|---|---:|
| 最少/最多顶点 | 12 / 5,000 |
| 最小面积 | 0.00001 |
| 最大 y 跨度 | 0.60 |
| 最短允许边 | 大于 0.0000000001 |
| 最大边 | 0.40 |
| 开口两端距最大 x | 不超过 0.02 |
| 开口尾缘最大间隙 | 0.10 |
| 两条前缘到尾缘支路各自最大 x 回退量 | 0.02 |
| 线段相交/接触计算 epsilon | 0.000000000001 |
| Lednicer 前缘连接容差 | 0.000001 |

还检查非相邻边自交或接触、相邻边反向重叠。多个轮廓拼接、明显不符合单段且弦向基本对齐假设的几何会隔离。阈值是本项目的**保守训练子集筛选规则**，不是普适的翼型或 CFD 合格标准。隔离文件可能是有效的厚翼型、偏转部件、原始尾缘精度问题或需人工审查的数据，不代表原始翼型一定错误。

本次没有验证曲率连续性、可制造性、边界层铺网适用性、单元 Jacobian、求解器可读性或气动性能。相交测试采用已公开的浮点容差，不宣称精确谓词的鲁棒性。

## 复现

在本目录布局下：

```shell
python scripts/clean_uiuc.py --input PATH_TO_ORIGINAL_COORDINATES --output PATH_TO_SEPARATE_OUTPUT
python -m unittest discover -s tests -p "test_*.py" -v
```

发布仓库将脚本放在 `scripts/data/`、测试放在 `tests/data/` 时：

```shell
python scripts/data/clean_uiuc.py --input data/raw/public_v1/uiuc/coordinates --output data/processed/uiuc_v1
python -m unittest discover -s tests/data -p "test_clean_uiuc.py" -v
```

`--input` 和 `--output` 必须是彼此独立、互不嵌套的目录，原始坐标只读。可用 `--thresholds-json config.json` 覆盖默认阈值，键名见 `Thresholds` 或 `summary.json`。改变阈值后应重新生成家族划分并保留对应清洗配置。几何与来源输出顺序确定；`summary.json` 的生成时间随执行更新。
