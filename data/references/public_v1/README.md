# CFD 实验参考数据与外部资源清单

本目录包含 4 份标准化实验观测表，共 **668 行**，以及 33 项外部网格、几何与方法资源的来源、SHA-256 和用途说明。适合在 GitHub 发布轻量 CSV、元数据与复现脚本；大网格及第三方代码包采用外部引用。

## 已清洗的实验表

| 数据表 | 观测数 | 保留的独立工况 |
|---|---:|---|
| `experiments/naca0012_ladson_force.csv` | 53 | M=0.15，Re=600 万，强制转捩；80、120、180 grit 分别 17、18、18 行 |
| `experiments/naca0012_gregory_cp.csv` | 73 | Re=288 万，上表面压力数字化数据；0°、10°、15°分别 25、24、24 行；原文件未给 Mach，不从其他实验补齐 |
| `experiments/onera_m6_case2308_cp.csv` | 271 | 原实验 M=0.8395，迎角 3.06°，Re_MAC=1,172 万 |
| `experiments/onera_m6_case2565_cp.csv` | 271 | 原实验 M=0.8372，迎角 6.06°，Re_MAC=1,171 万 |

每张表有独立 `metadata/*.json`，记录条件、单位、主键、来源 URL、原始文件哈希和解释边界。M6 每个工况有 7 个展向站：前四站各 34 点，后三站各 45 点；第六站按 TMR 更正标为 eta=0.96。实验 Reynolds 数以原钝尾缘机翼平均气动弦 646.07 mm 为参考；尖尾缘 CFD 对照练习的 Re_root=1,460 万另存为计算规范，未覆盖原实验条件。

清洗采用十进制定点文本，不改变原始有效数字；保留记录顺序和原文件行号。没有插值、平滑、跨工况合并、删去失速点、填充未知工况或裁剪压力坐标。Gregory 数据中 3 个略超出 [0,1] 的 x/c 来自原始数字化表，已保留并在 `cleaning_report.json` 标记。所有 4 份原始表均与当前官方 URL 的完整 SHA-256 核对一致。

这些实验观测用于 CFD 验证，不包含分块决策标签或专家状态—动作轨迹。

## 发布仓库中的复现方式

在发布仓库根目录执行（脚本位于 `scripts/data/`；只需 Python 标准库）：

```sh
python scripts/data/prepare_reference_data.py --input-root path/to/raw_sources --output-root data/references/public_v1
python scripts/data/validate_reference_data.py --data-root data/references/public_v1
```

输入根目录可直接包含四个原始文件，也可包含 `experiments/CLCD_Ladson_expdata.dat`、`experiments/CP_Gregory_expdata.dat`、`experiments/case_2308.dat`、`experiments/case_2565.dat`。若需要独立获取这四个小表，可加 `--fetch`；加 `--verify-upstream` 可再次对照当前官方响应哈希。上游内容发生变化时脚本报错，要求审查版本，不会默默替换基线。

`cleaning_report.json` 给出行数、分组数量、主键唯一性、非有限值检查、CSV 读回一致性和全部输出哈希。`validate_reference_data.py` 可在没有原始工作区的全新克隆中核验已发布 CSV。

## 外部网格与方法资源

`reference_manifest.json` 仅记录可移植的资产 ID、官方下载入口、原始 SHA-256、精确大小、用途和限制，不包含本机绝对路径。它覆盖 CGNS 多块样例、CRM、DLR-F6 ICEM 工程、NACA0012 五级网格、ONERA M6 几何与网格工具、par-quad 和 SRL-AssistedAFM。

```sh
python scripts/data/download_reference_assets.py --manifest data/references/public_v1/reference_manifest.json --output-root .cache/reference_assets --list
python scripts/data/download_reference_assets.py --manifest data/references/public_v1/reference_manifest.json --output-root .cache/reference_assets --ids cgns_channel_12 naca0012_familyI_level7
```

下载器只处理显式指定的资产；校验长度、SHA-256 及 gzip CRC。NACA 网格使用清单中固定的 ZIP 成员偏移，通过有界 HTTP Range 取单个成员，不下载完整约 2.98 GB 归档。服务器不支持正确 Range 时会报错。文件版本变化或已有文件哈希冲突时也会报错。

CGNS/CRM 中的 `.cgns` 文件可能为 ADF 而非 HDF5；清单保留已完成的检查范围。成品网格不等于真实专家操作记录。par-quad 提供的是合成示范生成工具；SRL-AssistedAFM 两包为 213+147 个非结构四边形成品网格，不能当作 360 个气动多块分块轨迹。

许可与署名证据见 `THIRD_PARTY_DATA.md`、`license_evidence.json`。任何本项目脚本许可证都不覆盖第三方原始数据。发布数据未生成虚假训练轨迹。
