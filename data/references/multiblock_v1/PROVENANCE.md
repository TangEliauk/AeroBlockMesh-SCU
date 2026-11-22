# 原件来源与处理边界

以下文件从项目已获取的公开原件复制，源文件未修改。复制后逐项与冻结的 [public_v1/reference_manifest.json](../public_v1/reference_manifest.json) 比对字节数及 SHA-256，完全一致。没有重新生成、修补或冒充原始网格。

| 原件 | 提供方与发布入口 |
| --- | --- |
| `sqnz_s.hdf.cgns.gz` | CGNS 用户提交样例，[十二块收缩通道](https://cgns.org/current/examples.html#constricting-channel) |
| `multi_hdf5.cgns.gz` | CGNS 用户提交样例，[四块多段翼型](https://cgns.org/current/examples.html#d-multielement-airfoil) |
| `dlr_f6_wingbody_icem_REV2.zip` | [NASA DPW-3 ICEM 目录](https://dpw.larc.nasa.gov/DPW3/multiblock_icem_cfd/)，保留包内 TIN/BLK 的原有声明 |
| `m6wing.x.fmt`、`gman.a.com` | [NASA Glenn ONERA M6 算例](https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing01/m6wing01.html) |

GMAN 文件在原工作区以 `m6-gman.a.txt` 存放，本次按官方下载名称保存为 `gman.a.com`；仅变更存放名称，字节和哈希未变。下载 URL、精确字节数与哈希以 [manifest.json](manifest.json) 为准。

这些第三方资产保留原来源及原署名，不套用本项目代码许可证。既有来源审查对 CGNS 用户样例、DPW/Glenn 工程文件未确认统一的资产级再分发许可证，清单保留未确认状态，不将网站页脚或软件许可证改写为这些文件的统一许可。详情见冻结的 [THIRD_PARTY_DATA.md](../../../THIRD_PARTY_DATA.md) 和 [许可记录](../public_v1/license_evidence.json)。

派生 JSON 记录源文件哈希及解析范围，只表达清洗程序实际读到的终态信息。没有从成品网格倒造专家操作顺序，也没有补写不存在的单位、边界或 CFD 真值。ICEM 的专有记录不自动等同于已确认的活动流体块；不同来源的 DLR-F6 网格和工程也不假定逐点对应。
