# 第三方数据归属与许可证据

核实日期：2026-11-22。原始数据、几何、网格与第三方代码保留各自来源及权利声明。本项目的清洗脚本、清单和说明不能把第三方资产重新授权为 MIT，也不声称这些来源审查或认可本项目。

## 已标准化的 TMR 实验数据

上游 [TMBWG/turbmodels 仓库](https://github.com/TMBWG/turbmodels)明确标注 **CC0-1.0**，其 [LICENSE 原文](https://raw.githubusercontent.com/TMBWG/turbmodels/main/LICENSE)可直接核实。该仓库是 [Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/) 的公开源代码与数据来源。此次 4 份数字表以该来源声明记录，未改称 MIT；CC0 不替第三方清理所有可能存在的权利，原实验作者与报告标识仍保留在每份 JSON。

- NACA0012 测力：C. L. Ladson，NASA TM 4074，1988；[TMR 数据页](https://tmbwg.github.io/turbmodels/naca0012_val.html)与[数字表](https://tmbwg.github.io/turbmodels/NACA0012_validation/CLCD_Ladson_expdata.dat)。
- NACA0012 上表面压力：N. Gregory、C. L. O'Reilly，R&M 3726，1970；[TMR 数据页](https://tmbwg.github.io/turbmodels/naca0012_val.html)与[数字化表](https://tmbwg.github.io/turbmodels/NACA0012_validation/CP_Gregory_expdata.dat)。数字化近似和原始来源均保留。
- ONERA M6 压力：V. Schmitt、F. Charpin，AGARD AR-138，1979；[TMR 数据页](https://tmbwg.github.io/turbmodels/onerawingnumerics_val.html)、[Case 2308](https://tmbwg.github.io/turbmodels/Onerawingnumerics_val/case_2308.dat)、[Case 2565](https://tmbwg.github.io/turbmodels/Onerawingnumerics_val/case_2565.dat)。ONERA 原实验归属、原钝尾缘工况和站位更正与清洗 CSV 一同保存。

## UIUC 翼型坐标

[官方坐标库](https://m-selig.ae.illinois.edu/ads/coord_database.html)公开提供单文件与打包下载，并链接多个使用和转换工具；[来源说明](https://m-selig.ae.illinois.edu/ads.html)说明坐标由 Selig、David Lednicer 及其他贡献者累积而成；[FAQ](https://m-selig.ae.illinois.edu/ads_faq.html)没有给出统一的 MIT、CC-BY 或 CC0 数据许可。网页上的版权页脚也不能直接当作每个坐标条目的独立许可。

因此，UIUC 第三方数据许可标识记录为 **NOASSERTION（未确认统一数据许可证）**，保留原始条目名称、文件标识、官方 URL、哈希与贡献来源。本项目未把“未找到统一许可证”解释为“全部禁止下载或再分发”，也未将公开下载自动等同于权利全部放弃；对上传的衍生数据说明清洗操作与来源，不作新的第三方权利保证。该状态不应被错误替换成项目脚本的 MIT 许可。

## CGNS 样例、NASA DPW、CRM 与 ICEM 工程

[CGNS Examples](https://cgns.org/current/examples.html)说明样例由用户提交、按现状提供，未由委员会逐一认证；[CGNS Licensing](https://cgns.org/license.html)明确针对软件，不能自动当作所有用户提交网格的独立数据许可。文档页的 CC0 页脚同样不用于覆盖外部贡献资产的独立权利。

[DPW3 ICEM 目录](https://dpw.larc.nasa.gov/DPW3/multiblock_icem_cfd/)和[DPW4 Airbus CRM 目录](https://dpw.larc.nasa.gov/DPW4/multiblock_Airbus/CGNS_Grids/)公开提供文件和说明，当前核实页面未给每份资产单独的统一再分发许可证。本次公开发布采用官方下载入口、精确字节数与 SHA-256，未复制这些网格/工程大文件；这是一种发布范围选择，并非判定原资产禁止再分发。

NASA 的[图像与媒体使用指南](https://www.nasa.gov/nasa-brand-center/images-and-media/)同时区分 NASA 内容与第三方内容，不能据 NASA 托管就把 Airbus、ONERA 等贡献的所有数据宣称为 NASA 原创或统一公共领域。TMR 仓库的明确 CC0 声明另按上一节记录。

## 第三方方法代码与示例包

- **par-quad**：[作者仓库](https://github.com/ArjunNarayanan/par-quad)，固定提交 `b8d4e22bc4adf217445f7260a774b0c73ef9d719`；[MIT LICENSE](https://github.com/ArjunNarayanan/par-quad/blob/b8d4e22bc4adf217445f7260a774b0c73ef9d719/LICENSE)。代码再分发需保留版权与许可告示。本次仅提供哈希和固定提交 URL；合成示范不是原始气动专家示范。
- **SRL-AssistedAFM**：[作者仓库](https://github.com/CMU-CBML/SRL-AssistedAFM)，固定提交 `e35a87895fa0e7c9191325f78efe8901bd0ee34e`；[BSD-3-Clause LICENSE](https://github.com/CMU-CBML/SRL-AssistedAFM/blob/e35a87895fa0e7c9191325f78efe8901bd0ee34e/LICENSE)。代码再分发需保留声明且不得借权利人名称暗示背书。本次 ZIP 只列引用；未另确认示例数据包独立于仓库声明的其他许可证，也未将它们更名为本项目数据。

研究引用应同时标注数据提供方与原论文/报告。许可状态和科学适用性是两个问题：即使有明确开放许可，也不能把少量终态网格或单个外形的多级密度当成独立的专家训练库。
