# 参考资料登记

## 二维原型采用和对照的来源

- Haarnoja、Zhou、Abbeel 和 Levine（2018），*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*。采用连续动作、离策略回放、双 Q 估计和熵正则的基本方法。[ICML 原文](https://proceedings.mlr.press/v80/haarnoja18b.html)
- Pan、Huang、Cheng 和 Zeng（2023），*Reinforcement learning for automatic quadrilateral mesh generation: A soft actor-critic approach*。FreeMesh-RL 在二维域中逐单元生成四边形，与 Pave 的平均缩放雅可比报告值同为 0.94。作为 SAC 网格应用的先例。[论文全文](https://arxiv.org/html/2203.11203) · [期刊 DOI](https://doi.org/10.1016/j.neunet.2022.10.022)
- Kim、Kim 和 You（2024），*Non-iterative generation of an optimal mesh for a blade passage using deep reinforcement learning*。二维结构化 O/H 叶栅采用确定性单步 actor–critic 预测网格参数，四个条件下的单步质量接近逐案例优化的收敛质量。作为一步预测基线的设计参考。[论文全文](https://arxiv.org/html/2209.05280) · [期刊 DOI](https://doi.org/10.1016/j.cpc.2023.108962)
- Lim、Lee、Lee 和 Yee（2024），*Development of agent-based mesh generator for flow analysis using deep reinforcement learning*。Double DQN 用于逐层生成二维 NACA 0012 近壁网格，提供几何奖励和有效性检查的设计参照。[期刊 DOI](https://doi.org/10.1007/s00366-024-02045-4)
- DiPrete、Garimella、Garcia Cardona 和 Ray（2024），*Reinforcement learning for block decomposition of planar CAD models*。学习轴对齐切割，切割后的块由 CUBIT 生成网格。作为拓扑分块研究的先例。[期刊 DOI](https://doi.org/10.1007/s00366-023-01940-6)
- Ladson、Brooks、Hill 和 Sproles（1996），*Computer Program to Obtain Ordinates for NACA Airfoils*。采用 NACA 四位数翼型解析公式，将原系数 −0.1015 改为 −0.1036 闭合尾缘，与 NASA Glenn 高阶 NACA 0012 算例的闭合方式一致。[NASA TM 4741](https://ntrs.nasa.gov/citations/19970008124) · [NASA Glenn 算例](https://www1.grc.nasa.gov/wp-content/uploads/case_c1.2.pdf)
- Sandia CUBIT 15.2，*Metrics for Quadrilateral Elements*。作为四边形缩放雅可比定义的参考。[CUBIT 用户文档](https://www.sandia.gov/files/cubit/15.2/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/quadrilateral_metrics.htm)

上述论文用于解释方法来源和评估口径。仓库的 NumPy 实现和演示检查点均来自本原型的解析翼型实验。

## 后续登记

后续引用论文、开源实现、数据集或求解器时，请记录：

- 完整标题、作者或维护者、年份与版本。
- DOI、正式发布页或原始仓库链接。
- 对应模块、使用目的与采用的具体内容。
- 代码、数据及模型权重的许可与获取方式。
- 已核验的信息、复现条件和与本文实现的差异。

待补充主题包括自动分块、结构化网格与跨块连接、强化学习及模仿学习、学习型网格生成、网格质量评价和 CFD 数值验证。确认来源后再填写具体条目，不使用无法核查的引用。
