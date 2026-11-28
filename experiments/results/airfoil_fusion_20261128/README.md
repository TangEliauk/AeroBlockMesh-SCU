# 翼型自动分块融合正式结果

这里归档本地正式运行 `airfoil_fusion_formal_v2` 的交付物。训练矩阵共 25 项 SAC 训练和 5 项行为克隆。60 份 best / final 权重由独立评估锁校验。审计覆盖 67 项评估、每项 90 个冻结翼型，共 6,030 条记录。

- [技术报告和图件](report/report.md)
- [逐例 Excel 工作簿](report/实验结果.xlsx)
- [评估汇总](evaluation/summary.json)
- [训练矩阵及权重摘要](training_matrix.json)
- [封存审计](audit.json)
- [CFD 求解记录](cfd/summary.json)
- [训练耗时与复现说明](../../../docs/airfoil_fusion_performance.md)

正式 CFD 包含 6 个网格工况和 12 次实际求解进程，联合收敛判据通过数为 0。这里保留了网格、配置和历史曲线。

从完整 Git 检出安装复现依赖后，在仓库根目录执行以下命令。脚本校验冻结输入、参考基线、权重、完成凭证、报告和 CFD 文件，并在 CPU 上用 E7 种子 42 的封存最佳权重复算一例独立翼型，核对动作、终图与两档网格。

```sh
python scripts/verify_airfoil_fusion_release.py --replay
```

完整逐例表保存在工作簿中。原始训练日志、逐决策轨迹与恢复检查点保留在本地正式运行目录。GitHub 归档用于读取和核验原结果，重训通过新的运行目录使用当前源码重新冻结输入。[完整复现指南](../../../docs/airfoil_fusion.md)
