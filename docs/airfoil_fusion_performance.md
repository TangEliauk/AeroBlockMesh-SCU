# 翼型自动分块训练耗时诊断

正式实验的主要开销来自 CPU 候选枚举、曲边几何检查和串行验证。E8 在生成每步动作掩码时，还会对候选动作执行两层可行性搜索。PyTorch 的梯度更新只占其中一小部分。

| 项目 | 计时范围 | 实测 |
|---|---|---:|
| E8 正式种子 42、43 | 每种子 4,000 步。单步包括动作选择及环境转移 | 中位数 3.77、3.92 秒，P90 为 8.05、7.88 秒 |
| E7 正式种子 42、43 | 同一单步计时口径 | 中位数 0.31、0.30 秒，P90 为 1.56、1.64 秒 |
| SAC CPU 更新 | 固定的真实示范批次，batch size 32，预热后 40 次 | 中位数 64.1 毫秒 |
| SAC CUDA 更新 | 同一批次序列，逐次同步，RTX 5060 Laptop GPU | 中位数 60.2 毫秒 |
| E8 种子 42 验证 | 9 次检查点各评估 32 个翼型，逐例墙钟之和 | 约 1.39 小时 |

正式日志的单步范围不含梯度更新、回合初始化、验证和检查点写盘。筛选日志属于动作执行前的掩码，耗时与前一环境转移会重叠。验证统计使用墙钟时间，系统停顿会抬高个别读数。这些口径不能直接相加作进程总耗时。

## CPU 剖析

在冻结训练集的一例上，对一次 E8 初始化与一次插点交互执行 `cProfile`。整段 13.09 秒，其中两次筛选累计 11.69 秒，曲边几何验证的 738 次调用累计 11.82 秒。这两类时间重叠。主要几何算子包括对称差 3.91 秒、区域合并 3.11 秒、尾迹 Hausdorff 检查 1.33 秒和相交运算 1.20 秒。

当前三路并发按方法和种子分配。每条训练进程依次产生状态与动作，每个检查点内的 32 例验证也依次执行。验证函数每次创建新的几何池，导致后续检查点重新计算已经遇到的相同候选。

## 缓存复用试验

用 E8／种子 42 的初始化权重验证一个真实开发翼型。空缓存时墙钟 16.64 秒，进程 CPU 时间 16.56 秒。复用同一翼型的环境缓存后用时 0.064 秒，4 个状态均命中筛选缓存。动作、奖励、掩码所在观测的哈希、终图及两档网格评分均对照核验。

单例适用于已经遇到过的轨迹。新拓扑或不同策略仍要计算未命中的状态。继续保留原有候选、掩码和证明规则，通过跨检查点缓存和按算例并行来压缩重复工作。

复查完整验证日志中重复的动作、奖励和终图序列。E8／种子 42 的 9 次验证共 288 例次，其中 256 例次重复了该翼型在早先检查点上的完整轨迹。种子 43 在同样的 288 例次中有 224 例次重复。命中率还受到缓存容量与访问顺序影响。[重复轨迹计数](../experiments/results/airfoil_fusion_20261128/performance/repeated_archived_trajectories.json)

在 8 个开发翼型和 3 份实际 E8 权重上对照验证。串行、每个检查点清空缓存用了 352.08 秒。跨检查点复用缓存后，串行耗时 118.70 秒。两个 CPU 子进程各处理固定翼型分片并保留缓存，含子进程启动用时 62.70 秒，比冷缓存基线快 5.6 倍。24 条结果的顺序、动作、奖励、终图、两档网格指标和模型选择指标均与原验证结果一致。本次试验抽取的检查点都完成了这些翼型，5.6 倍属于这批算例的验证环节。[并行验证原始结果](../experiments/results/airfoil_fusion_20261128/performance/prototype.json)

提速入口提供常驻 CPU 验证分片和低内存的同进程轨迹缓存。后者保留已计算的观测和完整结果。每个新权重都重新计算缓存观测上的确定性动作。只要发现某一步动作改变，就按原环境重算该翼型，另存新轨迹。观测数组缓存上限为 32 MiB。正式训练采用该低内存配置，测量时本机物理内存约 15.36 GiB，可用物理内存曾低于 0.4 GiB。

双进程实现还通过了 74 条正式开发验证记录的逐例检验，其中包含成功、预算耗尽和筛选判定失败的记录。E8／种子 42 的整批 32 例首次验证为 276.16 秒，下一份权重全部命中后为 1.13 秒。低内存实现对 E8 的四例复用用时 0.020 秒，策略发生变化的失败例按原环境重新计算。正式入口的集成校验额外核对了模型权重文件字节、环境结果、PyTorch 随机数状态和模型选择指标。[双进程验证](../experiments/results/airfoil_fusion_20261128/performance/production_gate.json) · [低内存缓存](../experiments/results/airfoil_fusion_20261128/performance/compact_cache.json) · [续跑入口校验](../experiments/results/airfoil_fusion_20261128/performance/activation_gate.json)

在线 E8 按翼型缓存完全相同的物理验证请求，保留图的顶点顺序、面顺序与浮点值。用两个种子、四个训练翼型的 24 步真实动作做前后回放。筛选证明、有效动作掩码、每步奖励和观测哈希全部一致。累计墙钟从 119.41 秒降到 99.89 秒，下降约 16.3%。此外，独立测试阶段在数据解封后也使用同一几何缓存，逐例评估仍调用原评估入口。这些耗时来自有限样本。[在线回放对照](../experiments/results/airfoil_fusion_20261128/performance/online_memo.json)

正式队列切换后，E4a／种子 44 的第 2,000 步验证为 32/32，首次用时 48.34 秒。第 2,500 步的 Actor 参数已更新，重新计算动作后复用了 32/32 条轨迹，耗时 0.154 秒。逐例动作、奖励、终图和网格指标一致。E8／种子 44 切换后的第 3,500 步整批冷启动验证为 32/32，用时 458.17 秒。这些是开发集的指定检查点记录。[正式切换记录](../experiments/results/airfoil_fusion_20261128/performance/accelerated_handoff.json) · [正式检查点核验](../experiments/results/airfoil_fusion_20261128/performance/coordination.json)

在正式 E8 的第 3,751–4,000 步在线日志中，种子 42、43、44 的单步中位耗时分别为 3.64、3.65、5.16 秒。在各自这段窗口内，前两个种子分别结束了 57、58 幕，其中成功 56、57 幕。种子 44 结束了 9 幕，均被筛选器判定为 `proven_unfinishable`。各窗口还可能包含跨边界的回合。这些记录表明同一网络下进入不同拓扑状态时，在线耗时也会变化。此处使用的计时范围与上表一致。

E8／种子 44 的第 4,000 步开发验证耗时 4,639.38 秒，用相同的 32 个翼型对比第 3,500 步的 458.17 秒。两次均无轨迹缓存命中。前一检查点每例执行 4 次决策，共 128 次，32 例均完成分块。末轮每例执行 26 次决策，共 832 次，32 例均由筛选器判定为 `proven_unfinishable`。在该末轮验证期间，两次进程采样相隔 70.86 秒，Python 工作进程累计 CPU 时间增加 70.42 秒，相当于一个逻辑核心持续计算。[末轮验证明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed44/validation_004000/report.json) · [进程计时原始记录](../experiments/results/airfoil_fusion_20261128/performance/e8_seed44_final_validation_cpu_sample.json)

E4／种子 45 在第 500 步的开发验证中执行了 1,024 次决策，32 例均未完成，耗时 944.12 秒。第 1,000 步的权重共执行 96 次决策，32 例均完成，耗时 34.99 秒。两次均未命中轨迹缓存。末轮仍为 32/32，重新计算策略动作后复用了全部 32 条轨迹，耗时 0.110 秒。[第 500 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E4_seed45/validation_000500/report.json) · [第 1,000 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E4_seed45/validation_001000/report.json) · [末轮明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E4_seed45/validation_004000/report.json)

E8／种子 45 的开发验证从初始化时的 32/32 降到第 500 步的 0/32。两次都没有轨迹缓存命中，分别耗时 412.01 秒和 5,836.31 秒，决策数由 128 增至 871。两次测试中，32 个翼型的首动作类型均为切分。逐例核对完整描述显示，具体首候选全部改变。第 500 步的轨迹共执行 608 次共享边插点，最终均由筛选器判定为 `proven_unfinishable`。[初始化明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed45/validation_000000/report.json) · [第 500 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed45/validation_000500/report.json)

同一运行第 1,000 步的开发验证恢复至 32/32，两档网格均达到预算。32 个翼型的动作序列与终图逐例匹配初始化结果，决策总数恢复为 128。该检查点未命中轨迹缓存，验证耗时 497.42 秒。[第 1,000 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed45/validation_001000/report.json)

缓存按观测数组的字节数计限，并先淘汰较早的轨迹。第 500 步的 871 次决策仅计边特征、邻接索引和全局特征，就需要 33,996,872 字节，超过 32 MiB 上限。这个下限还未计入候选动作数组。写入这批失败轨迹会淘汰初始化轨迹，因此第 1,000 步的相同动作序列需要重新验证。

从这批归档中选取两个开发翼型重放。原编码存储 63 条观测需要 2,857,779 字节，无损编码需要 231,001 字节。在同一个 2 MiB 测试上限下，原编码需要重算两例，压缩编码重新核对策略动作后复用了两例的初始化轨迹，累计用时 0.014 秒。逐步观测哈希、动作、终图、两档网格和模型选择一致。压缩编码用于此后启动的验证进程，正式进程沿用 32 MiB 缓存上限。[压缩启用凭证](../experiments/results/airfoil_fusion_20261128/validation_cache_compression_handoff.json)

启用后，正式 E8／种子 46 的初始开发验证耗时 419.94 秒。第 500 步使用了新的权重文件，逐例核对动作后复用了全部 32 条轨迹，耗时 0.313 秒。两次均有 32/32 个翼型完成分块，两档网格均符合预算。[初始化明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed46/validation_000000/report.json) · [第 500 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed46/validation_000500/report.json)

同一种子在第 1,000 步的验证累计执行了 768 次决策，32 例均由筛选器判定为 `proven_unfinishable`。这批轨迹全部新算，耗时 4,901.58 秒。第 1,500 步恢复到 32/32，决策总数回到 128。逐例核对动作序列、最终块图和两档网格结果，均与初始化一致。32 条轨迹在长失败批次之后仍可复用，这次验证耗时 0.372 秒。[第 1,000 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed46/validation_001000/report.json) · [第 1,500 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E8_seed46/validation_001500/report.json)

E6／种子 46 在第 1,000、1,500 和 2,000 步验证了同一批 32 个翼型，三轮均从头计算。完成分块且满足两档预算的翼型分别为 4、29 和 0 例，决策总数分别为 912、212 和 1,024 次，验证耗时分别为 1,060.88、172.99 和 1,223.07 秒。第 2,000 步的 32 例均用尽编辑步数。三轮使用相同的验证集和缓存设置，验证用时随轨迹长度显著波动。[第 1,000 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E6_seed46/validation_001000/report.json) · [第 1,500 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E6_seed46/validation_001500/report.json) · [第 2,000 步明细](../experiments/results/airfoil_fusion_20261128/performance/arms/E6_seed46/validation_002000/report.json)

## 独立评估耗时

下表来自锁定后率先完成的四项正式评估。每项使用相同的 90 个翼型。耗时为逐例墙钟之和，范围包括分块以及分块成功后的制网。多个进程同时运行时，各行耗时不能相加作为队列总耗时。

| 方法与种子 | 完成分块 | 决策数 | 几何验证调用数 | 逐例墙钟合计（秒） |
|---|---:|---:|---:|---:|
| E1／42 固定规则 | 90/90 | 360 | 10,260 | 123.00 |
| E2／42 合法动作随机 | 0/90 | 2,880 | 228,447 | 2,831.42 |
| E2／43 合法动作随机 | 0/90 | 2,880 | 228,005 | 3,920.00 |
| E3／42 即时贪心 | 0/90 | 2,880 | 191,162 | 3,420.95 |

随机和贪心的这三个任务均在每例达到 32 次决策后结束。记录的几何验证调用包括候选动作检查。两个随机种子的调用数接近，逐例墙钟仍相差约 18 分钟，因此用这些任务估算总耗时时需要保留这种波动。[E1／42](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E1_42_rule_summary.json) · [E2／42](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E2_42_rule_summary.json) · [E2／43](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E2_43_rule_summary.json) · [E3／42](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E3_42_rule_summary.json)

同一批 90 个翼型上，E7 和 E8 的种子 42 最佳权重均完成 90/90 次分块，各执行 360 次决策，逐例规范化分块相同。E7 的逐例墙钟合计 125.01 秒，E8 为 1,096.83 秒。E8 的筛选计时合计 1,003.27 秒，约占当次评估耗时的 91.5%。筛选阶段发起 112,269 次几何验证调用，该计数包含由几何缓存处理的请求。[E7／42](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E7_42_best_summary.json) · [E8／42](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E8_42_best_summary.json)

E4a／种子 43 的第 1,500 步权重在独立评估中完成 90/90 次分块，决策总数为 360，逐例耗时合计 122.57 秒。第 4,000 步权重在同一批翼型上完成 0/90 次，全部用尽 32 次决策，逐例耗时合计 3,948.01 秒。两次评估的首动作均为切分，但 90 例的具体首切候选全部改变。末轮轨迹累计执行 2,160 次共享边插点。[最佳权重](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E4a_43_best_summary.json) · [末轮权重](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E4a_43_final_summary.json)

E8／种子 44 的末轮正式评估完成 1/90 次分块，另有 89 例被筛选器判定为无法完成。全部算例累计执行 2,318 次决策。逐例墙钟合计 11,003.62 秒，其中筛选耗时 9,100.89 秒。采用并行计算后，原进程读取并核验了 49 条预先计算的算例凭证。从任务日志创建到评估结果写盘，实际经过 5,295.29 秒。逐例合计与任务实际经过时间使用不同的计时口径。[末轮正式记录](../experiments/results/airfoil_fusion_20261128/performance/evaluation/E8_44_final_summary.json) · [评估日志](../experiments/results/airfoil_fusion_20261128/queue_logs/evaluation_E8_44_final_a1.log) · [复算一致性记录](../experiments/results/airfoil_fusion_20261128/evaluation/shard_evidence/E8_44_final/replay_gate.json)

## 与现有基线的口径

- CH-SAC v2 在固定 C-H 拓扑上执行 7 维连续调参，发布配置由多个 CPU 进程计算 TFI 网格，GPU 负责批推理和更新。本轮从可编辑块图出发，需要枚举拓扑动作并检查候选后继图。固定提交的训练汇总 `wall_sec` 是 1,752.28 秒，42 号种子标记为续跑复用，其余四个种子各为约 436–442 秒。种子 42 的全量用时未列出，约 36.3 分钟的五种子总耗时仍需单独的计时记录。[CH-SAC v2 归档计时摘录与哈希](../experiments/results/airfoil_fusion_20261128/performance/ch_sac_wall_time.json)
- MiniMind 当前公开说明的约两小时案例对应单张 RTX 3090 的 SFT 一轮实测，3 元是相应的 GPU 租用费用。MiniMind 的预训练代码用 GPU 批处理和混合精度。翼型任务的同类优化点是给环境采样与验证提供并行 CPU 工作，并优先消除重复几何运算。[MiniMind 口径](https://github.com/jingyaogong/minimind/blob/master/README.md#-项目介绍) · [MiniMind 训练代码](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py)

## 复现

在保留完整正式运行目录的检出中执行。脚本使用冻结训练与开发验证数据、验证阶段权重及交互记录。以下命令以指南中的正式输出目录为例，GitHub 归档提供原测量 JSON 和关键报告。

```powershell
New-Item -ItemType Directory -Force -Path 'experiments/runs/airfoil_fusion_performance' | Out-Null
& .\.venv\Scripts\python.exe -B -u scripts/profile_airfoil_fusion.py --output experiments/runs/airfoil_fusion_performance/profile.json
& .\.venv\Scripts\python.exe -B -u scripts/profile_airfoil_fusion.py --parallel-validation --output experiments/runs/airfoil_fusion_performance/prototype.json
& .\.venv\Scripts\python.exe -B -u scripts/profile_airfoil_fusion.py --compact-cache --output experiments/runs/airfoil_fusion_performance/compact_cache_recheck.json
& .\.venv\Scripts\python.exe -B -u scripts/profile_airfoil_fusion.py --online-memo --output experiments/runs/airfoil_fusion_performance/online_memo_recheck.json
```

原始计时见 [正式日志汇总](../experiments/results/airfoil_fusion_20261128/performance/archived_timing.json)、[单例剖析](../experiments/results/airfoil_fusion_20261128/performance/profile.json)和[CUDA 及并行验证试验](../experiments/results/airfoil_fusion_20261128/performance/prototype.json)。
