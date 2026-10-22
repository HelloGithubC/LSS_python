**float32 能否解释本次 RR 偏差：受控实验与归因边界**

日期：2026-10-21，Asia/Shanghai。本文补充此前的 [RR 全流程报告](/mnt/xiaodong/xiaoliang/software/LSS_python/doc/rr_mpi_review_20261021.md)，保留原报告和两份生产结果不变。

结论：**float32 确实可以使整目录计数和分块计数不一致，并且可以改变总计数；本次已在单进程中复现。更具体的原因涉及 Corrfunc 的 float32 SIMD 分箱路径，而不是 MPI 对整数求和产生舍入。** 但现有实验还不足以把全量结果的 `+1,047,222` 全部归因于这一机制。DS 所说“少数 bin、约 1e-10”也不符合这两份全量文件的实际差异。

本次完成 10 组整目录/分块对照，以及一个独立的跨目录漏计见证。均使用当前机器、单线程、显式指定 ISA，没有启动 MPI，没有提交 Slurm，没有重算 2.4 亿点的完整 RR，也没有修改生产计数实现。

**全量偏差需要解释什么。**

串行基准由用户确认来自 `bin/cal_2pCF_box --only_run_rr`；MPI 结果对应现存成功作业 127826。实际 MPI 配置是 boxsize=2000、ngrids=4、块宽 500、532 个 RR 任务，2 个 rank、每 rank 64 个线程。不能把当前 Slurm 默认 ngrids=10 当作该结果的实际配置。

| 指标 | 两份已保存全量结果的比较 |
| --- | ---: |
| 串行总 npairs | 93,416,884,869,410 |
| MPI 总 npairs | 93,416,885,916,632 |
| MPI − 串行 | +1,047,222 |
| 总计数相对差 | 1.121020039861038e-8 |
| 发生变化的二维 bin | 17,566 / 18,000，97.59% |
| 最大单 bin 绝对差 | 9,550 |
| 最大单 bin 相对差 | 7.989070950939115e-6 |
| 单 bin 绝对相对差中位数 | 3.3465983214175855e-8 |
| 径向求和后正差 / 负差 / 零差 | 142 / 5 / 3 |

分箱元数据、savg、weightavg 完全一致；差异在整数 npairs。最大相对差来自小尺度低计数 bin：250342 与 250344 相差 2。最大绝对差来自 s=[128,129)、mu=[0.7083333,0.7166667)：相差 9550，相对差约 8.52e-7。

因此，不能把“单个点对的计算误差很小”理解为“只有很少的直方图 bin 会变化”。全量有效计数约 9.34e13，边界附近极少量点对的归属变化就可能分散到大量 bin。另一方面，142 个径向 bin 为正差，且总差不为零，说明仅用“相邻 mu bin 之间移动、总数保持”解释并不完整；差异也不限于最外侧 s=150 的边界。

原比较的完整统计、逐 bin 表和径向表分别位于：

- `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/summary.json`
- `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/bin_differences.csv`
- `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/radial_comparison.csv`

**用真实 random 前缀隔离 dtype、ISA 和分块影响。**

实际读取当前 random 的前 100,000 行前三列，保存为 float32 样本；20,000 点实验取它的前缀。所有 float64 实验均把这些完全相同的 float32 数值精确提升到 float64，没有重新生成随机数，也没有恢复原先存盘时丢失的精度。

每组先对整目录调用一次 DDsmu 自相关，再使用实际 `split_catalog_into_blocks` 和 `build_pair_tasks`，计算块内 auto 与跨块 cross，按任务 multiplicity 原样整数合并。无权重、保留原生零距离行为，不加 Jackknife 或其他修正。固定 s=0..150 步长 1、mu=0..1 共 120 格、非周期、boxsize=2000、refinement=(2,2,1)、nthreads=1，显式固定 avx 或 fallback。除明确标注外 ngrids=4、enable_min_sep_opt=True。

| 点数 | dtype | ISA | ngrids | min-sep 优化 | 不同 bin | 最大绝对差 | 分块−整目录总差 | 径向变化 bin |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 20,000 | float32 | avx | 4 | 开 | 4 | 2 | 0 | 0 |
| 20,000 | float64 | avx | 4 | 开 | 0 | 0 | 0 | 0 |
| 20,000 | float32 | fallback | 4 | 开 | 0 | 0 | 0 | 0 |
| 20,000 | float64 | fallback | 4 | 开 | 0 | 0 | 0 | 0 |
| 20,000 | float32 | avx | 4 | 关 | 4 | 2 | 0 | 0 |
| 100,000 | float32 | avx | 4 | 开 | 2,874 | 8 | −12 | 16 |
| 100,000 | float64 | avx | 4 | 开 | 0 | 0 | 0 | 0 |
| 100,000 | float32 | fallback | 4 | 开 | 0 | 0 | 0 | 0 |
| 100,000 | float32 | avx | 3 | 开 | 2,731 | 10 | −16 | 17 |
| 100,000 | float32 | avx | 5 | 开 | 2,806 | 8 | −4 | 19 |

10 组均完成，无超时、无进程错误，10 个 worker stderr 文件均为 0 字节。首组还实际调用了旧 `run_tpCF` 和当前 `count_pair_task`：分别与受控整目录、受控分块结果逐 bin 完全一致。其余组用受控 DDsmu 调用，以便显式改变 ISA；没有声称它们逐一走过 MPI 入口。

这些结果说明：不经过 MPI 也能复现；网格改变会改变 float32 AVX 偏差；提升到 float64 或切换当前 fallback 路径后，这些样本的整目录/分块结果一致。20,000 点的 min-sep 开关不改变结果，不能据此排除该优化在所有其他输入中的影响。

还需要区分“整目录与分块一致”和“不同精度的结果一致”。100,000 点的 float32 fallback 与 float64 AVX 整目录之间仍有 78 个 bin 不同，虽然总数相同；float32 AVX 与 float64 AVX 则有 3,791 个 bin 不同、总差 −28。因此 fallback 并不消除所有 float32 几何舍入；把数据转成 float64 后，也不能期待必然复现旧 float32 基准的每一个 bin。

完整 10 组用 [all_cases.json](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/all_cases.json) 和 [case_summary.csv](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/case_summary.csv) 汇总。`summary.json` 仅包含最初的 8 组；另外两组 grid3/grid5 另行运行，已经纳入上述 10 组汇总。

**一个合法点对如何因 float32 SIMD 路径被漏掉。**

构造 P=(100,100,100)，Q=float32(100.05,100.01,200)。用存盘后的确切 float32 坐标提升至 float64 计算，距离约 100.0000130017，mu 约 0.99999986998，小于 1，位于有效范围内。先计算 P 与仅含 Q 的目录互相关，再往 Q 的目录加入 7 个与 P 距离大于 150 的点；这些新增点本身不能贡献任何点对。

| 几何精度 / ISA | 只有 Q 时计数 | 加入 7 个范围外点后计数 |
| --- | ---: | ---: |
| float32 / avx | 1 | **0** |
| float32 / fallback | 1 | 1 |
| float64 / avx | 1 | 1 |
| float64 / fallback | 1 | 1 |

输入的有效点对没有变化，float32 AVX 却将它漏掉。这已经证明问题可以影响返回直方图的总计数，而不只是让一个 bin 减一、相邻 bin 加一。脚本和原始输入、8 组完整直方图均已保存。

官方 Corrfunc 2.5.3 [AVX 分箱源码](https://github.com/manodeep/Corrfunc/blob/2.5.3/theory/DDsmu/countpairs_s_mu_kernels.c.src) 在约 570–586 行先用浮点数计算展平索引，再转整数；标量尾部约 606–641 行先将 mu 分箱转整数，再与径向整数偏移相加。对应本例：

```text
float32 mu * 120 = 119.9999771118164
internal radial offset = 101 * (120 + 1) = 12221

scalar: 12221 + int(119.999977...) = 12340 -> internal mu index 119
SIMD:   int(float32(12221 + 119.999977...)) = 12341 -> internal mu index 120
```

浮点加法吞掉了很小的小数差。内部每个径向 bin 为 mu 预留 121 格，但输出只返回索引 0..119；[结果打包源码](https://github.com/manodeep/Corrfunc/blob/2.5.3/theory/DDsmu/countpairs_s_mu_impl.c.src) 约 683–695 行的循环条件是 `j < nmu_bins`。本例被推入内部 mu=120 的点对不在返回结果内。普通 mu 边界也会受类似的先相加再截断影响，产生相邻格迁移。

加入范围外点会改变单次内核中可组成的向量批次；整目录与分块也会改变目录组织、候选点批次和向量尾部，因此即使所有原始坐标保持不变，也可能让同一对点经历不同的运算路径。这说明“都调用 Corrfunc”并不足以保证 float32 下计数严格不变。

这里的公式、实际二进制实验和 2.5.3 上游源码相互吻合；没有对已安装二进制做逐指令跟踪，也没有验证其构建源码与上游 tag 完全相同。持久保存的源码副本和二进制哈希见文件清单。

另做过两组 9 点 RR auto 对照，boxsize=500、ngrids=2，四种 dtype/ISA 模式下整目录和分块均为 11，未发现差异。它们是阴性结果，不能写成“最小 RR 例已经复现正偏差”。其参数与观测保存在 `negative_controls.json`；当时为会话内联执行，没有独立脚本、原始数组文件或独立 stdout/stderr 日志。

**对全量结果应如何定性。**

已经证明的是：float32 + Corrfunc SIMD 路径能够造成所关心的这类偏差，包含真实 random 的分块差异和独立的总计数漏失。当前代码对 npairs 使用 uint64 合并和 MPI SUM，计数远低于上限；不存在把这些整数当作浮点数累加后舍入的过程。无权重结果也不能归因于全零权重兼容。

尚未证明的是：上述机制是否足以解释全量 `+1,047,222` 的数值、符号和分布。100,000 点实验的分块总差是 −12，全量是正差；不能按点数或点对数简单线性外推。全量结果在 142 个径向 bin 中为正，还需要确认历史串行作业与 cn4/cn5 的实际 ISA、Corrfunc 二进制、输入身份，以及真实密度下各路径的占比。当前本机为 xiaodong-GPU01，支持 AVX、无 AVX512F；没有远程核实生产节点和历史构建。原 random 未做全文件哈希，历史串行环境也未恢复。

下一步若继续归因，应先在同一生产节点和同一二进制上固定 ISA，对同一批较密的坐标分别做整目录和分块的 float32/float64 对照，再按径向与 mu 分析差值。本次没有启动该额外生产环境验证。提升 dtype 的验证应同时提升两侧，不能只将 MPI 端改为 float64，再要求它逐 bin 等于旧 float32 结果。

`np.allclose` 只是选定容差后的比较方式，不能代替上述归因。当前全量结果在 atol=0 时，rtol=1e-10 到 1e-6 均不能通过，rtol=1e-5 才全部通过；这不自动等于统计上可接受，也不是生产框架正确性的证明。

**实际运行环境、脚本与复现方式。**

新产物目录固定为 `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/`。环境记录在该目录的 `environment.json`：Python 3.10.0，NumPy 2.2.6，Corrfunc 2.5.3，joblib 1.4.2，当前可编辑安装的 LSS_python。其已安装元数据版本是 0.5.0，不能据此替代仓库源码身份。

`probe_float32.py` 的目的为隔离精度、ISA、分块和 min-sep 影响。父进程 mmap 输入、保存前缀、生成 case 参数、逐个启动 worker；worker 计整目录、生成实际分块任务、整数合并，再保存原始直方图和指标。每个父进程 worker 超时默认 90 秒，参数 `--sizes` 默认 20000 100000；核心分箱、boxsize、线程数、refinement 为脚本中固定的本次实验参数。最初 8 组的实际命令为：

```sh
/home/xiaoliang/.conda/envs/xiaoliang/bin/python \
  /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/probe_float32.py \
  --catalog /mnt/xiaodong/xiaoliang/JT-2G/RR/random_box2000_3e-2.npy \
  --output-dir /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo
```

grid3/grid5 的参数文件另行生成，分别用同一脚本的内部 `--case-file` 模式执行，外层 subprocess 也设置 90 秒超时并保存日志。例如 grid3 的 worker 命令为：

```sh
/home/xiaoliang/.conda/envs/xiaoliang/bin/python \
  /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/probe_float32.py \
  --case-file /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/n100000_float32_avx_grid3_minsep1.case.json \
  --output-dir /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo
```

grid5 同理，将上述参数文件名中的 grid3 换为 grid5。独立调用 worker 自身不提供超时或日志重定向；本次外层执行器提供了这两项控制。父进程模式依赖 Python、NumPy、Corrfunc、本地 LSS_python；首次兼容验证还导入旧 tpcf 及其已有依赖，包括 joblib、SciPy、Numba 和 scikit-learn。无 MPI 或 Slurm 依赖。

`mu_edge_probe.py` 用脚本内的合成坐标生成上述 8 种跨目录结果，依赖 Python、NumPy、Corrfunc，不读取生产 random。实际命令为：

```sh
/home/xiaoliang/.conda/envs/xiaoliang/bin/python \
  /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/mu_edge_probe.py \
  --output-dir /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo
```

两个脚本无参数均打印帮助。上面的命令记录已经完成的执行，重跑须使用新的未占用输出目录：父进程拒绝已有 sample/case，最小见证拒绝已有 probe 输出。不要直接在本次目录重复调用 worker；其 NPZ 写入早于 result.json 的排他创建，重复调用存在先覆盖 NPZ 再报错的可能。

**完整文件交付清单、日志和临时文件。**

[file_inventory.csv](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/file_inventory.csv) 逐项给出绝对路径或动态命名规则、来源、用途/格式、是否实际存在/生成、覆盖与清理行为。它包含本次全部文件、实际读取的既有结果/比较材料/相关依赖，以及不能给出固定名字的临时和缓存规则；不能把其中的动态路径视为已核实的遗留文件。

| 产物或规则 | 来源和用途 | 本次状态 |
| --- | --- | --- |
| 本文绝对路径 `/mnt/xiaodong/xiaoliang/software/LSS_python/doc/rr_float32_analysis_20261021.md` | 新的分析报告 | 已生成 |
| 产物目录下 `probe_float32.py`、`mu_edge_probe.py` | 实际使用的两个实验脚本 | 已生成并运行 |
| `sample_float32.npy` | 原 random 前 100,000 行的 XYZ；shape=(100000,3)，float32，1,200,128 字节 | 已生成；作为全部前缀实验的派生输入 |
| `<case>.case.json`、`<case>.result.json`、`<case>.npz` | 每组参数、指标、whole/blocked 原始 uint64 直方图及分箱 | 各 10 个，均已生成；case 名逐项见清单 |
| `<case>.stdout.log`、`<case>.stderr.log` | 各 worker 的进度/结果输出及错误流 | 各 10 个，已生成；全部 stderr 为空 |
| `summary.json` | 父进程随完成进度更新的最初 8 组结果 | 已生成；运行中只更新自身文件 |
| `all_cases.json`、`case_summary.csv` | 后处理汇总完整 10 组 | 已生成 |
| `cross_case_metrics.json`、`full_pattern.json` | 跨精度/ISA 比较和既有全量径向符号分析 | 已生成 |
| `mu_edge_probe.json`、`mu_edge_probe.csv`、`mu_edge_probe.npz` | 最小见证结果、表格、输入坐标与 8 组原始直方图 | 已生成 |
| `negative_controls.json` | 会话内两组未发现偏差的 RR 实验记录 | 已生成；没有独立原始 NPZ/脚本/日志 |
| `environment.json` | 当前环境、native 二进制和派生样本哈希 | 已生成 |
| `verification.json` | 产物一致性、原输入哈希和脚本帮助检查 | 已生成，见最终核验记录 |
| 父进程、最小见证、内联后处理/阴性对照的 stdout/stderr | 保存在当前会话工具输出 | 未生成独立日志文件 |
| `/tmp/tmp<随机后缀>` | 当前 Corrfunc 数组分箱产生的临时文本 binfile，正常成功返回后删除 | 调用过程使用；具体名字未记录，未盘点或清理其他 `/tmp` 文件 |
| `<模块目录>/__pycache__/<模块名>.cpython-310.pyc` | Python 导入可能复用/更新的字节码缓存 | 未逐项核实是否由本次新建/更新；清单列出可确定路径规则；未清理 |
| MPI/Slurm 日志、checkpoint、额外生产 RR | 本次未启动此类流程 | 未生成 |

两个生产 PKL、原 random、原全流程报告及其比较文件均保留。本文和附属材料只写入仓库内的新路径；没有清理或覆盖既有生产结果。后处理 JSON/CSV 和本文在整理时可更新本次新建文件；最终内容以保存文件和核验记录为准。
