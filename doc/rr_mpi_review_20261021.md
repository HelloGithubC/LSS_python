**RR 串行基准与 MPI 分块结果：全流程及独立审阅材料**

报告日期：2026-10-21，Asia/Shanghai。供用户转交 Claude 审阅。

审阅目标是检查 MPI 分块实现能否复现既有整目录 Corrfunc 计数。用户已确认：没有 `_mpi` 后缀的文件由 `bin/cal_2pCF_box` 的 `--only_run_rr` 模式产生，作为本次回归基准；有 `_mpi` 后缀的是待验证结果。仓库实际目录名是 **`bin/`**，不是 `bins/`。

已确认的全量比较结果：两份文件的分箱、dtype、`savg` 和 `weightavg` 完全一致，只有 `npairs` 不同；18,000 个 bin 中 17,566 个不同。MPI 总计数相对串行多 `1.121020039861038e-08`，最大单 bin 相对差 `7.989070950939115e-06`。这些差异尚未完整归因，不能把默认 `np.allclose` 通过当作实现等价的证明。

本报告补充核实了成功 MPI 作业：**127826，实际 `ngrids=4`，不是当前提交脚本的默认值 10**。证据来自现存日志，而非从文件名推断。报告编写期间仅读取结果、源代码、运行环境信息和日志，生成审阅材料；没有重新计算全量 RR、提交作业、修改计算实现或覆盖原始结果。

**比较对象与输入来源。**

| 对象 | 绝对路径 | 来源、内容、核实状态 |
| --- | --- | --- |
| 串行基准 | `/mnt/xiaodong/xiaoliang/JT-2G/RR/RR-random_box2000_3e-2-150s0to150_120mu.pkl` | 用户确认由原脚本 RR-only 模式生成；已实际读取并计算哈希。原始作业号、命令、线程数、节点及当时依赖版本未找到。 |
| MPI 结果 | `/mnt/xiaodong/xiaoliang/JT-2G/RR/RR-random_box2000_3e-2-150s0to150_120mu_mpi.pkl` | 已实际读取；作业 127826 日志记录该路径从“cache missing”到保存成功。 |
| 随机目录 | `/mnt/xiaodong/xiaoliang/JT-2G/RR/random_box2000_3e-2.npy` | 当前文件头为 `(240000000, 4)`、`float32`，3,840,000,128 字节。仅用 mmap 检查文件头和前两行；没有全量扫描、计算全文件哈希或核实生成算法/种子。MPI 日志也记录该 shape/dtype。 |
| Data 占位参数 | `/mnt/xiaodong/xiaoliang/JT-2G/src/__RR_ONLY_PLACEHOLDER__.npy` | 按当前 MPI 提交目录推导的占位路径；RR-only 不读取它。本报告没有创建该文件，也没有检查是否原已存在。 |

当前 random 前两行的第 4 列都是 1.0；不能据此声称整个文件的权重列都已扫描确认。本次 MPI 作业明确 `with_weight=False`，因此该列不参与计数。历史串行作业的完整参数未存档；以下串行流程说明针对与本结果相对应的无权重 RR-only 路径。

两份结果都是 joblib 序列化的 NumPy 结构化数组，磁盘大小各 864,401 字节，长度 18,000，可重排为 `(150, 120)`。字段顺序为：`smin:f8, smax:f8, savg:f8, mu_max:f8, npairs:u8, weightavg:f8`。`s` 分箱为 0 到 150、步长 1；`mu` 分箱为 0 到 1、120 格。结果数组中的 `float64` 元数据不表示点对几何计算使用了 float64；计算精度取决于传入坐标的 dtype。

| 文件 | 当前 mtime（UTC+8） | SHA-256 |
| --- | --- | --- |
| 串行 RR | 2026-09-30 18:28:28.574977 | `6ac6e5c6bf8bb25bcef2533dca855401e2fe99d75f44d9bcb7b76075816fd946` |
| MPI RR | 2026-10-21 14:04:57.162394 | `5da415723120636b3461cc2555aea5b11d3fe609f17389950b338d519e35d264` |

mtime 是当前文件系统元数据，不替代作业记录；两个哈希在本次报告编写时重新读取文件核对，和之前比较报告一致。

**串行基准的生成流程。**

入口脚本：[cal_2pCF_box](/mnt/xiaodong/xiaoliang/software/LSS_python/bin/cal_2pCF_box:1)。核心调用：[tpcf.py 的 run_tpCF](/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/tpcf.py:693)，RR 分支在同文件第 759 行附近。

```text
bin/cal_2pCF_box --only_run_rr
  -> read_data(random_filename, weight_set_one=not with_weight)
  -> LSS_python.tpcf.run_tpCF(None, random_catalog, ..., run_parts=["RR"])
  -> Corrfunc.theory.DDsmu(autocorr=True, periodic=False, ...)
  -> Corrfunc._countpairs.countpairs_s_mu (C extension)
  -> joblib.dump(RR_result, output_RR)
```

处理步骤如下：

1. 解析数据占位名、random 文件名、`nthreads` 及分箱选项；无参数显示帮助。
2. RR-only 跳过 Data 读取。若 RR 输出已存在且未指定 `--force_rr/-forcerr`，直接退出。
3. 无权重模式下，旧脚本将 random 读成同 dtype 的 `(N,4)` 数组：复制前三列，第四列设成 1。后续 `with_weight=False` 的 RR 分支只向 Corrfunc 传入 X/Y/Z，不传 weights。
4. `sedges=np.linspace(0,150,151)`，调用一次完整随机目录的 `DDsmu` 自相关：`mu_max=1.0, nmu_bins=120, periodic=False`，默认 refinement 为 `(2,2,1)`。`nthreads` 是该单 Python 进程内的 OpenMP 线程数；“单进程”不意味着“单线程”。
5. 把 Corrfunc 返回的原始结构化数组直接保存。RR-only 不调用 `cal_tpCF_from_pairs`，不做归一化、Landy–Szalay、Jackknife 修正或自配对扣除。

因此该文件适合作为要复现的 Corrfunc 回归行为。以上是对当前源码和用户提供来源的联合追踪，不能冒充找到了 9 月 30 日的完整执行命令或当时的源码快照。

**MPI 结果的生成流程与成功作业证据。**

提交入口是当前 [run_RR_mpi.slurm](/mnt/xiaodong/xiaoliang/JT-2G/src/run_RR_mpi.slurm:1)，执行 [cal_2pCF_box_mpi](/mnt/xiaodong/xiaoliang/software/LSS_python/bin/cal_2pCF_box_mpi:199)。实际成功运行参数优先采用 [作业 127826 stdout](/mnt/xiaodong/xiaoliang/JT-2G/src/run_RR_mpi_127826.log:1)。

| 配置或阶段 | 作业 127826 已记录的内容 | 证据位置 |
| --- | --- | --- |
| 模式、分箱 | sys / RR-only，150 个 s bin，120 个 mu bin，s=[0,150] | stdout 第 1–17 行 |
| rank / OpenMP | 2 个 MPI rank，每 rank 64 线程 | stdout 第 7–8、29 行 |
| 空间网格 | `[4,4,4]`，64 块，每边宽 500，boxsize=2000 | stdout 第 9–11、32 行 |
| 权重、refinement | with_weight=False，refinement=[2,2,1] | stdout 第 14–16 行 |
| 是否真的计算 | 输出显示 cache missing，随后加载完整 random 并计算 | stdout 第 23–29 行 |
| 块对任务 | 64 个内部任务、468 个跨块任务，共 532；每 rank 266 | stdout 第 34–40 行 |
| 节点与数据块 | rank 0 在 cn4，收到 64 个 random 块；rank 1 在 cn5，收到 58 块 | stdout 第 38–45 行 |
| 原始数据载荷 | rank 0 为 2.682 GiB，rank 1 为 2.431 GiB | stdout 第 38–39 行 |
| 完成、保存 | 两个 rank 均完成；Reduce 后保存目标 `_mpi.pkl`；总耗时 6180.97 s | stdout 第 223–235 行 |
| stderr | `/mnt/xiaodong/xiaoliang/JT-2G/src/run_RR_mpi_127826.err`，当前 0 字节 | 已检查文件大小 |

日志中的目标路径、缓存不存在、完整计算结束、保存成功及文件 mtime 相互支持，构成作业 127826 生成该结果的强证据。本次没有查询 `sacct`，没有取得 Slurm 保存的原始提交脚本或结束时的结果哈希。

当前 Slurm 脚本默认 `ngrids=10`，支持第二个位置参数覆盖。不能用当前默认值描述已完成结果。当前脚本申请两节点、每节点一 rank、每 rank 64 CPU、每 CPU 500M 内存；这些是当前脚本中的申请值，实际分配量未另查调度器。它设置 `ulimit -s 102400` 和 `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`，使用 `mpiexec -n $SLURM_NTASKS`。没有记录当时解释器、包版本、编译参数或最终选择的 ISA。

调用链为：

```text
run_RR_mpi.slurm
  -> mpiexec -n 2 bin/cal_2pCF_box_mpi ... --only_run_rr
  -> rank 0: prepare_run / read_data
  -> LSS_python.tpcf_mpi.run_tpCF_mpi
       -> root: _prepare_work
            -> prepare_catalogs
            -> split_catalog_into_blocks
            -> build_pair_tasks
            -> _assign_tasks
       -> scatter task lists and required catalog blocks
       -> each rank: count_pair_task -> count_pairs
            -> LSS_python.corrfunc.call_DDsmu
            -> Corrfunc.theory.DDsmu(auto or cross)
            -> Corrfunc._countpairs.countpairs_s_mu
       -> each rank: sum_pair_counts -> merge_pair_counts
       -> MPI.Reduce(uint64 npairs, SUM)
       -> MPI.Reduce(float64 wpairs, SUM)
       -> root: _corrfunc_result -> joblib.dump
```

MPI 流程的具体职责如下：

1. 只有 root 读文件、决定缓存是否跳过和保存最终 RR；其余 rank 的目录参数可为 None。无权重 random 在新 CLI 中变为 `(N,3)` 的 float32 数组。与旧脚本 `(N,4)` 的前三列相比，坐标数值和精度保持，数组布局/步长可能不同。
2. root 使用同一套非周期网格，按 `floor(position/block_width)` 分块；保留原始全局坐标、不平移坐标、不做周期环绕。上边界点归入最后一块，空块保留。
3. 显式检查 `smax <= min(boxsize/ngrids)`；本次是 `150 <= 500`。只生成块内及三轴索引差各不超过 1 的邻块任务。
4. RR 块内任务 `autocorr=True`；跨块仅生成 `i<j` 的 `autocorr=False` 任务，合并时乘 2。乘 2 由 `PairTask.multiplicity` 给出，用于恢复整目录自相关的计数方向，不是统计修正。
5. 以首块编号分组，估算候选点对工作量，按组成本降序贪心分给累计成本最小的 rank。调度静态；root 也计算。每个 rank 只接收其任务用到的块，但块可在多个 rank 上复制。本次 rank 0 实际需要全部 64 块，因此本作业的数据分发没有消除 root 持有全目录的需求。
6. 同一 rank 按顺序执行任务，每次调用 Corrfunc 用 64 个 OpenMP 线程。任务返回 raw `PairCounts`，局部合并检查重复任务、分箱和权重模式，然后应用 multiplicity。
7. MPI 对 `npairs` 使用 uint64 SUM，整数计数不经浮点类型归并。`wpairs` 另以 float64 SUM 归并；当前无权重输出不使用它恢复 `weightavg`，`weightavg` 保持为 0。
8. root 构造与默认 Corrfunc 字段一致的数组，保存 joblib。MPI 路径不调用 `finalize_pair_counts`、`pair_normalizations`、`cal_tpCF_from_pairs` 或 Jackknife 流程。零距离自配对保留。

当前源码的主要定位：

| 文件及绝对路径 | 关键入口 |
| --- | --- |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/tpcf_mpi.py` | `_assign_tasks`:51；`_count_local_tasks`:93；`_prepare_work`:127；`_corrfunc_result`:220；`run_tpCF_mpi`:241；两次 Reduce:325–326 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/pair_counting.py` | `PairTask`:72；`count_pairs`:100；`merge_pair_counts`:143；`prepare_catalogs`:214；分块:259；任务生成:278；单任务:308；任务归并:321 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/corrfunc.py` | `call_DDsmu`:4，单次 native 调用参数 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/bin/cal_2pCF_box_mpi` | `read_data`:112；`prepare_run`:135；`main`:199 |

这些定位对应本报告检查时的源码；Git HEAD 为 `f868d249572a106cbf24738f894f4b8a4189817b`，分支 main，开始检查时工作区干净。文件哈希另存于环境清单，以防后续行号或内容变化。没有证明历史串行作业使用了同一 Git 版本。

**两条路径的关键 Corrfunc 参数。**

| 参数 | 串行基准路径 | MPI 每个子任务 |
| --- | --- | --- |
| 目录范围 | 整个 random 一次 | 内部块或相邻块对 |
| autocorr | True | 块内 True，跨块 False |
| 坐标 dtype | 无权重读取保留源 dtype，当前 random 为 float32 | 保持 float32 |
| periodic | False | False |
| boxsize 参数 | 原 RR 调用未显式传入 | 显式传入总体盒子尺寸 |
| mu_max / nmu_bins | 1.0 / 120 | 1.0 / 120 |
| s bin edges | `np.linspace(0,150,151)` | 相同 |
| refinement | 默认 (2,2,1)，历史命令是否覆盖未记录 | 成功日志确认 (2,2,1) |
| nthreads | 历史值未知 | 64 / rank |
| weights | 无权重分支不传 | 本次不传 |
| output_savg | 使用 DDsmu 默认 False | 使用默认 False |
| isa | 使用 DDsmu 默认 fastest | 使用默认 fastest |
| fast_divide_and_NR_steps | 默认 0 | 默认 0 |
| copy_particles | 默认 True | 默认 True |
| enable_min_sep_opt | 默认 True | 默认 True |
| max_cells_per_dim | 默认 100 | 默认 100 |

参数默认值来自当前安装的 DDsmu Python 包装器；不能倒推出旧作业当时安装版本。`periodic=False` 下，上游 2.5.3 C 实现将三轴 wrap 置为 0，见参考 `countpairs_s_mu_impl.c.src` 第 271–278 行；显式 boxsize 不应在该路径引入周期环绕。这里仍建议审阅者核对实际二进制/源码对应关系。

**依赖位置与当前检查环境。**

当前检查主机是 xiaodong-GPU01，Intel Xeon E5-2699 v4，具备 AVX/AVX2/FMA，未列出 AVX512F。**这是检查主机，不能替代 cn4/cn5 或历史串行节点的 CPU/ISA 信息。**没有远程登录 cn4/cn5 查询环境，也没有初始化新的 MPI 计算。

| 依赖 | 当前版本 | 当前实际路径与用途 |
| --- | --- | --- |
| Python | 3.10.0 | `/home/xiaoliang/.conda/envs/xiaoliang/bin/python`，解释器 |
| LSS_python | metadata 为 0.5.0；pyproject 为 0.6.0 | editable 安装指向 `/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/`；以真实模块路径和文件哈希为准 |
| NumPy | 2.2.6 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/numpy/__init__.py`，目录和直方图 |
| joblib | 1.4.2 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/joblib/__init__.py`，读写 pkl |
| Corrfunc Python | 2.5.3 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/Corrfunc/theory/DDsmu.py`，DDsmu 包装器 |
| Corrfunc helpers | 2.5.3 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/Corrfunc/utils.py`，临时分箱文件等 |
| future | 1.0.0 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/future/__init__.py`，Corrfunc 声明的 Python 兼容依赖 |
| wurlitzer | 3.1.1 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/wurlitzer.py`，Corrfunc 的 C stdout/stderr 管道重定向辅助 |
| Corrfunc native | 2.5.3 package | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/Corrfunc/_countpairs.cpython-310-x86_64-linux-gnu.so`，真正执行计数的扩展 |
| mpi4py Python | 4.0.3 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/mpi4py/__init__.py` |
| mpi4py native | 4.0.3 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/mpi4py/MPI.cpython-310-x86_64-linux-gnu.so` |
| MPI launcher | MPICH/Hydra 4.3.0 | `/software/mpich4.3-gnu12/bin/mpiexec`；`mpirun` 同目录 |
| MPI library | MPICH installation | `/software/mpich4.3-gnu12/lib/libmpi.so.12`，当前 mpi4py 的 ldd 依赖 |
| OpenMP runtime | 版本未单独查询 | `/lib/x86_64-linux-gnu/libgomp.so.1`，当前 Corrfunc 的 ldd 依赖 |
| GSL | soname 27 | `/lib/x86_64-linux-gnu/libgsl.so.27`，当前 Corrfunc 的 ldd 依赖 |
| GSL CBLAS | soname 0 | `/lib/x86_64-linux-gnu/libgslcblas.so.0`，当前 Corrfunc 的间接链接依赖 |
| SciPy | 1.15.2 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/scipy/__init__.py`，tpcf/base/compressor 导入依赖 |
| numba | 0.66.0 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/numba/__init__.py`，base 导入依赖 |
| scikit-learn | 1.7.2 | `/home/xiaoliang/.conda/envs/xiaoliang/lib/python3.10/site-packages/sklearn/__init__.py`，compressor 导入依赖 |

声明依赖的位置是 `/mnt/xiaodong/xiaoliang/software/LSS_python/pyproject.toml`。它声明 NumPy、numba、joblib、SciPy、tqdm、freud-analysis、emcee、h5py、multiprocess、Corrfunc，MPI 为可选依赖 `mpi=["mpi4py"]`。当前串行 CLI 导入整个 tpcf.py，后者还导入 `/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/base.py` 和 `/mnt/xiaodong/xiaoliang/software/LSS_python/src/LSS_python/compressor.py`，因此即使只计 RR，导入环境也涉及 SciPy、numba、sklearn。当前 pyproject 未显式列出 scikit-learn；本报告只记录现状，不修改依赖声明。

其他声明依赖的当前版本和逐项实际模块绝对路径，也已记录在 [environment_manifest.json](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/environment_manifest.json) 的 `modules` 中。该文件同时记录包 metadata 目录、editable 安装指向、扩展二进制 SHA-256、ldd、MPI launcher 版本、源码/测试/日志哈希。

当前检查会话 PATH 能解析两个 CLI 到本仓库的 `bin/`，但不能解析 `sbatch`。没有在此环境安装、升级依赖。此前生产作业启动时是否使用同一个 Python/包安装、所有 rank 是否加载相同二进制，以及它们选择了哪个 ISA，仍待核实。

**实际比较结果与数学口径。**

逐 bin 定义 `delta = npairs_MPI - npairs_serial`，相对差分母是对应串行 bin；统计程序先转成 Python 整数，避免 uint64 相减下溢和求和溢出。两份文件没有零计数 bin。

| 指标 | 已实际计算的数值 |
| --- | ---: |
| 完全相同的 bin | 434 / 18,000 |
| 不同 bin | 17,566 / 18,000，97.5888889% |
| delta>0 / delta<0 的 bin | 9,695 / 7,871 |
| 串行总计数 | 93,416,884,869,410 |
| MPI 总计数 | 93,416,885,916,632 |
| 总差值 | +1,047,222 |
| 总计数相对差 | 1.121020039861038e-08 |
| 单 bin 最大绝对差 | 9,550 |
| 单 bin 最大相对差 | 7.989070950939115e-06 |
| 全部 bin 的绝对相对差 p50 / p90 / p99 | 3.3465983214e-08 / 1.3986631569e-07 / 6.5419423212e-07 |
| 所有 bin 的绝对差之和 | 3,479,778 |
| 按 mu 求和后仍不同的 s bin | 147 / 150 |

最大绝对差的 bin 为 `s=[128,129), mu=[0.7083333333333334,0.7166666666666667)`：串行 `11,205,992,070`，MPI `11,206,001,620`，差 `+9,550`，相对 `8.522226269967368e-07`。最大相对差出现在 `s=[0,1), mu=[0.675,0.6833333333333333)`：`250,342 -> 250,344`，相差 2。

`np.array_equal` 不通过。对所有 bin 使用 `atol=0`，`rtol` 从 1e-10 至 1e-6 都不全部通过，1e-5 才全部通过。因此之前提到的“约 1e-10”不能作为这份全量结果的实际误差概括。总计数和大多数径向汇总也不同，不能只用“相邻 mu bin 搬移且计数守恒”解释整个现象。

比较产物已经真实生成并检查过 CSV 行数及整数差求和一致性：

| 绝对路径 | 用途、格式、状态 |
| --- | --- |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/compare_rr.py` | 已创建、已执行的比较程序；Python/NumPy/joblib；无参数显示帮助 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/summary.json` | 已生成；统计、输入哈希、误差分位数、最大差异位置 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/bin_differences.csv` | 已生成；17,566 行差异 bin，包含索引、区间、两侧计数、signed/relative delta |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/radial_comparison.csv` | 已生成；150 行按 mu 累加的径向统计 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/comparison.md` | 已生成；前次只读比较报告。其“未检查生产日志”的描述针对当时范围，本报告补查了日志。 |

**已有数值线索与不能提前下的结论。**

此前会话用当前 random 前 20,000 个点、`ngrids=3, sbin=150, mubin=120` 做了整目录/分块串行对照：float32 有 12 个 bin 不同、最大差 2、总差 0；同一批 float32 坐标转 float64 后没有差异。这是先前工具执行输出中的观测，没有独立保存该次样本、完整运行日志和可复现脚本，不能把它当作当前全量文件的归因证明；本报告没有重跑这一实验。

曾额外尝试 750×600 小样本探测：首次进程退出码 139，尚未定位根因；调整栈限制后的一次探测耗时较长，被主动终止，没有有效比较结果。因此不能声称该追加探测通过。该配置也不是本文比较的两份 150×120 文件。

上游 Corrfunc 2.5.3 的 DDsmu C 模板显示：SIMD 路径先以浮点合成展平索引再取整，而标量尾部先对 mu-bin 取整再用整数合成索引。在有限精度下，这两种写法可能将边界附近点对归入相邻 mu bin；分块改变 cell-pair 及循环尾部长度，因而可能改变部分点对所走路径。AVX512 的距离计算还存在 fused multiply-add 路径，与标量算术顺序不同。是否在生产节点上实际选用该 kernel、是否足以解释全量净计数差，尚未验证。

上游参考源码已从前次下载的 `/tmp` 文件复制到持久目录：

| 本地绝对路径 | 上游来源和审阅位置 |
| --- | --- |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_kernels.c.src` | [官方 2.5.3 tag](https://github.com/manodeep/Corrfunc/blob/2.5.3/theory/DDsmu/countpairs_s_mu_kernels.c.src)；AVX512 约 213–277 行、AVX 约 570–586 行、标量尾部约 606–641 行 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_impl.c.src` | [官方 2.5.3 tag](https://github.com/manodeep/Corrfunc/blob/2.5.3/theory/DDsmu/countpairs_s_mu_impl.c.src)；非周期 wrap 约 271–278 行、调用 kernel 约 466–510 行、自相关倍数及自配对约 610–650 行 |

这些文件是**上游 tag 源码参考**，不是已验证的本机二进制构建源码，也不是生产节点源码快照。不要根据安装包版本号相同，就断定所有编译参数和实际二进制相同。

当前 MPI `npairs` 的局部合并和 Reduce 都使用 uint64，当前计数远低于 uint64 上限。浮点求和顺序不能解释整数 Reduce 本身产生舍入；仍应检查任务覆盖/倍数/输入身份以及 Corrfunc 分箱和裁剪路径。本报告不预设所有差异都属于可接受误差，也不预设错误一定来自 MPI 任务组织。

**建议 Claude 重点核查的问题。**

1. 对相同输入坐标、相同 ISA 与 Corrfunc 二进制，`整目录 auto` 和 `块内 auto + 跨块 cross×2` 的差异能否在单进程复现？先把分块影响和 MPI 影响分开。
2. 固定同一套块对任务时，单进程逐任务求和与 MPI 多 rank 的 uint64 归并是否完全一致？是否确实每个任务只出现一次，跨块倍数只应用一次？
3. 当前 `smax=150、block_width=500` 下，邻区覆盖是否完整；float32 坐标在网格边界的归属是否稳定；所有块是否保持全局坐标？
4. SIMD 展平索引取整、距离 FMA/算术顺序、cell 最小距离裁剪和 `mu_max=1`/`smax=150` 的包含排除边界，哪些会导致径向和总计数不守恒？不要只看 mu-bin 间迁移。
5. 在受控小样本上固定 `isa='fallback'` 或同一支持的 ISA，比较 float32/float64、ngrid=1/4/10、`enable_min_sep_opt` 开关，逐项改变因素，建立能解释全量量级的证据。
6. 两份结果是否确实来自内容相同的 random 文件、相同 Corrfunc 构建和相同计算精度？目前没有历史输入哈希和两侧作业环境清单。
7. 生产基准原始线程数/节点/ISA 未知，是否需要补充原始作业记录？当前 filename 不包含 ngrids、ISA、依赖版本，不能仅凭文件名恢复这些条件。

这些是后续审阅方向，**本报告编写期间没有执行上述重算或实验**。当前 Python `count_pairs`/`run_tpCF_mpi` 接口没有暴露 ISA 或最小距离优化开关；测试这些选项应使用受控的直接 DDsmu 调用或独立验证代码，不要静默改变生产结果。

**运行、提交、重现方式与覆盖规则。**

原串行 CLI 的关键位置参数是 `data_filename random_filename nthreads`。按当前脚本可构造下面的 RR-only 命令；它是重现方式说明，**不是已找到的历史命令，也没有在此运行**。示例显式选择 64 线程，并使用新文件名，避免覆盖原基准：

```sh
/home/xiaoliang/.conda/envs/xiaoliang/bin/python \
  /mnt/xiaodong/xiaoliang/software/LSS_python/bin/cal_2pCF_box \
  __RR_ONLY_PLACEHOLDER__.npy \
  /mnt/xiaodong/xiaoliang/JT-2G/RR/random_box2000_3e-2.npy 64 \
  -smin 0 -smax 150 -sbin 150 -mubin 120 --only_run_rr \
  -rr /mnt/xiaodong/xiaoliang/software/LSS_python/validation/RR_serial_review_NEW.pkl -v
```

MPI CLI 的受控重现方式如下，同样**没有执行**，需在获得相应资源和相容 MPI 环境后使用：

```sh
/software/mpich4.3-gnu12/bin/mpiexec -n 2 \
  /home/xiaoliang/.conda/envs/xiaoliang/bin/python \
  /mnt/xiaodong/xiaoliang/software/LSS_python/bin/cal_2pCF_box_mpi \
  __RR_ONLY_PLACEHOLDER__.npy \
  /mnt/xiaodong/xiaoliang/JT-2G/RR/random_box2000_3e-2.npy 64 \
  --boxsize 2000 --ngrids 4 \
  -smin 0 -smax 150 -sbin 150 -mubin 120 --only_run_rr \
  -rr /mnt/xiaodong/xiaoliang/software/LSS_python/validation/RR_mpi_review_NEW.pkl -v
```

两个 `*_review_NEW.pkl` 都是预期重算输出，本次未生成、未检查是否已有同名文件；实际使用应选未占用的新名称。原脚本和新 CLI 的 RR-only 均会复用已有 RR 并跳过；`--force_rr/-forcerr` 才允许重算并覆盖 RR。`-force/-f` 面向 DD/DR/tpCF，不等于强制 RR。当前生产 Slurm 脚本没有把 `--force_rr` 传入。

当前生产 Slurm 的运行方式是：

```sh
cd /mnt/xiaodong/xiaoliang/JT-2G/src
sbatch /mnt/xiaodong/xiaoliang/JT-2G/src/run_RR_mpi.slurm sys 4
```

这与成功日志中的模式/网格相符，但精确历史 `sbatch` 命令未恢复。本次没有提交；在结果已存在时，上述原脚本会跳过 RR。若要真实重算，应使用新的输出路径或显式了解覆盖行为后调整验证脚本。原提交脚本依赖 Slurm、MPICH、Python 环境和共享路径，不负责加载 conda 环境或固定 Corrfunc ISA。

已有结果的轻量比较不需要 MPI、Corrfunc 计算或 Slurm，可运行：

```sh
/home/xiaoliang/.conda/envs/xiaoliang/bin/python \
  /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/compare_rr.py \
  /mnt/xiaodong/xiaoliang/JT-2G/RR/RR-random_box2000_3e-2-150s0to150_120mu.pkl \
  /mnt/xiaodong/xiaoliang/JT-2G/RR/RR-random_box2000_3e-2-150s0to150_120mu_mpi.pkl \
  --output-dir /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/REVIEW_NEW
```

该比较程序只依赖 Python/NumPy/joblib：读取两数组、验证 layout/元数据、精确整数相减、生成 JSON 与 CSV 及 Markdown。无参数打印帮助；输出目录可新建，已有报告文件则拒绝覆盖。上面 `REVIEW_NEW` 目录是示例路径，本次未运行该命令，也未生成该目录的结果。

相关测试程序位于：`/mnt/xiaodong/xiaoliang/software/LSS_python/tests/test_pair_counting.py`、`/mnt/xiaodong/xiaoliang/software/LSS_python/tests/test_tpcf_mpi.py`、`/mnt/xiaodong/xiaoliang/software/LSS_python/tests/test_cal_2pcf_box_mpi.py`。前次会话记录过核心 25 项、MPI API 2 项、CLI 6 项通过，verbose 修改后 API+CLI 共 8 项通过；**本报告没有重跑它们**，也没有把小样本通过升级为全量逐 bin 一致的证明。独立数值验证入口为 `tests/validate_pair_counting.py` 与 `tests/validate_tpcf_mpi.py`，完整绝对路径在清单中。

**日志、缓存、临时文件和未完成作业。**

成功日志的绝对路径已列在前文，stderr 已确认空文件。其余日志检查发现：127779 仅输出用法，127780 找不到可执行文件，127781 被取消，127803 有 OOM/原生 MPI 中止，127806 与 127823 被取消，不能算作本文件的成功生成作业。127823 使用 ngrids=10，不能据此替代最终的 ngrids=4。所有读取过的日志和脚本逐项列在附属文件清单中。

`/mnt/xiaodong/xiaoliang/JT-2G/src/run_tpCF_all_119378.log` 记录了复用已有无 MPI 后缀的 150×120 RR，可证明文件曾被下游读取，但不能证明其最初由该作业生成。119387 对应的 750×600 流程不属于本次 150×120 基准来源。未找到原始 RR-only 基准作业日志。

Slurm stdout/stderr 命名规则来自当前脚本：`<提交工作目录>/run_RR_mpi_<JobID>.log` 与 `.err`。本次已核实的路径在 `/mnt/xiaodong/xiaoliang/JT-2G/src/`；将来提交仍取决于实际工作目录/输出选项，不能固定推定所有作业日志都在脚本目录。

Corrfunc 对数组形式的 `sedges` 会在每次 DDsmu 调用创建临时文本分箱文件，生成函数位于当前安装 `Corrfunc/utils.py` 第 324–377 行，使用 `tempfile.NamedTemporaryFile(delete=False)`；包装器成功返回后在 `DDsmu.py` 第 349–351 行删除它。当前检查主机的目录是 `/tmp`，命名规则 `/tmp/tmp<随机后缀>`；生产节点如设置 TMPDIR，实际规则是 `<各节点 tempfile.gettempdir()>/tmp<随机后缀>`，具体历史目录/文件名未记录。C 异常或进程被杀时可能没有执行删除。没有扫描、删除或把当前 `/tmp/tmp*` 文件认领为这两个作业的遗留物。

本 RR-only 流程只把最终 RR 持久化：块、任务及局部计数在内存中，不保存逐块磁盘结果或 checkpoint。DD_temp.pkl、DR_temp.pkl、tpCF_temp.pkl 是完整模式可用的路径，RR-only 不生成它们；不能据此声称这些文件在其他工作目录不存在。原随机目录、最终两份 RR 和现存生产日志都是既有输入，本报告没有生成或覆盖它们。

**本次报告交付文件与完整清单。**

| 绝对路径 | 来源、用途、格式、实际状态 |
| --- | --- |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/doc/rr_mpi_review_20261021.md` | 本次新建的最终报告，Markdown；供 Claude 审阅 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/environment_manifest.json` | 本次已生成；检查环境、模块位置/版本、源码/二进制/输入/日志哈希；当前环境与历史环境明确区分 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/file_inventory.csv` | 本次生成的完整逐项文件清单；包含输入、源码依赖、实际查看的日志、既有比较产物、新报告产物及动态临时/预期重算输出规则 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_kernels.c.src` | 本次从 `/tmp/countpairs_s_mu_kernels.c.src` 复制的上游参考源码，已生成；不是已验证的生产二进制源码 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_impl.c.src` | 本次从 `/tmp/countpairs_s_mu_impl.c.src` 复制的上游参考源码，已生成；同上 |

文件清单 CSV 逐项给出绝对路径、来源/生成环节、用途、格式、是否存在、此次采取的操作与是否本次生成。它也覆盖依赖模块和已读历史日志，无需根据通配符猜测具体已检查文件。未复制完整 random、两份 pkl 或大型 Python 环境；Claude 如没有这些本地路径的读取权限，需要用户另外提供相关文件。没有调用或给外部 Claude 发消息。

本次没有创建独立 stdout/stderr 日志，执行输出保留于当前会话；没有生成额外临时数据、缓存、checkpoint 或压缩归档。仅在新审阅目录和新 Markdown 路径写入报告材料，不清理任何已有文件。原比较产物保留不变，当前两份 RR 哈希再次核对一致。

建议 Claude 先读本报告，再看 `summary.json`、成功作业 `run_RR_mpi_127826.log`、四个计数相关 Python 文件和两份上游 C 参考源码；在尚未建立受控对照前，保留“串行作为回归基准、MPI 结果待解释”的审阅口径。
