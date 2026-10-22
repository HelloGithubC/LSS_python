**xismu、Corrfunc 测量与重采样统计的模块布局**

实施日期：2026-10-21，Asia/Shanghai。

结果对象和分析工具统一从 `LSS_python.xismu` 导入。`LSS_python.tpcf` 提供 Corrfunc 测量和计数结果转换；`LSS_python.cov` 保留重采样流程及统计重建。现有 MPI 调度模块及底层计数行为保持不变。

| 模块 | 核心职责和入口 |
| --- | --- |
| `LSS_python.xismu` | `xismu`；积分/重分箱；`average_xismus_dict_list`；`get_diff_array*`；`meannorm`、`packarray1d/2d`、`cal_convert_factor`；`create_xismu_from_pairs`；`migrate_xismu_file` |
| `LSS_python.tpcf` | `run_tpCF`、`run_tpCF_mock`、`cal_tpCF_from_pairs`；`compute_DD_JK`、`compute_DR_JK`、`compute_RR_JK`、`compute_subsample_pairs`、`extract_npairs` |
| `LSS_python.cov` | `run_jackknife_tpCF`、`run_subsample_tpCF`；分区、RR 缓存、归一化、Jackknife 扣除和 ξ 重建、协方差统计 |
| `LSS_python.pair_counting` | 既有通用计数/任务接口；共享的非周期 `get_neighbors` |
| `LSS_python.tpcf_mpi` | 既有 MPI 分配、局部计数和全局 Reduce；本轮未修改 |

`xismu` 不导入计数模块或 Corrfunc。AP 几何和压缩器继续在实际使用时导入，避免仅加载结果对象就引入计数和拟合依赖。

**新导入方式与兼容。**

```python
from LSS_python.xismu import (
    xismu, average_xismus_dict_list, get_diff_array,
)
from LSS_python.tpcf import run_tpCF, cal_tpCF_from_pairs
from LSS_python.cov import run_jackknife_tpCF, run_subsample_tpCF
```

旧公开工具名仍通过直接导入导出，同一函数只有一份实现。`LSS_python.tpcf.xismu` 通过模块 `__getattr__` 返回新模块中的同一个类；使用旧导入或加载引用旧类路径的对象存档，会发出每进程一次的 `FutureWarning`，指出新导入方式和 `migrate_xismu FILE` 命令。普通 `tpcf` 导入和新路径导入不触发该提示。

`xismu.save()` 原有的数组/字典文件格式不变。直接 joblib 序列化的对象、对象数组或容器则使用新的类路径；旧文件仍可通过兼容入口读取。项目内 AP、拟合、绘图、命令行、测试和 notebook 的正常引用已切换到新路径。

**计数与统计的调用关系。**

```text
cov.run_jackknife_tpCF
  -> tpcf.compute_DD_JK / compute_DR_JK / compute_RR_JK
  -> cov 中的总计数、扣除、归一化与 Jackknife ξ 重建
  -> xismu.create_xismu_from_pairs

cov.run_subsample_tpCF
  -> tpcf.compute_subsample_pairs(run_parts=("RR",))，按需计算 RR
  -> cov 保存/读取 RR 缓存并保留既有自配对处理
  -> tpcf.compute_subsample_pairs(run_parts=("DD", "DR"))
  -> cov 归一化与自配对处理
  -> xismu.create_xismu_from_pairs
```

新增计数入口：

```python
compute_subsample_pairs(
    data_list, random, sedges, mubin, with_weight, boxsize,
    refine_factors=(2, 2, 1), nthreads=1, verbose=False, *,
    run_parts=("DD", "DR", "RR"),
)
```

输入为已准备的子目录列表及一个子盒大小的共享 random，`boxsize` 为传给既有适配器的子盒尺寸。返回 `DD_sub`、`DR_sub`、`RR_result`，保留 Corrfunc 结构化数组；未请求的字典为空，未请求的 RR 为 `None`。DD/DR 沿用原有不足两个点时返回 `None` 的行为。该入口不处理缓存、归一化、自配对扣除、全零权重兼容或结果对象构造。

此次保持计数顺序、dtype、跨块倍数、原有自配对规则、全零权重兼容、输出字段和缓存格式。`cal_tpCF_from_pairs` 仍在 `tpcf`。MPI 消费既有 `pair_counting` 接口，不改为新的重采样路径。

**旧对象文件原地迁移。**

工具源码：`/mnt/xiaodong/xiaoliang/software/LSS_python/bin/migrate_xismu`。用户命令入口：`/home/xiaoliang/bins/migrate_xismu`，以符号链接指向源码，与现有工具组织一致。

```sh
/home/xiaoliang/bins/migrate_xismu /absolute/path/result.pkl
/home/xiaoliang/bins/migrate_xismu /absolute/path/results/*.pkl
```

`/absolute/path/…` 是用户待选文件的示例，不表示已迁移或生成这些文件。本次只用测试目录中的副本验证，没有迁移生产文件。

脚本依赖当前 Python 环境、更新后的 LSS_python、NumPy/joblib 及项目依赖。无参数或 `--help` 打印帮助；唯一位置参数为一个或多个文件路径。每个文件按顺序执行：

1. 解析真实文件路径并加载 joblib/pickle，在兼容入口记录是否实际解析了旧类路径；该记录独立于每进程一次的警告，也覆盖 joblib 对象数组中的独立 pickle 流。
2. 没有旧类路径则输出 `skipped: ... (no legacy xismu references)`，包括没有目标对象以及已经迁移的文件；保留文件内容、修改时间和 inode，不创建临时文件。仍需读取一次文件进行判断，但跳过写入、二次加载和逐项数值校验。
3. 有目标对象则在源目录创建唯一临时文件，通过 `joblib.dump` 保存同一对象状态，不调用构造函数或 `xismu.save()`。
4. 重新加载，核对对象/容器、属性、数组形状、dtype、数值和新类路径，保留源文件权限后以 `os.replace` 原地替换。
5. 成功输出 `migrated` 及对象数。失败输出 `error` 到 stderr，继续处理其余文件；只要有文件失败，最终退出码为 1。

`migrated` 后的数量是本次重写文件中唯一 `xismu` 实例的总数。旧路径在加载时会直接解析为新类，因此不能仅通过加载后的实例类型来判断是否需要迁移；兼容入口的加载记录用于区分新旧存档。再次运行已成功迁移的文件会返回 0 并跳过，即使同一进程此前已发出旧路径警告，尚未迁移的其他文件仍能正常识别。

**输入文件就是最终输出文件，会被原地覆盖；不创建 `.bak`，没有副本/备份参数，也没有交互确认。** 临时文件命名为 `<源文件绝对目录>/.<源文件名>.migrate-<随机后缀>`，成功替换或失败时清理本次临时文件和同前缀 sidecar。替换前失败时，原文件内容保留。命令 stdout/stderr 默认进入终端，不额外写日志或检查点。识别到的常见 joblib 压缩类型会保留；对象与数组校验不依赖压缩字节相同。

2026-10-22 修复了已迁移文件仍被重复重写的问题。42 项对象、兼容、迁移及差分测试通过，包括六种压缩配置下的重复调用、字节/修改时间/inode 不变、新旧混合对象，以及警告已发出后的连续旧文件迁移。该次修复的日志、临时测试文件和完整文件清单位于 `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/xismu_migration_repeat/20261022_1u38xqul/`；`report.md` 说明复现、修复及耗时边界，`file_inventory.csv` 列出绝对路径和实际状态。未读取或迁移用户报告中的生产文件。

**验证及完整文件交付。**

本次验证目录：`/mnt/xiaodong/xiaoliang/software/LSS_python/validation/xismu_reorganization/20261021_kjo9l5p6/`。完整逐项清单为该目录的 `file_inventory.csv`，记录所有输入、源码、最终输出、中间文件、日志、缓存及动态临时文件规则，包含绝对路径、来源、用途、格式和实际生成状态。

验证包括：

实际结果：195 项 pytest 测试通过（149 项 AP/通用计数、8 项 MPI、38 项对象/兼容/迁移/差分），25 份迁移前后数值结果严格一致；24 项原实现 AST/未改动源码检查通过。旧存档测试产生的 1 条 FutureWarning 为预期提示。详细命令与日志路径记录于验证目录的 `verification.json`。

- 迁移前的固定小样本基准：普通测量、Jackknife/subsample 新计算和缓存读取，共 25 份冻结结果；加权、无权重、单位权重补列、float32、空区域。
- 迁移后的同环境数值比较，以及原函数/类 AST 等价检查；数学主体原样迁移，独立逻辑问题不在本轮修正。
- 新旧导入、旧路径仅提示一次、结果模块不加载 Corrfunc/统计/压缩器的依赖检查。
- 现有 AP、点对计数、MPI 和命令行回归测试；只运行本机小样本，没有 Slurm 提交或全量 RR 重算。
- 旧类路径存档的临时副本迁移、容器和数组、权限、无对象跳过、写入/重新加载/校验失败、临时文件清理、批量输入和失败退出码。

数值验证程序为上述目录的 `compare_measurements.py`，依赖项目现有科学计算环境。无参数打印帮助；`capture` 生成迁移前基准，`verify` 比较当前实现。输入由固定种子 20261021 在内存生成：boxsize=20、2×2×2 分区、s=0..5、4 个 μ bin、单线程；没有读取生产 random。

```sh
/home/xiaoliang/.conda/envs/xiaoliang/bin/python \
  /mnt/xiaodong/xiaoliang/software/LSS_python/validation/xismu_reorganization/20261021_kjo9l5p6/compare_measurements.py verify
```

`capture` 会覆盖同名本地基准、fixture 和清单，并为冷缓存测试删除对应的本地 RR 缓存，**不要用当前代码重跑 capture 替换迁移前证据**。`verify` 保留原基准，使用单独的 `verify_rr` 缓存目录，更新自己的验证报告；这些路径及操作在清单中列出。

pytest 使用本验证目录下的独立 `pytest-*` basetemp，相关 stdout/stderr 日志也保留在本目录。复用同一个 `--basetemp` 时 pytest 会清理其旧内容，其他目录不受影响。Python 字节码、pytest 缓存以及绘图库缓存的实际文件和可确定规则见清单。原生产结果没有迁移、覆盖或删除。
