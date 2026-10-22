**对 Gemini 4 Argon 的 RR 审阅报告的复核**

日期：2026-10-21，Asia/Shanghai。输入为用户粘贴的审阅文本：`/home/xiaoliang/.codex/attachments/8ea53817-0059-4612-a09b-2ccb1a5a8d15/已粘贴的文本.txt`。本文结合现存源码、日志、全量差值表和本次受控实验，区分已经核实的事实与仍需验证的归因。

判断：**报告给出了目前应优先验证的解释，尤其是不同 SIMD 内核的距离计算和分箱差别；但“已完整还原历史运行、全部偏差约 100% 已解释、实现完全无缺陷”的结论超过了现有证据。** 它比“float32 有一点误差”具体得多，值得沿这条路线验证；不能直接作为此次 MPI RR 已通过验收的依据。

**哪些内容核实成立。**

| 报告内容 | 本次核对结果 |
| --- | --- |
| AVX 与 AVX-512 距离计算不同 | 上游 2.5.3 参考源码确实如此：AVX 分别计算三个平方，先加 x/y；AVX-512 先舍入 z²，再执行 y、x 的两次 FMA。既改变乘加舍入，也改变求和次序。 |
| AVX 有标量尾部，AVX-512 使用掩码处理尾部 | 与参考源码一致；这会影响目录组织变化后的浮点计算路径。 |
| SIMD 展平索引可能把有效 mu 格推入不返回的 mu=120 槽 | 与源码和独立实测一致。我们的最小见证中，同一个有效点对在 float32 AVX 下从计数 1 变为 0；只是加入了 7 个超出 smax 的点。 |
| 径向尖峰与平方距离的浮点间距变化有关 | 实测 s_index=32、45、64、90、128 的径向差分别为 +3050、+6234、+15352、+37588、+102372。这些位置与 k² 跨越二进制幂的特征吻合，是值得追踪的数值线索。 |
| mu 两端和内部孤立负峰 | 实际 mu=0 总差 −58750、mu=119 总差 −55572；delta[134,48:51]=[248,−1244,194]，与报告引用相符。135×121+49=16384 附近的索引间距变化提供了具体解释候选。 |

源码证据分别在 [距离 FMA](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_kernels.c.src:212)、[AVX 距离](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_kernels.c.src:506)、[AVX 浮点索引](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_kernels.c.src:584) 和 [输出打包](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_comparison/20261021_150s_120mu_p8ja8f1t/review_context_20261021/countpairs_s_mu_impl.c.src:683)。这些是已保存的官方 tag 参考副本，不是已验证与生产二进制完全对应的构建源码。

**历史运行的证据存在断点。**

用户已经明确说明串行基准由仓库 `bin/cal_2pCF_box` 的 RR-only 模式生成。Argon 改为断言它由 `run_tpCF_all.sh -> cal_2pCF_box_weight_ap` 的完整流程生成，并推定在 xiaodong-GPU01 上使用 40 线程。这与用户给出的来源不一致，不能仅靠脚本 mtime 取代用户提供的来源。

当前 [run_tpCF_all.sh](/mnt/xiaodong/xiaoliang/JT-2G/src/run_tpCF_all.sh:16) 确实设置 40 线程，并调用 `cal_2pCF_box_weight_ap`；脚本有这个内容并不证明它在相应时刻执行，也不证明它负责写入特定 RR。mtime 只能作为时间线索；“tpCF result exists, skip” 只能证明当时已有文件，不能证明刚由哪个脚本生成。

[119378 stdout](/mnt/xiaodong/xiaoliang/JT-2G/src/run_tpCF_all_119378.log:1) 证明该作业复用了现有 RR；[119378 stderr](/mnt/xiaodong/xiaoliang/JT-2G/src/run_tpCF_all_119378.err:5) 和 [119387 stderr](/mnt/xiaodong/xiaoliang/JT-2G/src/run_tpCF_all_119387.err:5) 确实记录 `Using AVX512 kernel`。但它们记录的 DD/DR 调用不等于两份待比较 RR 的生成调用，也没有在这些位置给出 cn4/cn5 的身份。

[127826 stdout](/mnt/xiaodong/xiaoliang/JT-2G/src/run_RR_mpi_127826.log:42) 明确记录 MPI rank 位于 cn5/cn4，并保存目标 MPI RR，但未记录 CPU 型号或具体内核；stderr 为空。同一 Slurm 分区不能独立证明同一 CPU 型号、同一二进制构建或同一运行时分派。因此“串行 AVX、MPI AVX512”目前是有线索支持的假说，尚未形成这两次 RR 运行各自的直接取证链。

**所谓单向偏置应表述为需要验证的统计机制。**

整数平方阈值可精确表示且低位为偶数，这意味着恰落在相应中点的数会按向偶规则舍入到阈值。这个事实本身不能比较两个不同表达式树对任意输入的结果，也不能独立推出任意输入分布都满足净通量 F(k)>0。

本次用 C float、显式中间舍入和 `fmaf` 模拟参考源码的两种表达式，在阈值 129²=16641 附近生成 2,000,000 个候选三元组，观测到 AVX 表达式独自跨界 301 次、FMA 表达式独自跨界 224 次。两个方向都存在：

| float32 dx, dy, dz | AVX 表达式的 s² | 嵌套 FMA 表达式的 s² |
| --- | ---: | ---: |
| 22.1973324, 70.0398254, 106.031601 | 16641 | 16640.998046875 |
| 70.8777008, 107.770668, 1.68357587 | 16640.998046875 | 16641 |

这个样本的净方向仍与 Argon 的假说相同，不能把两个反向见证误写成“否定真实 random 上的净偏置”。它说明需要指定输入分布和测量统计量；“严格单向定理”不能仅由阈值的偶数低位推出。嵌套 FMA 也不是对 dx²+dy²+dz² 一次精确求和：z² 和内层 FMA 都仍有各自的舍入。

该实验模拟运算表达式，没有执行 AVX-512 指令或 Corrfunc 的 AVX-512 内核，更没有测量全量 random 的边界事件。输入生成方式和种子均保存在 C 源码与环境记录中，不能把 301/224 当作生产事件比例。

**累计差不等于独立测得的物理通量，预测也仍有残差。**

如果定义 F_obs(K)=sum(delta_radial[0:K])，那么 delta_radial(k)=F_obs(k+1)−F_obs(k) 自动成立。这是累计与差分关系，不能独立证明每个差值都源自距离边界迁移。

如果 F_rad 表示只由距离舍入导致的净跨界数，而 E_mu 表示 mu 裁剪/溢出引起的径向净变化，完整关系至少应写成：

```text
delta_radial(k) = F_rad(k+1) - F_rad(k) + E_mu(k) + E_other(k)
sum(delta_radial) = F_rad(150) - F_rad(0) + sum(E_mu) + sum(E_other)
```

报告自己也指出 mu=120 溢出会丢计数，因此不能在没有独立扣除这些项时，把 +1,047,222 全部定义成仅在 s=150 发生的净接纳数。径向迁移当然可以在 s=150 改变总数；这里质疑的是“已精确测得全部来自这里”，不是该机制的可能性。

Argon 表中的理论值为 +1,196,478，实测为 +1,047,222，残差为 +149,256，占实测 14.25%。s=90 的累计值残差约 17.82%，s=128 约 13.41%。这些预测支持量级和部分形状，尚不能称为“~100% 定量复现”。粘贴文本没有附 Monte Carlo 源码、采样权重、随机种子、原始输出或不确定度，本次无法独立重跑它的积分模型。

Top 10 最大绝对差确实集中在 dz≈sqrt(8192) 附近，9 个 bin 的范围与该曲线相交；最大的 (s_index=128,mu_index=85) bin 满足 dz≥128×85/120=90.6667，大于 sqrt(8192)≈90.5097，严格说并不相交。这不削弱“附近有脊线”的观测，但“100% 精确落在线上”的表述需收敛。

报告对标量尾部比例使用约 3.5/30000 的估计；实际候选循环长度、起始偏移、有效点对筛选和不同任务的尾部占比未直接测量。因此 −58150、−1160 等数值目前属于模型估计，不能仅凭近似吻合当作已逐对归因。

**对实现和后续工作的影响。**

在 smax=150、块宽=500 的既定条件下，目前块覆盖、跨块 multiplicity=2 和 uint64 归并设计合理；没有从这些差值中发现必须通过 Jackknife 系数或跨块系数修正才能解决的证据。应优先固定数值路径做对照，不能为补偿未归因的数值差异而改统计系数。

我们已有的真实 random 前 100,000 点对照，在同机、同 ISA=avx、单线程下，float32 仍有 2874 个 bin 不同、总差 −12；float64 AVX 及 float32 fallback 下整目录/分块各自完全一致。这与 Argon 的“同 AVX 仍会因尾部变化出现小差异”兼容，也说明“固定 ISA”本身不保证 float32 分块逐 bit 一致。对 AVX-512 的整目录/分块是否完全一致，本次没有原生测试；报告中的预言需要实验，不能从无标量尾部直接推广成对整个计数流程的保证。

进一步最有判别力的验证是在同一支持 AVX-512 的节点、同一二进制、同一输入上做四种组合：整目录/分块 × avx/avx512f，再对完全相同的存盘坐标同时提升到 float64 做对照，记录实际内核与环境。可先用足够密的小体积样本或适中目录验证，不必立即重跑完整 2.4 亿点。本次没有提交该作业，也没有改动生产 API、传入 ISA 或选择新 RR 替换旧 RR。

**复核程序、运行方法与文件交付。**

本次新审阅产物目录为 `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/`。C 实验目的为检查表达式是否存在双向边界跨越；流程是固定种子生成 x/y、按接近球壳的 z 构造候选，分别计算两个 s² 表达式，统计阈值两侧分类并打印两个见证。固定阈值 16641，候选循环 2,000,000 次；部分候选因剩余平方为负而跳过。依赖 C11 编译器和 libm，无 Python/MPI/Corrfunc 运行依赖。

本次已执行的编译与运行命令为：

```sh
cc -O0 -std=c11 -ffp-contract=off -fexcess-precision=standard \
  /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/arithmetic_threshold_audit.c \
  -lm -o /mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/arithmetic_threshold_audit
/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/arithmetic_threshold_audit
```

原始初查在 `/tmp/argon_threshold_audit.c` 和 `/tmp/argon_threshold_audit` 完成，使用 `cc -O0 -std=c11 ... -lm`；源码复制到上述持久目录后，用明确禁止隐式融合的选项重新编译并运行，仍为 301/224。两处源码内容相同，两个二进制不必有相同哈希。原 `/tmp` 两文件未删除。独立原生 AVX-512 验证未运行。

数据复核使用会话内联 Python，依赖 Python 3.10、NumPy；读取既有 CSV/JSON，计算两端 mu 求和、累计径向差、报告引用预测的残差及 Top 10 曲线交集，再写入 `metrics.json`。没有保存额外 Python 脚本，没有修改已有比较结果；其 stdout/stderr 在本次会话中，无独立日志。此前 10 组 Corrfunc 对照的完整工作流程见 [float32 实验报告](/mnt/xiaodong/xiaoliang/software/LSS_python/doc/rr_float32_analysis_20261021.md)。

全部输入、输出、相关源码和日志的逐项绝对路径、用途、格式、生成状态与动态临时/缓存规则见 [完整文件清单](/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/file_inventory.csv)。本次新增：

| 绝对路径 | 来源、内容与状态 |
| --- | --- |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/doc/rr_argon_review_20261021.md` | 本补充复核报告，已生成 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/arithmetic_threshold_audit.c` | 持久保存的表达式对照源码，已生成并执行 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/arithmetic_threshold_audit` | C 编译的 ELF 可执行文件，已生成 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/build.stdout.log` | 编译 stdout，已生成、为空 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/build.stderr.log` | 编译 stderr，已生成、为空 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/arithmetic.stdout.log` | 301/224 和两个方向的数值见证，已生成 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/arithmetic.stderr.log` | 运行 stderr，已生成、为空 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/audit_environment.json` | 编译命令、编译器版本、固定种子、源码/二进制哈希、适用范围，已生成 |
| `/mnt/xiaodong/xiaoliang/software/LSS_python/validation/rr_float32/20261021_alw0uzwo/argon_review/metrics.json` | 全量特征复核、预测残差与曲线交集结果，已生成 |
| `/tmp/argon_threshold_audit.c`、`/tmp/argon_threshold_audit` | 初查源码及二进制，已生成且保留；初查 stdout 仅在会话中 |

C 编译器内部临时文件由编译器管理，实际路径未记录，未扫描或清理 `/tmp`；按当前临时目录规则为 `/tmp/cc<随机后缀>` 等，不能据此认定有遗留文件。重编译上述固定 `-o` 路径会覆盖本次实验二进制，Shell 重定向也可能覆盖日志；复现建议复制源码并选择新目录。本次创建审阅子目录时拒绝已有目录，未覆盖任何生产文件、用户附件或既有全流程报告；未生成 Slurm 日志、MPI 输出或 checkpoint。
