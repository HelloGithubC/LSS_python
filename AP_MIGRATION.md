# Dense AP 面积算法迁移（0.6.0）

`tpcf_convert_main` 和 `xismu.cosmo_conv_DenseToSparse` 的 dense 路径默认使用
`convert_algorithm="polygon"`。已有位置参数不变；该新参数只能按关键字传入。
`c_api=True` 使用 C++，`False` 使用等价的串行 Numba 实现。

`curved_gauss8`、`curved_analytic` 已在 C++ 与 Numba 实现，按关键字显式选择；
未知名称抛出 `ValueError`。正式接口没有 `legacy`。
`convert_method="simple"` 沿用现有插值；算法选择参数控制 dense 路径。

## 几何与计数

DA、Hz 在一次转换中各计算 fiducial/model 两个值。dense 路径统一使用
`z=max(redshift,1e-4)`；任一 wa 为 None 时按常数 w 处理。几何内核只接收
`alpha_perp=DA_m/DA_f`、`alpha_parallel=Hz_f/Hz_m`。
PS 等其他转换路径不随此次迁移改变。

候选源格搜索使用四个逆变换角点，并按相交角向格边界上的缩放因子保守扩展径向范围。
映射后的源格以四条直线构成四边形，按周界顺序与目标矩形裁剪：

`rate = area(source_polygon ∩ target_rectangle) / area(full_source_polygon)`。

分母是裁剪前的完整四边形面积。这里采用映射后 `(s,mu)` 坐标面积上格内均匀的近似，
polygon 使用直线弦近似。两种 curved 方法使用真实曲边，分母是完整曲边源格的面积，
分子是与目标格的交集面积。三种算法均不按 `s² ds dmu` 加权。超出输出域的贡献被裁掉，不重新归一化。
目标格与映射窗口相接即选择 AP 转换，窗口外使用恒等几何重分箱；混合两种区域时不要求
全域计数守恒。底层四项窗口参数的径向和角向范围现在均生效。

高层输出使用 assistant 的尺寸和坐标，源/目标要求相同 smax；直接调用对象方法仍可
不提供 assistant、显式给出 sbin2/mubin2。DD/DR/RR、归一化常数及权重矩保留原有含义。
近基准快速返回直接返回 assistant；设置 `ap_tol=0` 强制执行重分箱。

## 曲边数值规则

`curved_gauss8` 先找齐曲边/径向边交点、分段后直接积分非负重叠宽度。
以整段与两个半段的八点求积比较误差，复用半段值；内部相对目标 `1e-12`，
最大二分深度 24，并考虑坐标相减的浮点可表示误差下限。未收敛明确报错，
不切换解析法或 polygon；C++ 在线程外报告。一般曲边分母也用自适应 Gauss8，按源角向格缓存。
严格 `a==b` 时两端共同直接计算常数 `R=a` 对应的矩形面积，不使用一般曲边公式。
`curved_analytic` 使用同一几何分段，以稳定的 asinh/asin 差式或近各向同性展开积分。
两者均检查正且有限的完整面积，不加入面积正则化。详见 [tpCF_AP_note.md](doc/tpCF_AP_note.md)。

## 构建与验证

在库根目录运行 `make AP_pybind`，随后使用新 Python 进程加载扩展。
构建只更新 AP，不需要重新构建 mesh/FFT。生成 wheel 时应使用干净的 build 目录，
避免 setuptools 将旧构建缓存中的已删除模块再次打包。

```sh
PYTHONPATH=src python -m pytest -q tests/test_ap_polygon.py tests/test_ap_cpp.py tests/test_ap_curved.py
PYTHONPATH=src python tests/validate_ap_migration.py --help
```

原有 polygon 迁移的全量复现命令如下（本轮曲边验证未执行此全量/PCA 扫描），使用独立输出目录：

```sh
PYTHONPATH=src mpiexec -n 30 python tests/validate_ap_migration.py --run --mpi \
  --project /mnt/xiaodong/xiaoliang/JT-2G/cov_bias_test \
  --output /tmp/lss-ap-migration-validation
```

旧算法和复现方法见 `archive/ap_dense_legacy/README.md`。默认算法改变后，历史脚本若仍将
当前共享库入口标为 legacy，必须改为通过隔离进程使用归档版本才能做正确的新旧对照。
此次库替换不自动备份、失效或重建 JT-2G 的既有结果、PCA 模型和缓存。
归一化均值接近零与零 RR 是不同问题；现有统计量正则化保持原样。

曲边代表点及完整转换耗时使用新目录独立输出：

```sh
PYTHONPATH=src python tests/validate_ap_curved.py --run \
  --project /mnt/xiaodong/xiaoliang/JT-2G/cov_bias_test \
  --output /tmp/lss-ap-curved-validation --repeats 5
```

该检查枚举实际训练数据中的普通点及 AP 比例极值，覆盖已知异常点，比较三种算法
在固定 precision 下的代表点卡方；不执行整批训练或重建 PCA。完整源格的独立参考
在 `test_ap_curved.py` 中枚举全部源格并使用独立求根和数值积分。

首次接入时安装位置回归测试 **100 项通过**，另有八组 polygon 替换前后位级一致性检查。
独立积分的 72 幅单源格映射/算法最大权重绝对误差分别为 `3.56e-15`（Gauss8）、
`4.67e-15`（解析法）。71 个代表红移对/参数组合中，两种曲边法的最大卡方差
为 `2.4011e-10`；相对 polygon 的最大变化约 `0.42035`，仅代表所选点。
预热后完整双切片转换，相对 polygon 的中位耗时比：C++ 为 Gauss8 `3.67` 倍、
解析法 `1.69` 倍；串行 Numba 为 `1.63` 倍、`0.956` 倍。
详细范围、计数/统计量/卡方表和复现路径见 `doc/tpCF_AP_note.md` 第 9 节。

C++/Numba 已统一使用同一空段预判、严格各向同性矩形退化、端点加中点的舍入尺度，
以及固定容量栈的从左到右 Gauss8 自适应累加。对不存在内部浮点中点的区间，
仅在端点变化界满足既有误差目标时接受原有整段求积值，否则报错；没有面积正则化或静默回退。
上述首次接入耗时表是统一前基准；本次统一的验证见 `doc/tpCF_AP_note.md` 第 9.2 节。

双端统一后的安装位置回归 **124 项通过**。24 项 polygon 小网格对照逐元素不变；
71 个实际代表点组合、两个后端和三个算法的 426 个统计向量与统一前全部逐元素相同。
六组严格各向同性用例中，两种曲边算法与两个后端也逐元素一致。
本轮完整记录位于工作区 `cov_bias_test/diagnostics/ap_curved_backend_unification_20261018/`。
