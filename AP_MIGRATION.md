# Dense AP 四边形算法迁移（0.6.0）

`tpcf_convert_main` 和 `xismu.cosmo_conv_DenseToSparse` 的 dense 路径默认使用
`convert_algorithm="polygon"`。已有位置参数不变；该新参数只能按关键字传入。
`c_api=True` 使用 C++，`False` 使用等价的串行 Numba 实现。

`curved_gauss8`、`curved_analytic` 是保留名称，当前抛出
`NotImplementedError`；未知名称抛出 `ValueError`。正式接口没有 `legacy`。
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
不是曲边的精确积分，也不是按 `s² ds dmu` 加权。超出输出域的贡献被裁掉，不重新归一化。
目标格与映射窗口相接即选择 AP 转换，窗口外使用恒等几何重分箱；混合两种区域时不要求
全域计数守恒。底层四项窗口参数的径向和角向范围现在均生效。

高层输出使用 assistant 的尺寸和坐标，源/目标要求相同 smax；直接调用对象方法仍可
不提供 assistant、显式给出 sbin2/mubin2。DD/DR/RR、归一化常数及权重矩保留原有含义。
近基准快速返回直接返回 assistant；设置 `ap_tol=0` 强制执行重分箱。

## 构建与验证

在库根目录运行 `make AP_pybind`，随后使用新 Python 进程加载扩展。
构建只更新 AP，不需要重新构建 mesh/FFT。生成 wheel 时应使用干净的 build 目录，
避免 setuptools 将旧构建缓存中的已删除模块再次打包。

```sh
PYTHONPATH=src python -m pytest -q tests/test_ap_polygon.py tests/test_ap_cpp.py
PYTHONPATH=src python tests/validate_ap_migration.py --help
```

完整项目对照使用独立输出目录，读取已保存的四边形测试结果，不覆盖原科学结果：

```sh
PYTHONPATH=src mpiexec -n 30 python tests/validate_ap_migration.py --run --mpi \
  --project /mnt/xiaodong/xiaoliang/JT-2G/cov_bias_test \
  --output /tmp/lss-ap-migration-validation
```

旧算法和复现方法见 `archive/ap_dense_legacy/README.md`。默认算法改变后，历史脚本若仍将
当前共享库入口标为 legacy，必须改为通过隔离进程使用归档版本才能做正确的新旧对照。
此次库替换不自动备份、失效或重建 JT-2G 的既有结果、PCA 模型和缓存。
归一化均值接近零与零 RR 是不同问题；现有统计量正则化保持原样。
