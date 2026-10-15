# 旧版 dense AP 实现归档

这是替换四边形算法前的完整 Python 包源码及构建配置快照，包含当时 dense 所依赖的
函数和导入关系。它位于正式 src 目录之外，不参与当前包构建或默认导入。
`manifest.json` 记录原库 Git commit、Python/ABI 和全部快照文件的 SHA-256。
当时的二进制在本机归档中保留，但忽略提交；跨机器复现应从归档源码重建。

## 为什么替换

旧 C++ 搜索只逆变换两个对角点，可能漏掉有正交集的源格。旧面积算法依赖 out/scope
分类，可能把正交集赋为零、把部分覆盖赋为全权重，或者漏掉所有顶点在格外的交集。
新算法使用完整四点搜索、保守扩展和四边形裁剪。新旧算法均不是曲边精确积分。
这些差异可能改变 RR、统计量、卡方和 PCA；新算法不会普遍消除统计量均值接近零。

## 独立进程复现

请在此目录运行，使用与待测环境相符的 Python 和编译器：

```sh
make AP_pybind
PYTHONPATH="$PWD/src" python -c 'import LSS_python.AP as ap; print(ap.__file__)'
PYTHONPATH="$PWD/src" python /absolute/path/to/your_legacy_test.py
```

输出路径必须指向本归档下的 src。旧入口没有 convert_algorithm 参数。
旧版 C++ 与旧版 Numba 自身也可能有差异，比较时必须记录 c_api 的值。
不要在已加载新版 LSS_python/AP_pybind 的进程中替换 sys.path 再尝试导入旧版；
请分别启动进程并把待测参数、计数和结果存入独立文件。
不要将归档目录安装到当前可编辑安装上，也不要覆盖当前活动二进制。

归档测试输出应另存。若必须完整回滚，请从该快照重建独立环境并验证导入路径，
不要只替换某一个头文件或某一个 Python 模块。
