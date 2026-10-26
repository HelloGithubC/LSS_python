# Weight 计算拆分临时 Todo

状态：三个阶段均已完成；当前 gn 系列默认使用梯度模/密度，保留显式关闭除密度的选项。

## 阶段一：抽离计算部分（已完成）

- [x] 检查当前工作区和直接调用方；保留已有未提交修改。
- [x] 保存拆分前的小样本结果，作为本次迁移的数值基准（65 个结果数组）。
- [x] 将现有核函数、等邻居数密度/梯度和固定半径密度计算移入 `src/LSS_python/marks.py`。
- [x] 从 `MCF.py` 移除上述入口，不添加兼容包装或重新导出；保留 `create_random`。
- [x] 迁移仓库内直接调用方，包括生成脚本、lab 聚合导入和 AP 脚本的导入地址。AP 计算流程不变。
- [x] 验证拆分前后结果一致、旧入口不可用，以及生成脚本无参数显示帮助。
- [x] 更新运行记录和完整文件清单。

验证中发现已有问题：`lab.py` 从 `LSS_python.base` 导入不存在的 `cal_HI_factor`，导致整体导入失败；本次已单独执行并验证迁移后的 marks 导入语句，未修复这一无关问题。首次失败的日志已保留，最终报告也记录了该限制。

梯度沿用 Fortran 的 `nb_list0` 和 `der_w_kernel` 定义；本阶段不修改数值公式，不优化查询，不改变幂运算位置。

## 阶段二：统一方法 token 和生成入口（已完成）

- [x] 确定三种方法的统一 token 语法：`n<k>`、`gn<k>`、`r<radius>`，完整 token 为 `<方法>_weipow<alpha>`。
- [x] 将现有方法选择规则与 token 管理集中到 `LSS_python.weight_token`；bin token 文件仅保留 CLI。
- [x] 为三种生成方法定义标识及参数，保留现有 `n30_weipow...` 的语义及联合幂指数顺序。
- [x] 让生成脚本通过统一方法 token 分流，生成原始密度/梯度；幂运算继续在测量阶段执行。
- [x] 迁移当前项目五个测量/AP/协方差脚本的 token 库导入，不改变其计算行为。
- [x] 补迁移此前被忽略规则隐藏的 `tests/test_mcf.py` 导入，并将两份相关测试纳入可跟踪范围。
- [x] 验证：97 个新增测试、10 个原有数值测试通过；三种 CLI 输出与直接计算完全一致；原 n6 输出、旧 canonical token 文本及七个脚本无参帮助通过检查。
- [x] 记录运行方式、全部输入/输出/日志/临时目录及硬链接隔离处理。

已确认：方法 token 使用 `n<k>`（等邻居数密度）、`gn<k>`（等邻居数梯度）、`r<radius>`（固定支撑半径密度）；完整 token 统一为 `<方法token>_weipow<alpha>`，保留 `n30_weipow...`。同方法的联合幂指数继续使用 `<方法token>_weipow<a1>_<a2>_...`，保留顺序。

生成入口只根据方法 token 分流并输出原始量，测量入口负责构造 mark 和应用幂指数。`gn<k>` 当前保存 7 列位置、rho 和三分量原始梯度；密度两种方法为 4 列。2026-10-26 后 gn 默认在测量读入环节除以配套 rho，可用 --no-divide-by-rho 关闭。首轮短 token 固定默认自身/平滑约定；生成 CLI 明确拒绝非默认 self/h 参数，底层 marks API 保持可用。线程数和分块大小不进入方法 token。

阶段二范围：仅修改当前 LSS_python 项目，当时测量脚本仅迁移导入，方法参数和统计接入留给阶段三；其他项目由用户手动迁移。阶段二曾发现两个脚本有项目外硬链接，已恢复关联链接在该轮开始前的内容并隔离本项目副本，处理记录已保留。

接口及用法见 [token 使用说明](doc/weight_tokens.md)；本轮完整文件清单和验证记录见 [阶段二记录](validation/weight_refactor_stage2/README.md)。

## 阶段三：接入非 AP 测量脚本（已完成）

- [x] 盘点 box、MPI box、mock、普通/MPI 协方差共五个测量入口，并保存修改前快照；修改前确认本轮可编辑文件均为单链接，无需额外隔离。
- [x] 迁移普通/MPI 测量和协方差脚本，统一 --weight-method/--weight-token/--weight-power，保存声明的方法及幂次；默认 n30 保留为兼容默认值。
- [x] 迁移 mock 入口，保留基础观测权重与 random 权重的原有语义；在读入边界组合为四列，避免原始 mark 被重复相乘。
- [x] 查明附近 Fortran 的标量定义并接入 gn 三分量目录：先计算 `sqrt(gx**2+gy**2+gz**2)`，然后应用 token 幂指数。
- [x] 验证方法与完整 token 冲突会报错；幂运算只执行一次，random/base 不应用 data 的幂指数；n/r 和已经准备好的标量 gn mark 可用。
- [x] 验证 AP 脚本和 marks 数值模块的内容保持不变，更新运行记录及完整文件清单。

首次适配时查明：`LSS_grad_fields.f90:350` 和 `LSS_chisq.f90:198` 均取三个分量的模长，当时支持 Cartesian 六列向量和 mock 七列向量。该段属于原始取模接入历史；当前默认流程及新输入格式见下方“梯度模除密度”。mock base/random 仍不应用 data 幂指数。

验证进展：协议与端到端/旧 MPI 用例有 135 项通过；另 1 项仅因新增元数据字段触发旧字段集合断言，更新为允许四个明确的元数据键后单独复查通过；另加的 2 项向量防误读检查通过，合计 138 项检查通过对应验证。实际运行了本机双进程 MPI。首次环境内 MPI 通信受限的失败日志及所有重跑日志保留。

本次取模接入后，一次完整运行得到 167 passed，包含三种方法、五个实际测量入口、从生成文件到测量的链路和双进程 MPI 回归；另有先行读入单元检查 54 passed。没有失败或跳过的最终用例。AP、生成脚本和梯度计算内核内容保持不变。

阶段三首次适配的历史见 [阶段三记录](validation/weight_refactor_stage3/README.md)；当前工作流程、实际输入/输出/日志/缓存及完整绝对路径清单见 [梯度取模接入记录](validation/weight_refactor_gradient_norm/README.md)。

范围：暂不处理 AP 的 token 接入、AP 计算行为、DD/DR 缓存以及性能优化。

阶段一的工作流程、复查命令、验证结果和完整文件清单见 [阶段一记录](validation/weight_refactor_stage1/README.md)。
临时验证目录受现有 Git 忽略规则管理，文件均已保留；没有删除基准或日志。

## 后续调整：生成文件命名（已完成）

- [x] 将 create_rho 第三个位置参数改为可选；默认在输入目录保存 `<输入stem>_<当前方法token>.npy`。
- [x] 显式输出补齐 .npy；检查文件名中的完整当前方法 token，缺少时必须输入 y/yes 才继续；父目录中的 token 不计入。
- [x] 其他回答、EOF 或 Ctrl-C 在读入数据前取消；--force 不跳过名称确认；已有输出且未给 --force 时仍直接跳过。
- [x] 126 项 token/生成测试、10 项普通/MPI 测量链路测试、7 次实际命令行检查通过；完整记录见 [输出命名验证](validation/create_rho_output_naming/README.md)。

`n30_loginv_weipow<alpha>` 的命名已讨论，本轮尚未接入 loginv 数值变换或扩展 token 语法；此次仅完成生成文件名和确认规则。

## 后续调整：梯度模除密度（已完成，2026-10-26）

- [x] 保持 n/gn 的 KDTree 路径与 r 的 freud 路径分离，所有原有密度/梯度核函数不变。
- [x] create_drho 返回同一次查询得到的 `(rho, gradient)`；生成脚本保存七列 `[x,y,z,rho,gx,gy,gz]`。
- [x] 五个非 AP 测量入口默认使用 `(norm(gradient)/rho)**alpha`；新增 --no-divide-by-rho / Python divide_by_rho=False 使用原始梯度模。
- [x] 同步 mock 八列布局 `[RA,DEC,R,base,rho,gx,gy,gz]`，保持 base/random 的原有语义；已有单列 gn mark 由调用方按所选模式预先准备。
- [x] 原六列 Cartesian / 七列 mock 梯度文件只允许显式 raw 模式；默认模式缺 rho 时明确报错，要求重生成，归一化 rho 必须有限且大于零。
- [x] gn 结果新增 gradient_divide_by_rho 元数据；token 文字保持 gn<k>_weipow<alpha>，该开关作为附加计算约定记录，比较两种模式应使用不同输出路径。
- [x] 216 项非 CLI 检查及 18 项 CLI/集成检查通过，包括五个入口两种 gn 模式和本机 MPI；AP 与计数/协方差数值模块未修改。

当前实现、运行方式和完整输入/输出/日志/缓存清单见 [梯度归一化记录](validation/gradient_density_normalization/README.md)。早先六列取模和命名验证目录保留作为历史记录。
