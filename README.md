# dqskin — 双四元数骨骼蒙皮、绑定修正与命名骨架层级

离线角色变形工程的可复用模块：对一批顶点执行双四元数蒙皮
（Dual-Quaternion Skinning, DQS），含绑定修正。

- **Python 版本**：3.14.7（仅标准库，零第三方依赖，`requirements.txt` 为空注释）
- **运行环境**：Linux 原生；核心、测试、演示全部离线，无需网络/账号/数据库/Docker
- **示例数据**：全部由仓库内确定性代码生成

## 接口

```python
from dqskin import (
    skin, skin_named, resolve_global_poses,
    Bone, PoseNode, GlobalPose, RigidTransform, Quat, Vec3,
)

results = skin(vertices, bones, weights)                 # 下标式旧入口
results = skin_named(vertices, bindings, currents, weights, *, normalize="reject")
pose    = resolve_global_poses(nodes, *, normalize="reject")  # 拓扑序+全局姿态
```

### 数据结构

| 类型 | 字段 | 说明 |
|---|---|---|
| `Vec3` | `x, y, z` | 三维向量 |
| `Quat` | `w, x, y, z` | 单位四元数，`w` 为实部 |
| `RigidTransform` | `rotation: Quat, translation: Vec3` | 刚体变换：先旋转后平移，`p' = R·p + t` |
| `Bone` | `name, parent, bind, current` | 旧入口骨骼：`parent` 为父骨骼下标（int）或 `None`（根）；`bind`/`current` 为绑定/当前**局部**刚体变换 |
| `PoseNode` | `name, parent, local` | 命名骨架节点：`name` 为骨架内唯一、非 `None` 的可哈希标识；`parent` 为父节点名或 `None`；`local` 为相对父（根相对世界）的局部刚体变换。支持 `PoseNode`、`(name, parent, local)` 与等价 dict |
| `GlobalPose` | `order: tuple, transforms: dict` | `resolve_global_poses` 返回：拓扑序节点名与 `name -> 全局 RigidTransform` 字典 |
| `SkinResult` | `position: Vec3, normal: Vec3` | 蒙皮输出 |

`vertices`：每项 `(position, normal)`，长度 3 的有限数值序列；`normal` 必须非零。
旧入口 `bones`：每项 `Bone`、`(name, parent, bind, current)` 元组或等价 dict。
旧入口 `weights`：与顶点等长，第 i 项为 `[(bone_index, weight), ...]`；权重有限、非负、
骨骼下标不重复、每行总权 > 0（**不要求和为 1**，内部归一化，缩放不变）。
命名入口 `weights`：第 i 项为 `[(bone_name, weight), ...]`，权重规则相同，
骨骼名必须在绑定节点中声明且同顶点内不重复。

### 命名骨架与拓扑排序（本次新增）

`resolve_global_poses(nodes, *, normalize=...)`：

1. 校验并收集节点：名字非 `None`、可哈希、**单次绑定内唯一**（重复绑定拒绝）；
2. 父节点合法性：`parent` 为 `None` 或已绑定节点名，否则拒绝（非法父节点）；
3. **Kahn 拓扑排序**：多张量就绪时按输入顺序出列（结果确定、可复现），
   扫描结束仍有剩余节点即存在环并拒绝（含自环）；
4. 按拓扑序累乘全局姿态：`W_root = L_root`，`W_i = W_parent · L_i`。

`skin_named(vertices, bindings, currents, weights, *, normalize=...)` 对绑定与当前
两套命名姿态分别拓扑排序、求全局姿态；两套姿态的**节点名集合必须一致**（顺序可不同，
各自声明父链），然后把全局变换拍平为以全局变换为根局部变换的 `Bone` 列表交给旧
`skin` 入口——此时 `M = G_cur · G_bind⁻¹` 与沿父链累乘局部变换逐骨相同，蒙皮算法不变。

### 四元数规范化规则（固定，二选一）

由 keyword-only 参数 `normalize` 控制，绑定与当前姿态同样生效：

| 取值 | 规则 |
|---|---|
| `"reject"`（`QUAT_NORMALIZE_REJECT`，默认） | `|‖q‖−1| ≤ QNORM_TOL` 才接受且原样使用，超差抛 `NonRigidTransformError`，不做静默归一化（与旧 `skin` 完全一致） |
| `"normalize"`（`QUAT_NORMALIZE_NORMALIZE`） | 各分量先做有限性校验，再按 `q̃ = q/‖q‖` 显式归一化；零四元数（`‖q‖=0`）无方向，抛 `InvalidInputError` |

未识别的 `normalize` 取值抛 `InvalidInputError`。

### 算法流程（与实现一一对应）

1. （命名入口）Kahn 拓扑排序，沿父链累乘局部变换得全局变换；
2. 蒙皮变换 `M_i = W_current_i · W_bind_i⁻¹`（绑定修正）；
3. `M_i` 转单位双四元数 `dq = qr + ε·qd`，`qd = ½(0,t)⊗qr`；
4. 逐顶点：以**最大权重骨骼**的实部为符号基准，其余骨骼实部内积为负则
   实部与对偶部**同号翻转**；加权求和；
5. 单位双四元数正交化（归一化实部并扣除对偶部漂移，强制 `<qr,qd>=0`）；
6. 用结果变换位置（旋转+平移）与法向（仅旋转，不平移）。

## 输入上限

- 骨骼/节点数：**1–32**（`MAX_BONES = 32`），层级为森林（允许多根），禁止环与自环；
- 顶点数：无硬上限（受内存限制）；
- 所有数值必须为有限 `int/float`：**拒绝 NaN/Infinity**；
- 整数参数（父下标、骨骼下标）**拒绝 bool**；权重为数值但也拒绝 bool；
- 旧入口旋转四元数必须单位：`|‖q‖−1| ≤ QNORM_TOL = 1e-6`，超差即拒绝（非刚体），
  不做静默归一化；命名入口默认规则相同，可显式选择 `normalize="normalize"`。

## 错误语义

所有错误继承 `SkinError`（同时是 `ValueError`），消息含定位路径
（如 `bones[0].current.rotation`、`nodes[1].name`、`weights[0][1][1]`）：

| 异常 | 触发条件 |
|---|---|
| `InvalidInputError`（也是 `TypeError`） | 类型/结构非法、NaN/Inf、bool 充当数值、零法向、形状不符、节点名为 `None`/不可哈希、`normalize` 取值非法、归一化模式下零四元数 |
| `HierarchyError` | 空骨架、>32 骨/节点、父下标越界、父名不存在（非法父节点）、自环、父链成环、节点名重复绑定、绑定/当前节点集合不一致 |
| `NonRigidTransformError` | `normalize="reject"` 下旋转四元数非单位（超出 `QNORM_TOL`） |
| `WeightError` | 负权、零总权、空权重行、骨骼下标/名字越界或未知、同顶点内重复 |

**原子性**：全部校验先于任何计算；任一输入非法即抛异常，不返回部分结果。
所有入口均为纯函数、无内部状态、不改写输入，失败不会留下部分变更。
本模块不含搜索/证明流程，因此不存在“预算耗尽”与“无解”的分支。

## 误差口径

- 双精度浮点计算；测试容差：解析参考 `1e-9`，180° 附近混合 `1e-6`；
- 权重整体缩放（正数）结果不变（测试验证到 `1e-9`）；
- 法向长度在蒙皮前后守恒（纯旋转）；
- `QNORM_TOL` 是**输入合法性**口径，不是输出误差口径；
- `normalize="normalize"` 仅修正书写模长，旋转方向不变。

## 设计取舍与已知限制

- 符号基准取最大权重骨骼（并列取下标小者，确定性）；零权重骨骼不参与符号判断；
- 混合是“最短路径”球面混合：两骨骼旋转相差恰好 180° 时，等权混合的实部
  理论上退化到范数为零附近，结果对权重微扰敏感（DQS 固有限制，非实现缺陷）；
- 每顶点骨骼影响数未设上限（受 32 骨总数约束）；
- 拓扑排序同层按输入顺序 tie-break，结果确定但不承诺其他拓扑序；
- 命名入口要求绑定/当前姿态节点集合一致，不支持仅在当前姿态新增/删除骨骼；
- 只实现本题契约的刚体+DQS 子集：**不做动画曲线插值、不做非刚体缩放、
  不实现混合蒙皮模式**，不宣称兼容任何完整行业标准（如 glTF 蒙皮）；
- 不实现动画编辑器、不读写文件格式。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 演示（约 0.03 秒）
```

`demo.py` 真实计算并打印：乱序输入的三层骨架经 Kahn 拓扑排序后，父节点 +90°
旋转把子节点局部偏移 `(0,2,0)`/`(0,3,0)` 带到 `(-2,0,0)`/`(-3,0,0)`（手算核对）；
扭转区域（两骨 ±150° 等权）DQS 半径保持 ≈1 而 LBS 对照塌陷到 ≈0.866；
非单位四元数在 reject/normalize 两种规则下的不同处理；以及实际触发的
环、非法父节点、重复绑定与负权重拒绝。

测试期望值来自解析手算与 `tests/refimpl.py` 中独立的矩阵实现
（不调用 `dqskin`），往返一致性仅作补充。
