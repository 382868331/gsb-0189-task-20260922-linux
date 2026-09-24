"""双四元数骨骼蒙皮核心实现。

约定（与 README 一致）：
- 四元数 q = (w, x, y, z)，表示旋转，w 为实部。
- 刚体变换 RigidTransform(rotation=Quat, translation=Vec3)，先旋转后平移：
  p' = R * p + t。
- 骨骼世界变换 W = W_parent * L（父世界变换左乘局部变换），根骨骼 W = L。
- 蒙皮矩阵 M_i = W_current_i * inverse(W_bind_i)。
- 双四元数 dq = qr + eps * qd，qd = 0.5 * (0, t) ⊗ qr。
- 顶点混合：以最大权重骨骼的实部为符号基准，实部内积为负的骨骼整体翻转
  （实部与对偶部同号翻转），加权求和后单位化（正交化），再作用于位置与法向。

姿态节点入口（新增）：
- PoseNode(name, parent, local)：以名字标识的局部刚体变换 + 父节点名。
- compute_global_pose(nodes)：拓扑排序后沿父链累乘求全局姿态；
  固定的规范化规则——每次累乘后立即将旋转四元数单位化（除以其模长），
  不做符号翻转等其他处理。
- skin_with_pose(vertices, bind_pose, current_pose, weights)：分别求绑定/
  当前全局姿态，构造等价骨骼列表交给旧蒙皮入口 skin。

所有校验在计算前完成：任一输入非法时抛出异常，不返回部分结果。
"""

from __future__ import annotations

import math
from collections import deque
from typing import NamedTuple, Sequence

__all__ = [
    "Vec3",
    "Quat",
    "RigidTransform",
    "Bone",
    "PoseNode",
    "SkinResult",
    "SkinError",
    "InvalidInputError",
    "HierarchyError",
    "NonRigidTransformError",
    "WeightError",
    "skin",
    "compute_global_pose",
    "skin_with_pose",
    "QNORM_TOL",
    "MAX_BONES",
]

# 输入上限：骨骼树（森林）最多 32 根骨骼。
MAX_BONES = 32

# 单位四元数模长容差（相对）：|‖q‖ - 1| <= QNORM_TOL。
# 这是“输入合法性”的判定口径，不是输出误差口径；超差即拒绝，不做静默归一化。
QNORM_TOL = 1e-6


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


class Vec3(NamedTuple):
    """三维向量 (x, y, z)。"""

    x: float
    y: float
    z: float


class Quat(NamedTuple):
    """单位四元数 (w, x, y, z)，w 为实部。"""

    w: float
    x: float
    y: float
    z: float


class RigidTransform(NamedTuple):
    """刚体变换：先旋转 rotation，后平移 translation。"""

    rotation: Quat
    translation: Vec3


class Bone(NamedTuple):
    """骨骼。

    name    —— 任意可哈希标识，仅用于错误信息定位。
    parent  —— 父骨骼在 bones 中的下标（int），根骨骼为 None。
    bind    —— 绑定姿态下的局部刚体变换。
    current —— 当前姿态下的局部刚体变换。
    """

    name: object
    parent: object  # None 或 int；运行期严格校验
    bind: RigidTransform
    current: RigidTransform


class SkinResult(NamedTuple):
    """单个顶点的蒙皮结果。"""

    position: Vec3
    normal: Vec3


class PoseNode(NamedTuple):
    """姿态节点：以名字标识的局部刚体变换（新增能力）。

    name   —— 任意可哈希标识，同一姿态内唯一（重名即“重复绑定”，拒绝）。
    parent —— 父节点的 name，根节点为 None。
    local  —— 相对父节点的局部刚体变换；旋转四元数必须单位（QNORM_TOL）。
    """

    name: object
    parent: object  # None 或父节点的 name；运行期严格校验
    local: RigidTransform# ---------------------------------------------------------------------------
# 错误类型（均可定位：消息中带 bones[i] / vertices[i] / weights[i][j] 等路径）
# ---------------------------------------------------------------------------


class SkinError(ValueError):
    """本模块所有输入错误的基类（同时是 ValueError）。"""


class InvalidInputError(SkinError, TypeError):
    """类型/结构非法：非数值、NaN/Inf、bool 充当数值、形状错误等。"""


class HierarchyError(SkinError):
    """骨骼层级非法：空骨架、超过 32 根、父下标越界、存在环、多根等。"""


class NonRigidTransformError(SkinError):
    """旋转四元数非单位（超出 QNORM_TOL），即变换不是刚体变换。"""


class WeightError(SkinError):
    """权重非法：非数值、负权、重复骨骼下标、下标越界、总权为零等。"""


# ---------------------------------------------------------------------------
# 输入解析与校验
# ---------------------------------------------------------------------------


def _is_bool(v: object) -> bool:
    return isinstance(v, bool)


def _parse_real(value: object, where: str) -> float:
    """解析一个有限实数；拒绝 bool、非数值、NaN、Inf。"""
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise InvalidInputError(f"{where}: 需要有限数值（int/float），得到 {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise InvalidInputError(f"{where}: 拒绝 NaN/Infinity，得到 {value!r}")
    return v


def _parse_index(value: object, where: str) -> int:
    """解析一个整数下标；拒绝 bool 与非 int。"""
    if _is_bool(value) or not isinstance(value, int):
        raise InvalidInputError(f"{where}: 需要 int 下标（拒绝 bool），得到 {value!r}")
    return value


def _parse_vec3(value: object, where: str) -> Vec3:
    if isinstance(value, Vec3):
        items = (value.x, value.y, value.z)
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        items = tuple(value)
    else:
        raise InvalidInputError(f"{where}: 需要长度为 3 的序列或 Vec3，得到 {value!r}")
    return Vec3(
        _parse_real(items[0], f"{where}[0]"),
        _parse_real(items[1], f"{where}[1]"),
        _parse_real(items[2], f"{where}[2]"),
    )


def _parse_quat(value: object, where: str) -> Quat:
    if isinstance(value, Quat):
        items = (value.w, value.x, value.y, value.z)
    elif isinstance(value, (tuple, list)) and len(value) == 4:
        items = tuple(value)
    else:
        raise InvalidInputError(f"{where}: 需要长度为 4 的序列或 Quat，得到 {value!r}")
    return Quat(
        _parse_real(items[0], f"{where}[0]"),
        _parse_real(items[1], f"{where}[1]"),
        _parse_real(items[2], f"{where}[2]"),
        _parse_real(items[3], f"{where}[3]"),
    )


def _parse_rigid(value: object, where: str) -> RigidTransform:
    if isinstance(value, RigidTransform):
        rot, tra = value.rotation, value.translation
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        rot, tra = value
    else:
        raise InvalidInputError(
            f"{where}: 需要 RigidTransform 或 (rotation, translation)，得到 {value!r}"
        )
    q = _parse_quat(rot, f"{where}.rotation")
    t = _parse_vec3(tra, f"{where}.translation")
    n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
    if abs(math.sqrt(n2) - 1.0) > QNORM_TOL:
        raise NonRigidTransformError(
            f"{where}.rotation: 旋转四元数非单位（|‖q‖-1| = "
            f"{abs(math.sqrt(n2) - 1.0):.3e} > {QNORM_TOL:.1e}），拒绝非刚体变换"
        )
    return RigidTransform(q, t)


def _parse_vertex(vertex: object, where: str) -> tuple[Vec3, Vec3]:
    if not (isinstance(vertex, (tuple, list)) and len(vertex) == 2):
        raise InvalidInputError(
            f"{where}: 需要 (position, normal) 二元组，得到 {vertex!r}"
        )
    p = _parse_vec3(vertex[0], f"{where}.position")
    n = _parse_vec3(vertex[1], f"{where}.normal")
    nlen = math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
    if nlen == 0.0:
        raise InvalidInputError(f"{where}.normal: 法向为零向量，无法蒙皮")
    return p, n


def _parse_bones(bones: object) -> list[Bone]:
    if not isinstance(bones, (tuple, list)):
        raise InvalidInputError(f"bones: 需要序列，得到 {type(bones).__name__}")
    nb = len(bones)
    if nb == 0:
        raise HierarchyError("bones: 至少需要 1 根骨骼")
    if nb > MAX_BONES:
        raise HierarchyError(f"bones: 骨骼数 {nb} 超过上限 {MAX_BONES}")

    parsed: list[Bone] = []
    for i, b in enumerate(bones):
        where = f"bones[{i}]"
        if isinstance(b, Bone):
            name, parent, bind, current = b.name, b.parent, b.bind, b.current
        elif isinstance(b, (tuple, list)) and len(b) == 4:
            name, parent, bind, current = b
        elif isinstance(b, dict):
            try:
                name = b["name"]
                parent = b["parent"]
                bind = b["bind"]
                current = b["current"]
            except KeyError as exc:
                raise InvalidInputError(f"{where}: 缺少键 {exc}") from None
        else:
            raise InvalidInputError(
                f"{where}: 需要 Bone、(name, parent, bind, current) 或 dict，得到 {b!r}"
            )
        if parent is None:
            pidx = None
        else:
            pidx = _parse_index(parent, f"{where}.parent")
            if not (0 <= pidx < nb):
                raise HierarchyError(
                    f"{where}.parent: 父下标 {pidx} 越界（合法范围 0..{nb - 1}）"
                )
            if pidx == i:
                raise HierarchyError(f"{where}.parent: 骨骼不能以自己为父（自环）")
        parsed.append(
            Bone(
                name=name,
                parent=pidx,
                bind=_parse_rigid(bind, f"{where}.bind"),
                current=_parse_rigid(current, f"{where}.current"),
            )
        )

    _check_hierarchy(parsed)
    return parsed


def _check_hierarchy(bones: list[Bone]) -> None:
    """森林结构校验：无环（父链必在 <=32 步内到达根）。"""
    nb = len(bones)
    for i in range(nb):
        seen = {i}
        j = bones[i].parent
        steps = 0
        while j is not None:
            if j in seen:
                raise HierarchyError(
                    f"bones: 从 bones[{i}] 沿父链检测到环（经过 bones[{j}]）"
                )
            seen.add(j)
            steps += 1
            if steps > nb:  # 防御：理论上 seen 已保证终止
                raise HierarchyError(f"bones: 父链异常（bones[{i}]）")
            j = bones[j].parent


def _parse_weights(weights: object, nverts: int, nbones: int) -> list[list[tuple[int, float]]]:
    if not isinstance(weights, (tuple, list)):
        raise InvalidInputError(f"weights: 需要序列，得到 {type(weights).__name__}")
    if len(weights) != nverts:
        raise InvalidInputError(
            f"weights: 长度 {len(weights)} 与顶点数 {nverts} 不一致"
        )
    parsed: list[list[tuple[int, float]]] = []
    for i, row in enumerate(weights):
        where = f"weights[{i}]"
        if not isinstance(row, (tuple, list)):
            raise InvalidInputError(f"{where}: 需要 (bone_index, weight) 序列")
        if len(row) == 0:
            raise WeightError(f"{where}: 空权重行（总权为零）")
        entries: list[tuple[int, float]] = []
        seen: set[int] = set()
        total = 0.0
        for j, item in enumerate(row):
            iwhere = f"{where}[{j}]"
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise InvalidInputError(
                    f"{iwhere}: 需要 (bone_index, weight) 二元组，得到 {item!r}"
                )
            bidx = _parse_index(item[0], f"{iwhere}[0]")
            if not (0 <= bidx < nbones):
                raise WeightError(
                    f"{iwhere}[0]: 骨骼下标 {bidx} 越界（合法范围 0..{nbones - 1}）"
                )
            if bidx in seen:
                raise WeightError(f"{iwhere}[0]: 骨骼下标 {bidx} 在同一顶点内重复")
            seen.add(bidx)
            w = _parse_real(item[1], f"{iwhere}[1]")
            if w < 0.0:
                raise WeightError(f"{iwhere}[1]: 拒绝负权重 {w!r}")
            total += w
            entries.append((bidx, w))
        if total <= 0.0:
            raise WeightError(f"{where}: 总权重为零，无法归一化")
        parsed.append(entries)
    return parsed


# ---------------------------------------------------------------------------
# 四元数 / 刚体变换内部运算（输入均已校验为单位四元数）
# ---------------------------------------------------------------------------


def _qmul(a: Quat, b: Quat) -> Quat:
    """Hamilton 积 a ⊗ b（先作用 b，后作用 a）。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return Quat(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qconj(q: Quat) -> Quat:
    return Quat(q.w, -q.x, -q.y, -q.z)


def _qrot(q: Quat, v: Vec3) -> Vec3:
    """用单位四元数旋转向量：q ⊗ (0,v) ⊗ q* 的向量部。"""
    # 展开为 v + 2*w*(qvec × v) + 2*(qvec × (qvec × v))
    qx, qy, qz = q.x, q.y, q.z
    # c1 = qvec × v
    c1x = qy * v.z - qz * v.y
    c1y = qz * v.x - qx * v.z
    c1z = qx * v.y - qy * v.x
    # c2 = qvec × c1
    c2x = qy * c1z - qz * c1y
    c2y = qz * c1x - qx * c1z
    c2z = qx * c1y - qy * c1x
    return Vec3(
        v.x + 2.0 * (q.w * c1x + c2x),
        v.y + 2.0 * (q.w * c1y + c2y),
        v.z + 2.0 * (q.w * c1z + c2z),
    )


def _compose(a: RigidTransform, b: RigidTransform) -> RigidTransform:
    """a * b：先作用 b，再作用 a。"""
    return RigidTransform(
        rotation=_qmul(a.rotation, b.rotation),
        translation=_vadd(_qrot(a.rotation, b.translation), a.translation),
    )


def _inverse(t: RigidTransform) -> RigidTransform:
    qc = _qconj(t.rotation)
    return RigidTransform(
        rotation=qc,
        translation=_qrot(qc, Vec3(-t.translation.x, -t.translation.y, -t.translation.z)),
    )


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z)


def _vscale(a: Vec3, s: float) -> Vec3:
    return Vec3(a.x * s, a.y * s, a.z * s)


# ---------------------------------------------------------------------------
# 双四元数
# ---------------------------------------------------------------------------


def _rigid_to_dq(t: RigidTransform) -> tuple[Quat, Quat]:
    """刚体变换 -> 单位双四元数 (qr, qd)，qd = 0.5 * (0,t) ⊗ qr。"""
    qr = t.rotation
    tq = Quat(0.0, t.translation.x, t.translation.y, t.translation.z)
    qd = _qmul(tq, qr)
    return qr, Quat(0.5 * qd.w, 0.5 * qd.x, 0.5 * qd.y, 0.5 * qd.z)


def _dq_normalize(qr: Quat, qd: Quat) -> tuple[Quat, Quat]:
    """单位双四元数正交化。

    先归一化实部（n = ‖qr‖），再从对偶部扣除实部方向上的漂移，
    强制 <qr, qd> = 0，保证结果是单位双四元数（对应一个刚体变换）。
    """
    n = math.sqrt(qr.w * qr.w + qr.x * qr.x + qr.y * qr.y + qr.z * qr.z)
    if n == 0.0:
        # 仅在权重全零时可能发生，而权重校验已拒绝该情形；防御性兜底。
        raise WeightError("内部错误：双四元数实部模长为零")
    inv = 1.0 / n
    qr = Quat(qr.w * inv, qr.x * inv, qr.y * inv, qr.z * inv)
    qd = Quat(qd.w * inv, qd.x * inv, qd.y * inv, qd.z * inv)
    dot = qr.w * qd.w + qr.x * qd.x + qr.y * qd.y + qr.z * qd.z
    qd = Quat(
        qd.w - dot * qr.w,
        qd.x - dot * qr.x,
        qd.y - dot * qr.y,
        qd.z - dot * qr.z,
    )
    return qr, qd


def _dq_translation(qr: Quat, qd: Quat) -> Vec3:
    """单位双四元数 -> 平移：t = 2 * qd ⊗ qr* 的向量部。"""
    p = _qmul(qd, _qconj(qr))
    return Vec3(2.0 * p.x, 2.0 * p.y, 2.0 * p.z)


# ---------------------------------------------------------------------------
# 姿态节点：局部变换 + 父节点 -> 拓扑排序 -> 全局姿态
# ---------------------------------------------------------------------------


def _check_hashable(value: object, where: str) -> None:
    try:
        hash(value)
    except TypeError:
        raise InvalidInputError(f"{where}: 需要可哈希标识，得到 {value!r}") from None


def _parse_pose_nodes(nodes: object, where: str) -> list[PoseNode]:
    """解析姿态节点序列；校验结构、数值与旋转单位性（拓扑校验在 _topo_sort）。"""
    if not isinstance(nodes, (tuple, list)):
        raise InvalidInputError(f"{where}: 需要序列，得到 {type(nodes).__name__}")
    if len(nodes) == 0:
        raise HierarchyError(f"{where}: 至少需要 1 个姿态节点")
    parsed: list[PoseNode] = []
    for i, nd in enumerate(nodes):
        w = f"{where}[{i}]"
        if isinstance(nd, PoseNode):
            name, parent, local = nd.name, nd.parent, nd.local
        elif isinstance(nd, (tuple, list)) and len(nd) == 3:
            name, parent, local = nd
        elif isinstance(nd, dict):
            try:
                name = nd["name"]
                parent = nd["parent"]
                local = nd["local"]
            except KeyError as exc:
                raise InvalidInputError(f"{w}: 缺少键 {exc}") from None
        else:
            raise InvalidInputError(
                f"{w}: 需要 PoseNode、(name, parent, local) 或 dict，得到 {nd!r}"
            )
        _check_hashable(name, f"{w}.name")
        if parent is not None:
            _check_hashable(parent, f"{w}.parent")
        parsed.append(
            PoseNode(name=name, parent=parent, local=_parse_rigid(local, f"{w}.local"))
        )
    return parsed


def _topo_sort(nodes: list[PoseNode], where: str) -> list[PoseNode]:
    """Kahn 拓扑排序（父先于子）；拒绝重复绑定、非法父节点、自环与环。

    顺序确定性：根按输入序入队，同一父节点的子节点按输入序出队。
    """
    first_index: dict[object, int] = {}
    for i, nd in enumerate(nodes):
        if nd.name in first_index:
            raise HierarchyError(
                f"{where}: 节点名 {nd.name!r} 重复绑定"
                f"（第 {first_index[nd.name]} 与第 {i} 项）"
            )
        first_index[nd.name] = i

    children: dict[object, list[PoseNode]] = {nd.name: [] for nd in nodes}
    roots: list[PoseNode] = []
    for i, nd in enumerate(nodes):
        if nd.parent is None:
            roots.append(nd)
            continue
        if nd.parent == nd.name:
            raise HierarchyError(
                f"{where}[{i}].parent: 节点不能以自己为父（自环）"
            )
        if nd.parent not in children:
            raise HierarchyError(
                f"{where}[{i}].parent: 父节点 {nd.parent!r} 不存在（非法父节点）"
            )
        children[nd.parent].append(nd)

    order: list[PoseNode] = []
    queue = deque(roots)
    while queue:
        nd = queue.popleft()
        order.append(nd)
        queue.extend(children[nd.name])
    if len(order) != len(nodes):
        done = {nd.name for nd in order}
        stuck = [nd.name for nd in nodes if nd.name not in done]
        raise HierarchyError(f"{where}: 父链存在环，涉及节点 {stuck!r}")
    return order


def _qnormalize(q: Quat) -> Quat:
    """单位化 q / ‖q‖（固定的规范化规则的唯一环节）。"""
    n = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    if n == 0.0:
        # 输入已校验为单位四元数，乘积不会坍缩到 0；防御性兜底。
        raise SkinError("内部错误：四元数模长为零，无法规范化")
    inv = 1.0 / n
    return Quat(q.w * inv, q.x * inv, q.y * inv, q.z * inv)


def _compose_global(parent: RigidTransform, local: RigidTransform) -> RigidTransform:
    """全局累乘 W = W_parent · L；旋转乘积立即单位化（固定规范化规则）。"""
    return RigidTransform(
        rotation=_qnormalize(_qmul(parent.rotation, local.rotation)),
        translation=_vadd(_qrot(parent.rotation, local.translation), parent.translation),
    )


def _globals_from_order(order: list[PoseNode]) -> dict[object, RigidTransform]:
    globals_: dict[object, RigidTransform] = {}
    for nd in order:
        globals_[nd.name] = (
            nd.local
            if nd.parent is None
            else _compose_global(globals_[nd.parent], nd.local)
        )
    return globals_


def compute_global_pose(
    nodes: Sequence[object],
) -> dict[object, RigidTransform]:
    """由局部姿态（局部刚体变换 + 父节点）求全局姿态。

    参数
    ----
    nodes : 姿态节点序列，每项 PoseNode、(name, parent, local) 或等价 dict。
        parent 为父节点的 name 或 None（根）。局部旋转四元数必须单位
        （容差 QNORM_TOL）。节点数本身不设上限（skin_with_pose 沿用
        skin 的 32 骨上限）。

    返回
    ----
    dict[name, RigidTransform]，按拓扑序（父先于子）插入；全局变换为世界
    系下的刚体变换。规范化规则固定：每次累乘后立即将旋转四元数单位化。

    异常
    ----
    重复绑定（重名）、非法父节点（不存在）、自环、环 -> HierarchyError；
    结构/数值非法 -> InvalidInputError；旋转非单位 -> NonRigidTransformError。
    全部校验先于计算，失败不返回部分结果。
    """
    parsed = _parse_pose_nodes(nodes, "pose")
    return _globals_from_order(_topo_sort(parsed, "pose"))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def skin(
    vertices: Sequence[object],
    bones: Sequence[object],
    weights: Sequence[object],
) -> list[SkinResult]:
    """对一批顶点做双四元数蒙皮。

    参数
    ----
    vertices : 长度 V 的序列，每项 (position, normal)；position/normal 为
        长度 3 的有限数值序列（或 Vec3）。normal 必须非零。
    bones    : 长度 1..32 的骨骼序列，每项 Bone 或
        (name, parent, bind, current)。parent 为父骨骼下标或 None（根）。
        bind/current 为 RigidTransform 或 (quat, vec3)；旋转四元数必须
        单位（容差 QNORM_TOL）。层级必须是森林（无环）。
    weights  : 长度 V 的序列，第 i 项是顶点 i 的 [(bone_index, weight), ...]。
        权重为有限非负数值，骨骼下标不重复，每行总权 > 0。

    返回
    ----
    list[SkinResult]，长度 V，与输入顶点一一对应。

    异常
    ----
    任一输入非法即抛出 SkinError 子类（消息含定位路径），不返回部分结果；
    本函数无内部状态，失败不会留下部分变更。
    """
    # ---- 全部校验先行 ----
    if not isinstance(vertices, (tuple, list)):
        raise InvalidInputError(f"vertices: 需要序列，得到 {type(vertices).__name__}")
    if len(vertices) == 0:
        raise InvalidInputError("vertices: 至少需要 1 个顶点")
    parsed_verts = [_parse_vertex(v, f"vertices[{i}]") for i, v in enumerate(vertices)]
    parsed_bones = _parse_bones(bones)
    parsed_weights = _parse_weights(weights, len(parsed_verts), len(parsed_bones))

    # ---- 世界变换（父链累乘）与蒙皮变换 M = W_cur * W_bind^{-1} ----
    nb = len(parsed_bones)

    def _worlds(attr: str) -> list[RigidTransform]:
        worlds: list[RigidTransform | None] = [None] * nb

        def world(i: int) -> RigidTransform:
            cached = worlds[i]
            if cached is not None:
                return cached
            bone = parsed_bones[i]
            local = getattr(bone, attr)
            w = local if bone.parent is None else _compose(world(bone.parent), local)
            worlds[i] = w
            return w

        for k in range(nb):
            world(k)
        return worlds  # type: ignore[return-value]

    bind_world = _worlds("bind")
    cur_world = _worlds("current")

    dqs: list[tuple[Quat, Quat]] = []
    for i in range(nb):
        m = _compose(cur_world[i], _inverse(bind_world[i]))
        dqs.append(_rigid_to_dq(m))

    # ---- 逐顶点混合 ----
    results: list[SkinResult] = []
    for vi, ((p, n), entries) in enumerate(zip(parsed_verts, parsed_weights)):
        # 符号基准：最大权重骨骼（并列取下标小者，确定性）
        ref_idx = max(entries, key=lambda e: (e[1], -e[0]))[0]
        ref_qr = dqs[ref_idx][0]

        acc_qr = [0.0, 0.0, 0.0, 0.0]
        acc_qd = [0.0, 0.0, 0.0, 0.0]
        for bidx, w in entries:
            if w == 0.0:
                continue  # 零权重骨骼不影响结果（也不参与符号判断）
            qr, qd = dqs[bidx]
            dot = (
                qr.w * ref_qr.w + qr.x * ref_qr.x + qr.y * ref_qr.y + qr.z * ref_qr.z
            )
            s = -1.0 if dot < 0.0 else 1.0
            acc_qr[0] += w * s * qr.w
            acc_qr[1] += w * s * qr.x
            acc_qr[2] += w * s * qr.y
            acc_qr[3] += w * s * qr.z
            acc_qd[0] += w * s * qd.w
            acc_qd[1] += w * s * qd.x
            acc_qd[2] += w * s * qd.y
            acc_qd[3] += w * s * qd.z

        uqr, uqd = _dq_normalize(Quat(*acc_qr), Quat(*acc_qd))
        t = _dq_translation(uqr, uqd)
        pos = _vadd(_qrot(uqr, p), t)
        nrm = _qrot(uqr, n)
        results.append(SkinResult(position=pos, normal=nrm))

    return results


def skin_with_pose(
    vertices: Sequence[object],
    bind_pose: Sequence[object],
    current_pose: Sequence[object],
    weights: Sequence[object],
) -> list[SkinResult]:
    """局部姿态入口：拓扑排序求全局姿态后交给旧蒙皮入口 skin。

    参数
    ----
    vertices     : 同 skin。
    bind_pose    : 绑定姿态节点序列（PoseNode / (name, parent, local) / dict）。
    current_pose : 当前姿态节点序列，节点名字集合必须与 bind_pose 一致；
        父结构允许与绑定姿态不同（各自独立求全局姿态）。
    weights      : 同 skin；骨骼下标对应 **bind_pose 的输入顺序**
        （第 i 个下标即 bind_pose[i] 的节点）。

    返回 / 异常
    ----
    同 skin。姿态自身的非法（重复绑定、非法父节点、环、非单位局部旋转、
    绑定/当前名字集合不一致）在调用 skin 之前抛出；骨骼数上限（32）由
    skin 继承。全部校验先于计算，失败不留下部分变更。
    """
    bind_nodes = _parse_pose_nodes(bind_pose, "bind_pose")
    cur_nodes = _parse_pose_nodes(current_pose, "current_pose")
    bind_globals = _globals_from_order(_topo_sort(bind_nodes, "bind_pose"))
    cur_globals = _globals_from_order(_topo_sort(cur_nodes, "current_pose"))

    bind_names = {nd.name for nd in bind_nodes}
    cur_names = {nd.name for nd in cur_nodes}
    if bind_names != cur_names:
        only_bind = sorted(repr(n) for n in bind_names - cur_names)
        only_cur = sorted(repr(n) for n in cur_names - bind_names)
        raise HierarchyError(
            "bind_pose/current_pose: 节点名字集合不一致"
            f"（仅绑定侧 {only_bind}，仅当前侧 {only_cur}）"
        )

    # 全局姿态作为无父骨骼交给旧入口：skin 内部求得的
    # M_i = W_current_i · W_bind_i⁻¹ 与全局语义一致。
    bones = [
        Bone(
            name=nd.name,
            parent=None,
            bind=bind_globals[nd.name],
            current=cur_globals[nd.name],
        )
        for nd in bind_nodes
    ]
    return skin(vertices, bones, weights)
