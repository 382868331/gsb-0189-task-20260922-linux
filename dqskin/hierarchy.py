"""按节点名接收局部刚体变换，拓扑排序求全局姿态，并接入旧蒙皮入口。

本模块是对 core.skin 的迭代扩展（旧的下标式 skin 入口保持不变）：

- 输入用 **节点名** 而不是下标描述父子关系：PoseNode(name, parent, local)；
- resolve_global_poses 做 Kahn 拓扑排序（多张量就绪时按输入顺序出列，
  结果确定），沿父链 W = W_parent * L 累乘出每根骨骼的全局姿态；
- skin_named 把全局姿态拍平成 Bone 列表（每根骨骼以“根”的身份携带
  其全局变换），交给旧的 core.skin 完成双四元数蒙皮。

四元数规范化规则（固定，二选一，由 keyword-only 参数 normalize 控制）：
- QUAT_NORMALIZE_REJECT（默认）：|‖q‖-1| <= QNORM_TOL 才接受且原样使用，
  超差抛 NonRigidTransformError，不做静默归一化（与旧 skin 一致）；
- QUAT_NORMALIZE_NORMALIZE：各分量先有限性校验，再按 q̃ = q/‖q‖ 显式
  归一化；零四元数（‖q‖=0）无法定方向，抛 InvalidInputError。

所有校验先于计算；本模块全部为纯函数、不改写输入，失败不会留下半更新状态。
"""

from __future__ import annotations

import collections.abc as _abc
from typing import NamedTuple, Sequence

from .core import (
    Bone,
    HierarchyError,
    InvalidInputError,
    NonRigidTransformError,
    QNORM_TOL,
    Quat,
    MAX_BONES,
    RigidTransform,
    SkinResult,
    WeightError,
    _compose,
    _parse_quat,
    _parse_real,
    _parse_vec3,
    skin,
)

__all__ = [
    "PoseNode",
    "GlobalPose",
    "resolve_global_poses",
    "skin_named",
    "QUAT_NORMALIZE_REJECT",
    "QUAT_NORMALIZE_NORMALIZE",
]

#: 默认规则：四元数必须单位（容差 QNORM_TOL），超差拒绝，不静默归一化。
QUAT_NORMALIZE_REJECT = "reject"

#: 显式归一化规则：q -> q / ‖q‖；零四元数拒绝。
QUAT_NORMALIZE_NORMALIZE = "normalize"

_VALID_MODES = (QUAT_NORMALIZE_REJECT, QUAT_NORMALIZE_NORMALIZE)


class PoseNode(NamedTuple):
    """命名骨架节点。

    name   —— 骨架内唯一、可哈希且非 None 的节点标识（str/int/tuple 均可）。
    parent —— 父节点名（根节点为 None）。
    local  —— 相对父节点（根节点相对世界）的局部刚体变换 RigidTransform。
    """

    name: object
    parent: object  # None 或可哈希节点名；运行期严格校验
    local: RigidTransform


class GlobalPose(NamedTuple):
    """拓扑排序与全局姿态累乘结果。

    order      —— 节点名的拓扑序（父必先于子；同层按输入顺序，确定性）。
    transforms —— name -> 全局刚体变换 W（W_root = L_root，
                  W_i = W_parent * L_i）。
    """

    order: tuple
    transforms: dict


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------


def _check_mode(mode: object) -> str:
    if mode not in _VALID_MODES:
        raise InvalidInputError(
            f"normalize: 仅支持 {_VALID_MODES!r}，得到 {mode!r}"
        )
    return mode


def _check_name(value: object, where: str) -> object:
    if value is None:
        raise InvalidInputError(f"{where}: 节点名不能为 None")
    if not isinstance(value, _abc.Hashable):
        raise InvalidInputError(
            f"{where}: 节点名必须可哈希，得到 {type(value).__name__}"
        )
    return value


def _parse_local(value: object, where: str, mode: str) -> RigidTransform:
    """解析局部刚体变换，并按固定规则处理旋转四元数模长。"""
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
    n = n2 ** 0.5
    if mode == QUAT_NORMALIZE_REJECT:
        if abs(n - 1.0) > QNORM_TOL:
            raise NonRigidTransformError(
                f"{where}.rotation: 旋转四元数非单位（|‖q‖-1| = "
                f"{abs(n - 1.0):.3e} > {QNORM_TOL:.1e}），拒绝非刚体变换"
            )
        return RigidTransform(q, t)
    # normalize：q̃ = q/‖q‖
    if n == 0.0:
        raise InvalidInputError(
            f"{where}.rotation: 零四元数无法规范化（normalize={mode!r}）"
        )
    inv = 1.0 / n
    return RigidTransform(Quat(q.w * inv, q.x * inv, q.y * inv, q.z * inv), t)


def _parse_pose_nodes(nodes: object, mode: str) -> list[PoseNode]:
    if not isinstance(nodes, (tuple, list)):
        raise InvalidInputError(f"nodes: 需要序列，得到 {type(nodes).__name__}")
    n = len(nodes)
    if n == 0:
        raise HierarchyError("nodes: 至少需要 1 个节点")
    if n > MAX_BONES:
        raise HierarchyError(f"nodes: 节点数 {n} 超过上限 {MAX_BONES}")

    parsed: list[PoseNode] = []
    seen: set = set()
    for i, node in enumerate(nodes):
        where = f"nodes[{i}]"
        if isinstance(node, PoseNode):
            name, parent, local = node.name, node.parent, node.local
        elif isinstance(node, (tuple, list)) and len(node) == 3:
            name, parent, local = node
        elif isinstance(node, dict):
            try:
                name = node["name"]
                parent = node["parent"]
                local = node["local"]
            except KeyError as exc:
                raise InvalidInputError(f"{where}: 缺少键 {exc}") from None
        else:
            raise InvalidInputError(
                f"{where}: 需要 PoseNode、(name, parent, local) 或 dict，得到 {node!r}"
            )
        name = _check_name(name, f"{where}.name")
        if name in seen:
            raise HierarchyError(f"{where}.name: 节点名 {name!r} 重复绑定")
        seen.add(name)
        if parent is not None:
            _check_name(parent, f"{where}.parent")
        parsed.append(
            PoseNode(
                name=name,
                parent=parent,
                local=_parse_local(local, f"{where}.local", mode),
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# 拓扑排序 + 全局姿态
# ---------------------------------------------------------------------------


def resolve_global_poses(
    nodes: Sequence[object],
    *,
    normalize: str = QUAT_NORMALIZE_REJECT,
) -> GlobalPose:
    """对命名骨架做拓扑排序并求每个节点的全局刚体变换。

    参数
    ----
    nodes     : 长度 1..32 的 PoseNode / (name, parent, local) / dict 序列。
                节点名在本次绑定中唯一；parent 为 None（根）或另一个已绑
                定节点的名字；森林（多根）合法。
    normalize : 四元数规范化规则，'reject'（默认）或 'normalize'。

    返回
    ----
    GlobalPose(order, transforms)：拓扑序节点名元组与 name -> 全局变换
    字典。W_root = L_root，W_i = W_parent * L_i。

    异常
    ----
    HierarchyError          —— 空/超量、重复绑定、父名不存在（非法父节点）、
                               父链成环。
    NonRigidTransformError —— normalize='reject' 时四元数非单位。
    InvalidInputError       —— 结构/类型非法、NaN/Inf、零四元数（normalize 模式）。
    """
    mode = _check_mode(normalize)
    parsed = _parse_pose_nodes(nodes, mode)
    by_name = {node.name: node for node in parsed}

    # 非法父节点：父名必须在本次绑定的节点集合中（None 表示根）。
    for node in parsed:
        if node.parent is not None and node.parent not in by_name:
            raise HierarchyError(
                f"节点 {node.name!r}: 非法父节点 {node.parent!r}（未在节点列表中绑定）"
            )

    # Kahn 拓扑排序；多张量就绪时严格按输入顺序选取，保证确定性。
    worlds: dict = {}
    order: list = []
    remaining = list(parsed)
    while remaining:
        progressed = False
        for node in remaining[:]:  # 切片：保持输入顺序扫描并允许移除
            if node.parent is None or node.parent in worlds:
                if node.parent is None:
                    world = node.local
                else:
                    world = _compose(worlds[node.parent], node.local)
                worlds[node.name] = world
                order.append(node.name)
                remaining.remove(node)
                progressed = True
        if not progressed:
            cycle = [node.name for node in remaining]
            raise HierarchyError(f"nodes: 父链存在环，环上节点至少包括 {cycle!r}")

    return GlobalPose(order=tuple(order), transforms=worlds)


# ---------------------------------------------------------------------------
# 命名权重解析
# ---------------------------------------------------------------------------


def _parse_named_weights(
    weights: object, nverts: int, names: Sequence[object]
) -> list[list[tuple[object, float]]]:
    if not isinstance(weights, (tuple, list)):
        raise InvalidInputError(f"weights: 需要序列，得到 {type(weights).__name__}")
    if len(weights) != nverts:
        raise InvalidInputError(
            f"weights: 长度 {len(weights)} 与顶点数 {nverts} 不一致"
        )
    known = set(names)
    parsed_rows: list[list[tuple[object, float]]] = []
    for i, row in enumerate(weights):
        where = f"weights[{i}]"
        if not isinstance(row, (tuple, list)):
            raise InvalidInputError(f"{where}: 需要 (bone_name, weight) 序列")
        if len(row) == 0:
            raise WeightError(f"{where}: 空权重行（总权为零）")
        entries: list[tuple[object, float]] = []
        seen: set = set()
        total = 0.0
        for j, item in enumerate(row):
            iwhere = f"{where}[{j}]"
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise InvalidInputError(
                    f"{iwhere}: 需要 (bone_name, weight) 二元组，得到 {item!r}"
                )
            name = item[0]
            if name is None or not isinstance(name, _abc.Hashable):
                raise InvalidInputError(
                    f"{iwhere}[0]: 骨骼名必须为非 None 的可哈希标识，得到 {name!r}"
                )
            if name not in known:
                raise WeightError(
                    f"{iwhere}[0]: 未知骨骼名 {name!r}（未在绑定节点中声明）"
                )
            if name in seen:
                raise WeightError(f"{iwhere}[0]: 骨骼名 {name!r} 在同一顶点内重复")
            seen.add(name)
            w = _parse_real(item[1], f"{iwhere}[1]")
            if w < 0.0:
                raise WeightError(f"{iwhere}[1]: 拒绝负权重 {w!r}")
            total += w
            entries.append((name, w))
        if total <= 0.0:
            raise WeightError(f"{where}: 总权重为零，无法归一化")
        parsed_rows.append(entries)
    return parsed_rows


# ---------------------------------------------------------------------------
# 新蒙皮入口
# ---------------------------------------------------------------------------


def skin_named(
    vertices: Sequence[object],
    bindings: Sequence[object],
    currents: Sequence[object],
    weights: Sequence[object],
    *,
    normalize: str = QUAT_NORMALIZE_REJECT,
) -> list[SkinResult]:
    """用命名骨架做双四元数蒙皮（旧 skin 的命名入口，算法不变）。

    参数
    ----
    vertices  : 同 core.skin：每项 (position, normal)，normal 非零。
    bindings  : 绑定姿态命名节点序列（PoseNode / 三元组 / dict）。
    currents  : 当前姿态命名节点序列；节点名集合必须与 bindings 完全一致，
                顺序可不同，拓扑关系以各自声明为准。
    weights   : 与顶点等长，第 i 项为 [(bone_name, weight), ...]；名字必须
                在绑定节点中声明，同顶点内不重复，权重有限非负、总权 > 0。
    normalize : 四元数规范化规则，'reject'（默认）或 'normalize'，对绑定
                与当前两套姿态同样生效。

    返回
    ----
    list[SkinResult]，与 core.skin 返回值完全相同。

    异常
    ----
    SkinError 子类（消息含定位）：层级/绑定问题抛 HierarchyError，非单位
    四元数抛 NonRigidTransformError，权重问题抛 WeightError，类型/NaN 抛
    InvalidInputError。纯函数，校验全部先于计算，失败不留半更新状态。
    """
    mode = _check_mode(normalize)

    if not isinstance(vertices, (tuple, list)):
        raise InvalidInputError(f"vertices: 需要序列，得到 {type(vertices).__name__}")
    if len(vertices) == 0:
        raise InvalidInputError("vertices: 至少需要 1 个顶点")

    # 两套姿态分别拓扑排序 + 全局化；节点集合必须一致，否则索引无法对应。
    bind_pose = resolve_global_poses(bindings, normalize=mode)
    cur_pose = resolve_global_poses(currents, normalize=mode)
    bind_names = set(bind_pose.transforms)
    cur_names = set(cur_pose.transforms)
    if bind_names != cur_names:
        only_bind = [n for n in bind_pose.order if n not in cur_names]
        only_cur = [n for n in cur_pose.order if n not in bind_names]
        raise HierarchyError(
            "bindings 与 currents 节点集合不一致："
            f"仅绑定姿态存在 {[repr(n) for n in only_bind]}，"
            f"仅当前姿态存在 {[repr(n) for n in only_cur]}"
        )

    parsed_weights = _parse_named_weights(weights, len(vertices), bind_pose.order)

    # 拍平：每个节点以“根”的身份携带其全局变换，交给旧 skin 入口。
    # 此时 W_bind/W_cur 直接是全局变换，M = G_cur * G_bind^{-1} 与
    # 沿父链累乘局部变换的结果逐骨相同。
    index = {name: i for i, name in enumerate(bind_pose.order)}
    flat_bones = [
        Bone(
            name=name,
            parent=None,
            bind=bind_pose.transforms[name],
            current=cur_pose.transforms[name],
        )
        for name in bind_pose.order
    ]
    flat_weights = [
        [(index[name], w) for name, w in row] for row in parsed_weights
    ]
    return skin(vertices, flat_bones, flat_weights)
