"""公开数据结构与 :func:`skin` 主入口（双四元数线性混合蒙皮）。

公开契约（见 README“接口”一节）：

* :class:`RigidTransform` —— 不可变刚体变换，4x4 行主序齐次矩阵；
* :class:`Bone` —— 一根骨骼：绑定姿态局部刚体变换、当前局部刚体变换、父索引；
* :class:`SkinResult` —— 蒙皮输出：变换后的位置（必有）与法向（输入有则有）；
* :func:`skin` —— 唯一核心算法入口。

错误语义：所有契约违例抛出 :class:`dqskin.errors.SkinningError` 的子类，
消息带可定位前缀；``bool`` 冒充整数/数字抛 ``TypeError``；``NaN``/``Infinity``
一律拒绝，不静默纠正。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from . import _linalg as la
from .errors import (
    RigidTransformError,
    SkeletonError,
    SkinningError,
    VertexDataError,
    WeightError,
)

MAX_BONES = 32


# --------------------------------------------------------------------------- #
# 公开数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RigidTransform:
    """刚体齐次变换。

    ``matrix`` 为 4x4 行主序嵌套元组/列表::

        ( R00 R01 R02 tx )
        ( R10 R11 R12 ty )
        ( R20 R21 R22 tz )
        (   0   0   0  1 )

    构造时按固定容差（见 ``dqskin._linalg``）校验：末行必须为
    ``[0,0,0,1]``，3x3 部分必须正交且行列式为 +1（拒绝剪切、各向异性缩放、
    反射）。
    """

    matrix: tuple

    def __post_init__(self) -> None:
        # dataclass(frozen=True) 下用 object.__setattr__ 写入归一化后的矩阵。
        m = _validate_rigid_matrix(self.matrix, "RigidTransform.matrix")
        object.__setattr__(self, "matrix", m)


@dataclass(frozen=True)
class Bone:
    """一根骨骼。

    ``bind`` / ``current`` 都是**局部**刚体变换（相对父骨骼；根骨骼相对世界）。
    ``parent`` 为父骨骼索引，根骨骼取 ``None``。支持多根（森林）。
    """

    bind: RigidTransform
    current: RigidTransform
    parent: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.bind, RigidTransform):
            raise TypeError(
                f"Bone.bind: expected RigidTransform, "
                f"got {type(self.bind).__name__}"
            )
        if not isinstance(self.current, RigidTransform):
            raise TypeError(
                f"Bone.current: expected RigidTransform, "
                f"got {type(self.current).__name__}"
            )
        p = self.parent
        if p is not None:
            if isinstance(p, bool) or not isinstance(p, int):
                raise TypeError(
                    "Bone.parent: expected int or None, "
                    f"got {type(p).__name__}"
                )
            if p < 0:
                # 范围上界需要骨骼总数，在 skin() 构建层级时报告；
                # 负数在这里即可定位。
                raise SkeletonError(
                    f"Bone.parent: negative parent index {p} is not allowed"
                )
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(
                f"Bone.name: expected str or None, got "
                f"{type(self.name).__name__}"
            )


@dataclass(frozen=True)
class SkinResult:
    """:func:`skin` 的结果；所有向量为不可变元组。"""

    vertices: tuple
    normals: Optional[tuple] = None


# --------------------------------------------------------------------------- #
# 基础取值校验
# --------------------------------------------------------------------------- #


def _as_finite_number(
    v: object, where: str, exc: type = SkinningError
) -> float:
    """把输入转成 float；bool 拒绝（int 子类），NaN/Infinity 拒绝。"""
    if isinstance(v, bool):
        raise TypeError(f"{where}: bool is not accepted as a number")
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise exc(f"{where}: value is NaN or Infinity")
        return v
    raise TypeError(f"{where}: expected int or float, got {type(v).__name__}")


def _as_int(v: object, where: str) -> int:
    if isinstance(v, bool):
        raise TypeError(f"{where}: bool is not accepted as an integer")
    if isinstance(v, int):
        return v
    raise TypeError(f"{where}: expected int, got {type(v).__name__}")


def _as_finite_vec3(v: object, where: str) -> tuple:
    if isinstance(v, (str, bytes)) or not hasattr(v, "__len__"):
        raise VertexDataError(f"{where}: expected a 3-element sequence")
    if len(v) != 3:
        raise VertexDataError(f"{where}: expected 3 components, got {len(v)}")
    return tuple(
        _as_finite_number(v[i], f"{where}[{i}]", VertexDataError)
        for i in range(3)
    )


def _validate_rigid_matrix(m: object, where: str) -> tuple:
    if not hasattr(m, "__len__"):
        raise RigidTransformError(f"{where}: expected a 4x4 matrix sequence")
    if len(m) != 4:
        raise RigidTransformError(
            f"{where}: expected 4 rows, got {len(m)}"
        )
    rows = []
    for i in range(4):
        row = m[i]
        if not hasattr(row, "__len__"):
            raise RigidTransformError(f"{where}[{i}]: expected a row sequence")
        if len(row) != 4:
            raise RigidTransformError(
                f"{where}[{i}]: expected 4 columns, got {len(row)}"
            )
        rows.append(
            tuple(
                _as_finite_number(
                    row[j], f"{where}[{i}][{j}]", RigidTransformError
                )
                for j in range(4)
            )
        )

    # 末行 [0, 0, 0, 1]
    for j, want in enumerate((0.0, 0.0, 0.0, 1.0)):
        if abs(rows[3][j] - want) > la.RIGID_LAST_ROW_TOL:
            raise RigidTransformError(
                f"{where}: last row must be (0, 0, 0, 1), got "
                f"({rows[3][0]:g}, {rows[3][1]:g}, {rows[3][2]:g}, "
                f"{rows[3][3]:g})"
            )

    # 3x3 部分列正交归一：R^T R = I
    cols = [tuple(rows[r][c] for r in range(3)) for c in range(3)]
    for a in range(3):
        for b in range(a, 3):
            dot = cols[a][0] * cols[b][0] + cols[a][1] * cols[b][1] \
                + cols[a][2] * cols[b][2]
            want = 1.0 if a == b else 0.0
            if abs(dot - want) > la.RIGID_ORTHO_TOL:
                raise RigidTransformError(
                    f"{where}: rotation part is not orthonormal "
                    f"(R^T R[{a}][{b}] = {dot!r}, expected {want})"
                )
    det = (
        cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
        - cols[1][0] * (cols[0][1] * cols[2][2] - cols[0][2] * cols[2][1])
        + cols[2][0] * (cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1])
    )
    if det <= 0.0 or abs(det - 1.0) > la.RIGID_SCALE_TOL:
        raise RigidTransformError(
            f"{where}: rotation determinant is {det!r}; a rigid transform "
            f"requires determinant +1 (reflections and scales are rejected)"
        )
    return tuple(rows)


# --------------------------------------------------------------------------- #
# 骨骼树
# --------------------------------------------------------------------------- #


def _build_world_transforms(bones: Sequence[Bone]) -> tuple:
    """校验父索引/环，并把每根骨骼的 bind/current 局部变换累乘为世界变换。

    返回 ``(world_bind, world_current)``，各为 N 个 4x4 矩阵的元组。
    """
    n = len(bones)
    parents = []
    for i, bone in enumerate(bones):
        p = bone.parent
        if p is None:
            parents.append(None)
        else:
            p = _as_int(p, f"bones[{i}].parent")
            if not (0 <= p < n):
                raise SkeletonError(
                    f"bones[{i}].parent: index {p} out of range "
                    f"(skeleton has {n} bone(s))"
                )
            if p == i:
                raise SkeletonError(
                    f"bones[{i}].parent: bone cannot be its own parent"
                )
            parents.append(p)

    world_bind: list = [None] * n
    world_cur: list = [None] * n
    # 0 = 未访问, 1 = 在当前递归栈中, 2 = 已完成
    state = [0] * n

    def resolve(i: int, chain: tuple) -> None:
        if state[i] == 2:
            return
        if state[i] == 1:
            cycle = chain[chain.index(i):] + (i,)
            path = " -> ".join(f"bones[{k}]" for k in cycle)
            raise SkeletonError(f"cycle detected in bone hierarchy: {path}")
        state[i] = 1
        p = parents[i]
        if p is None:
            wb = bones[i].bind.matrix
            wc = bones[i].current.matrix
        else:
            resolve(p, chain + (i,))
            wb = la.mat_mul(world_bind[p], bones[i].bind.matrix)
            wc = la.mat_mul(world_cur[p], bones[i].current.matrix)
        world_bind[i] = wb
        world_cur[i] = wc
        state[i] = 2

    for i in range(n):
        resolve(i, ())
    return tuple(world_bind), tuple(world_cur)


# --------------------------------------------------------------------------- #
# 权重
# --------------------------------------------------------------------------- #


def _parse_vertex_weights(
    w: object, n_bones: int, vi: int
) -> tuple:
    """解析单顶点权重，返回按骨骼索引排序的 ``((bone, weight), ...)``。

    接受 ``{bone_index: weight}`` 映射或 ``(bone_index, weight)`` 对的序列。
    """
    where = f"weights[{vi}]"
    pairs: list = []
    if isinstance(w, dict):
        iterator = w.items()
    elif hasattr(w, "__len__") and not isinstance(w, (str, bytes)):
        iterator = w
    else:
        raise WeightError(
            f"{where}: expected a dict {{bone_index: weight}} or a sequence "
            f"of (bone_index, weight) pairs"
        )

    seen = set()
    for k, pair in enumerate(iterator):
        pair_w = f"{where}[{k}]"
        if not hasattr(pair, "__len__") or len(pair) != 2:
            raise WeightError(
                f"{pair_w}: expected a (bone_index, weight) pair"
            )
        bone_idx_raw, weight_raw = pair[0], pair[1]
        bone_idx = _as_int(bone_idx_raw, f"{pair_w}.bone")
        if not (0 <= bone_idx < n_bones):
            raise WeightError(
                f"{pair_w}.bone: index {bone_idx} out of range "
                f"(skeleton has {n_bones} bone(s))"
            )
        if bone_idx in seen:
            raise WeightError(f"{pair_w}.bone: duplicate bone index {bone_idx}")
        seen.add(bone_idx)
        weight = _as_finite_number(
            weight_raw, f"{pair_w}.weight", WeightError
        )
        if weight < 0.0:
            raise WeightError(
                f"{pair_w}.weight: negative weight {weight!r} is not allowed"
            )
        pairs.append((bone_idx, weight))

    if not pairs:
        raise WeightError(f"{where}: no bone weights given")
    pairs.sort(key=lambda p: p[0])
    total = sum(weight for _, weight in pairs)
    if total <= 0.0:
        raise WeightError(
            f"{where}: total weight is {total!r}; at least one positive "
            f"weight is required (zero total weight rejected)"
        )
    return tuple(pairs)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def skin(
    vertices: Sequence,
    bones: Sequence[Bone],
    weights: Sequence,
    normals: Optional[Sequence] = None,
) -> SkinResult:
    """双四元数线性混合蒙皮（DLB / dual-quaternion blending）。

    流程：
    1. 校验全部输入（刚体性、骨骼树、有限数、权重）；
    2. 父链累乘得到每根骨骼 bind/current 的**世界**变换；
    3. 相对变换 ``T_i = current_world_i @ inverse(bind_world_i)`` 转单位双四元数；
    4. 逐顶点：以最大权重骨骼为符号基准，其余骨骼实部内积为负则双部同翻转，
       加权求和，正交化为单位双四元数；
    5. 用该双四元数变换位置（R p + t）和法向（R n）。

    :param vertices: N 个三维位置序列；
    :param bones: 1..32 根 :class:`Bone`；
    :param weights: 与顶点等长，每项为 ``{bone_index: weight}`` 或
        ``(bone_index, weight)`` 对序列；权重非负且总和大于 0；
    :param normals: 可选，与顶点等长的三维法向序列；
    :returns: :class:`SkinResult`。
    """
    # --- 骨骼：基本类型 / 数量 / 刚体性 -----------------------------------
    if not hasattr(bones, "__len__"):
        raise SkeletonError("bones: expected a sequence of Bone")
    n = len(bones)
    if not (1 <= n <= MAX_BONES):
        raise SkeletonError(
            f"bones: count must be in 1..{MAX_BONES}, got {n}"
        )
    for i, bone in enumerate(bones):
        if not isinstance(bone, Bone):
            raise TypeError(
                f"bones[{i}]: expected Bone, got {type(bone).__name__}"
            )
        # RigidTransform 的 dataclass 已在构造时校验；这里防御外部绕过构造器。
        if not isinstance(bone.bind, RigidTransform):
            raise TypeError(f"bones[{i}].bind: expected RigidTransform")
        if not isinstance(bone.current, RigidTransform):
            raise TypeError(f"bones[{i}].current: expected RigidTransform")

    # --- 顶点 / 法向形状 ---------------------------------------------------
    if not hasattr(vertices, "__len__"):
        raise VertexDataError("vertices: expected a sequence of 3-vectors")
    n_verts = len(vertices)
    if not hasattr(weights, "__len__"):
        raise WeightError("weights: expected a sequence aligned with vertices")
    if len(weights) != n_verts:
        raise WeightError(
            f"weights: got {len(weights)} entries for {n_verts} vertices; "
            f"each vertex needs one weight entry"
        )
    if n_verts == 0:
        raise VertexDataError("vertices: at least one vertex is required")
    parsed_vertices = tuple(
        _as_finite_vec3(v, f"vertices[{i}]") for i, v in enumerate(vertices)
    )
    parsed_normals = None
    if normals is not None:
        if not hasattr(normals, "__len__"):
            raise VertexDataError("normals: expected a sequence of 3-vectors")
        if len(normals) != n_verts:
            raise VertexDataError(
                f"normals: got {len(normals)} entries for {n_verts} vertices"
            )
        parsed_normals = tuple(
            _as_finite_vec3(v, f"normals[{i}]")
            for i, v in enumerate(normals)
        )

    # 纯函数：所有权重先解析成功后才进入计算，失败时没有任何对外状态变更。
    parsed_weights = tuple(
        _parse_vertex_weights(w, n, i) for i, w in enumerate(weights)
    )

    # --- 世界变换 / 相对双四元数 ------------------------------------------
    world_bind, world_cur = _build_world_transforms(bones)
    rel_dqs = tuple(
        la.matrix_to_dual_quaternion(
            la.mat_mul(world_cur[i], la.mat_inverse_rigid(world_bind[i]))
        )
        for i in range(n)
    )

    # --- 逐顶点混合与变换 --------------------------------------------------
    out_v: list = []
    out_n: list = []
    for vi in range(n_verts):
        pairs = parsed_weights[vi]
        total = sum(w for _, w in pairs)
        anchor = max(range(len(pairs)), key=lambda k: pairs[k][1])
        anchor_dq = rel_dqs[pairs[anchor][0]]
        blended_parts = []
        norm_w = []
        for bone_idx, w in pairs:
            dq = rel_dqs[bone_idx]
            if bone_idx != pairs[anchor][0] and la.dq_real_dot(dq, anchor_dq) < 0.0:
                dq = la.dq_neg(dq)
            blended_parts.append(dq)
            norm_w.append(w / total)
        unit_dq = la.dq_normalize(la.dq_weighted_sum(
            tuple(blended_parts), tuple(norm_w)
        ))
        q0, translation = la.dq_to_transform(unit_dq)
        p = parsed_vertices[vi]
        rp = la.quat_rotate(q0, p)
        out_v.append((
            rp[0] + translation[0],
            rp[1] + translation[1],
            rp[2] + translation[2],
        ))
        if parsed_normals is not None:
            out_n.append(la.quat_rotate(q0, parsed_normals[vi]))

    return SkinResult(
        vertices=tuple(out_v),
        normals=tuple(out_n) if parsed_normals is not None else None,
    )
