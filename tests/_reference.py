"""测试专用的**独立**数学参考。

本模块刻意不 import dqskin：期望值全部用这里的 4x4 矩阵运算、
轴角公式和经典线性混合蒙皮（LBS）重新推导，避免“用被测核心生成期望值”。
仅用于测试。
"""

from __future__ import annotations

import math


def axis_angle_matrix(axis, angle, translation=(0.0, 0.0, 0.0)):
    """罗德里格斯公式直接生成刚体齐次矩阵（不经四元数）。"""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / n, ay / n, az / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return (
        (c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s,
         translation[0]),
        (ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s,
         translation[1]),
        (az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C,
         translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def mat_vec(m, v):
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def mat_dir(m, v):
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


def inverse_rigid(m):
    """独立写出的刚体逆（测试参考用，与 dqskin 内部实现无共享代码）。"""
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    t = [m[0][3], m[1][3], m[2][3]]
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            out[i][j] = r[j][i]
        out[i][3] = -(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2])
    out[3][3] = 1.0
    return tuple(tuple(row) for row in out)


def world_transforms(bones, attr):
    """按父链累乘世界矩阵；bones 为 (local_bind, local_current, parent) 三元组。

    ``attr`` 取 0(bind) 或 1(current)。独立递归实现。
    """
    n = len(bones)
    result = [None] * n

    def w(i):
        if result[i] is not None:
            return result[i]
        local = bones[i][attr]
        p = bones[i][2]
        result[i] = local if p is None else mat_mul(w(p), local)
        return result[i]

    return tuple(w(i) for i in range(n))


def lbs_relative_matrices(bones):
    """经典参考：T_i = current_world * inverse(bind_world)。"""
    wb = world_transforms(bones, 0)
    wc = world_transforms(bones, 1)
    return tuple(mat_mul(wc[i], inverse_rigid(wb[i])) for i in range(len(bones)))


def linear_blend_skin(vertices, bones_spec, weight_lists, normals=None):
    """经典矩阵线性混合蒙皮（LBS）独立参考：M* = sum w T_i / sum w。

    与双四元数算法完全不同的代码路径，用于“同向骨骼结果一致”等比对。
    """
    rels = lbs_relative_matrices(bones_spec)
    out_v, out_n = [], []
    for vi, v in enumerate(vertices):
        wl = weight_lists[vi]
        total = sum(w for _, w in wl)
        m = [[0.0] * 4 for _ in range(4)]
        for bone_i, w in wl:
            t = rels[bone_i]
            for r in range(4):
                for c in range(4):
                    m[r][c] += w * t[r][c]
        for r in range(4):
            for c in range(4):
                m[r][c] /= total
        m = tuple(tuple(row) for row in m)
        out_v.append(mat_vec(m, v))
        if normals is not None:
            out_n.append(mat_dir(m, normals[vi]))
    return out_v, out_n
