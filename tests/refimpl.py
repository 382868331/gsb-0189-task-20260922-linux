"""测试用独立参考实现。

与被测核心 dqskin 完全独立：不 import dqskin，用轴角->旋转矩阵、
4x4 齐次矩阵乘法与求逆实现同一数学定义，用于交叉验证期望值。
解析期望值（手算）优先；本文件仅作为矩阵参考与线性蒙皮（LBS）对照。
"""

from __future__ import annotations

import math


def axis_angle_matrix(axis, angle):
    """Rodrigues 公式：轴角 -> 3x3 旋转矩阵（行主序嵌套列表）。"""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    x, y, z = ax / n, ay / n, az / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def mat_vec(R, v):
    return tuple(
        R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3)
    )


def mat_mul(A, B):
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat_transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def rigid_from_axis_angle(axis, angle, translation):
    """(R, t) 形式的刚体变换。"""
    return axis_angle_matrix(axis, angle), tuple(translation)


def rigid_compose(A, B):
    """A * B：先作用 B，再作用 A。变换为 (R, t) 对。"""
    Ra, ta = A
    Rb, tb = B
    R = mat_mul(Ra, Rb)
    t = tuple(mat_vec(Ra, tb)[i] + ta[i] for i in range(3))
    return R, t


def rigid_inverse(T):
    R, t = T
    Rt = mat_transpose(R)
    return Rt, tuple(-x for x in mat_vec(Rt, t))


def rigid_apply(T, p):
    R, t = T
    r = mat_vec(R, p)
    return tuple(r[i] + t[i] for i in range(3))


def rigid_rotate(T, v):
    R, _ = T
    return mat_vec(R, v)


def lbs_blend(transforms, weights, p):
    """朴素线性混合蒙皮（对照用）：直接线性混合各骨骼作用后的点。"""
    out = [0.0, 0.0, 0.0]
    total = sum(weights)
    for T, w in zip(transforms, weights):
        q = rigid_apply(T, p)
        for i in range(3):
            out[i] += (w / total) * q[i]
    return tuple(out)
