"""纯 Python 的 4x4 刚体矩阵 / 四元数 / 双四元数内部工具。

不对外公开（前导下划线的内部约定），仅服务于 :mod:`dqskin.core`。
全部用嵌套 ``tuple`` 表示不可变数据：

* 矩阵：4x4 ``tuple``，行主序；
* 四元数：``(w, x, y, z)``；
* 双四元数：``(q0, qe)``，两个四元数元组。
"""

from __future__ import annotations

import math

# 刚体校验用固定容差。这是输入契约的一部分（不是用来掩盖结果误差的输出容差）：
# 正交误差 / 尺度误差超过该值的矩阵会被拒绝。
RIGID_ORTHO_TOL = 1e-9
RIGID_SCALE_TOL = 1e-9
RIGID_LAST_ROW_TOL = 1e-12


def _isfinite_float(v: object) -> bool:
    return isinstance(v, float) and math.isfinite(v)


def mat_mul(a: tuple, b: tuple) -> tuple:
    """4x4 矩阵乘法 a @ b（行主序，列展开）。"""
    return tuple(
        tuple(
            a[i][0] * b[0][j]
            + a[i][1] * b[1][j]
            + a[i][2] * b[2][j]
            + a[i][3] * b[3][j]
            for j in range(4)
        )
        for i in range(4)
    )


def mat_inverse_rigid(m: tuple) -> tuple:
    """刚体矩阵的精确结构逆：R^T 与 -R^T t，O(1) 且无通用求逆误差放大。"""
    r = m
    return (
        (r[0][0], r[1][0], r[2][0],
         -(r[0][0] * r[0][3] + r[1][0] * r[1][3] + r[2][0] * r[2][3])),
        (r[0][1], r[1][1], r[2][1],
         -(r[0][1] * r[0][3] + r[1][1] * r[1][3] + r[2][1] * r[2][3])),
        (r[0][2], r[1][2], r[2][2],
         -(r[0][2] * r[0][3] + r[1][2] * r[1][3] + r[2][2] * r[2][3])),
        (0.0, 0.0, 0.0, 1.0),
    )


def quat_rotate(q: tuple, v: tuple) -> tuple:
    """单位四元数旋转向量 v：v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + w*v)。"""
    w, x, y, z = q
    vx, vy, vz = v
    # u = q.xyz
    # t = 2 * cross(u, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v + w*t + cross(u, t)
    cx = y * tz - z * ty
    cy = z * tx - x * tz
    cz = x * ty - y * tx
    return (vx + w * tx + cx, vy + w * ty + cy, vz + w * tz + cz)


def matrix_to_quaternion(m: tuple) -> tuple:
    """刚体旋转部分 -> 单位四元数 (w, x, y, z)，Shepperd 方法，180 度附近稳定。

    与“先取 trace 再开根号”的简化做法不同，Shepperd 方法按对角元最大的轴
    分支计算，旋转角接近 180 度（trace 接近 -1）时不会退化。
    """
    m00, m01, m02 = m[0][0], m[0][1], m[0][2]
    m10, m11, m12 = m[1][0], m[1][1], m[1][2]
    m20, m21, m22 = m[2][0], m[2][1], m[2][2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0  # s = 4w
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0  # s = 4x
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0  # s = 4y
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0  # s = 4z
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def matrix_to_dual_quaternion(m: tuple) -> tuple:
    """刚体齐次矩阵 -> 单位双四元数 (q0, qe)。

    q0 为旋转；qe = 0.5 * qt * q0，qt = (0, tx/2, ty/2, tz/2) 为纯平移双四元数，
    约定与 Kavan 2008《Dual Quaternions for Rigid Transformation Blending》一致。
    """
    q0 = matrix_to_quaternion(m)
    tx, ty, tz = m[0][3], m[1][3], m[2][3]
    w0, x0, y0, z0 = q0
    # qe = 0.5 * Q((0, t), q0) = Q((0, t/2), q0)
    we = -0.5 * (tx * x0 + ty * y0 + tz * z0)
    xe = 0.5 * (tx * w0 + ty * z0 - tz * y0)
    ye = 0.5 * (-tx * z0 + ty * w0 + tz * x0)
    ze = 0.5 * (tx * y0 - ty * x0 + tz * w0)
    return (q0, (we, xe, ye, ze))


def dq_real_dot(a: tuple, b: tuple) -> float:
    """两个双四元数实部四元数的 4 维内积。"""
    qa = a[0]
    qb = b[0]
    return qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]


def dq_neg(d: tuple) -> tuple:
    """双部同时取反（双四元数 q 与 -q 表示同一刚体变换）。"""
    q0, qe = d
    return ((-q0[0], -q0[1], -q0[2], -q0[3]),
            (-qe[0], -qe[1], -qe[2], -qe[3]))


def dq_weighted_sum(dqs: tuple, weights: tuple) -> tuple:
    """``sum(w_i * dq_i)``，不做归一化（调用方负责）。"""
    a0 = [0.0, 0.0, 0.0, 0.0]
    ae = [0.0, 0.0, 0.0, 0.0]
    for (q0, qe), w in zip(dqs, weights):
        a0[0] += w * q0[0]
        a0[1] += w * q0[1]
        a0[2] += w * q0[2]
        a0[3] += w * q0[3]
        ae[0] += w * qe[0]
        ae[1] += w * qe[1]
        ae[2] += w * qe[2]
        ae[3] += w * qe[3]
    return ((a0[0], a0[1], a0[2], a0[3]),
            (ae[0], ae[1], ae[2], ae[3]))


def dq_normalize(blended: tuple) -> tuple:
    """把加权和正交化为单位双四元数。

    1. 实部归一化：q0 /= |q0|；
    2. 双部正交化：qe -= dot(qe, q0) * q0（双四元数单位条件 <q0,qe> = 0）；
    3. 双部缩放到单位双四元数规范长度（此处 |q0|=1，长度因子为 1，
       数学上 qe 不再缩放；保留通用形式 1/|q0| 以贴合 Kavan 论文公式）。

    符号一致的凸组合给出 |q0| > 0；近 180 度对向混合属于
    “双四元数蒙皮已知的最短路径反演”情形，见 README 限制说明。
    """
    q0, qe = blended
    n0 = math.sqrt(q0[0] * q0[0] + q0[1] * q0[1]
                   + q0[2] * q0[2] + q0[3] * q0[3])
    r0 = (q0[0] / n0, q0[1] / n0, q0[2] / n0, q0[3] / n0)
    dot = r0[0] * qe[0] + r0[1] * qe[1] + r0[2] * qe[2] + r0[3] * qe[3]
    re = (
        (qe[0] - dot * r0[0]) / n0,
        (qe[1] - dot * r0[1]) / n0,
        (qe[2] - dot * r0[2]) / n0,
        (qe[3] - dot * r0[3]) / n0,
    )
    return (r0, re)


def dq_to_transform(d: tuple) -> tuple:
    """单位双四元数 -> (旋转四元数 q0, 平移向量 t)。t = 2 * qe * q0^{-1}。"""
    q0, qe = d
    w0, x0, y0, z0 = q0
    we, xe, ye, ze = qe
    # q0^{-1} = (w0, -x0, -y0, -z0)（单位四元数共轭），t = 2 * qe * conj(q0)
    tx = 2.0 * (-we * x0 + xe * w0 - ye * z0 + ze * y0)
    ty = 2.0 * (-we * y0 + xe * z0 + ye * w0 - ze * x0)
    tz = 2.0 * (-we * z0 - xe * y0 + ye * x0 + ze * w0)
    return q0, (tx, ty, tz)
