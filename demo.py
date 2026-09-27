"""dqskin 演示：扭转区域双四元数蒙皮 vs 线性混合，以及错误输入拒绝。

固定输入、确定性计算、离线运行（约 1 秒内完成）。
"""

from __future__ import annotations

import math

from dqskin import Bone, Quat, RigidTransform, Vec3, WeightError, skin


def quat_axis_angle(axis, angle):
    n = math.sqrt(sum(c * c for c in axis))
    h = angle / 2.0
    s = math.sin(h) / n
    return Quat(math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s)


def lbs_point(transforms, weights, p):
    """朴素线性混合蒙皮对照：q = Σ w_i (R_i p + t_i) / Σ w_i。"""
    total = sum(weights)
    out = [0.0, 0.0, 0.0]
    for (q, t), w in zip(transforms, weights):
        # 旋转矩阵（由单位四元数展开）
        wq, x, y, z = q
        R = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - wq * z), 2 * (x * z + wq * y)],
            [2 * (x * y + wq * z), 1 - 2 * (x * x + z * z), 2 * (y * z - wq * x)],
            [2 * (x * z - wq * y), 2 * (y * z + wq * x), 1 - 2 * (x * x + y * y)],
        ]
        for i in range(3):
            out[i] += (w / total) * (sum(R[i][j] * p[j] for j in range(3)) + t[i])
    return out


def main() -> None:
    ident = RigidTransform(Quat(1.0, 0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    # ---- 场景：两段骨骼绕手臂纵轴（x 轴）反向扭转 ----
    # 绑定姿态均为恒等；当前姿态：骨 0 转 +150°，骨 1 转 -150°。
    # 中间顶点等权混合：DQS 应得 ~180° 纯旋转（半径保持），
    # LBS 会把点拉向轴线（“糖果纸”塌陷）。
    twist = math.radians(150.0)
    bones = [
        Bone("upper", None, ident,
             RigidTransform(quat_axis_angle((1, 0, 0), twist), Vec3(0, 0, 0))),
        Bone("lower", None, ident,
             RigidTransform(quat_axis_angle((1, 0, 0), -twist), Vec3(0, 0, 0))),
    ]
    # 顶点位于轴外半径 1 处（x=0 截面上 y=1），法向沿 +y
    vertices = [((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))]
    weights = [[(0, 0.5), (1, 0.5)]]

    (res,) = skin(vertices, bones, weights)
    dqs_radius = math.hypot(res.position.y, res.position.z)

    lbs = lbs_point(
        [(bones[0].current.rotation, (0, 0, 0)),
         (bones[1].current.rotation, (0, 0, 0))],
        [0.5, 0.5],
        (0.0, 1.0, 0.0),
    )
    lbs_radius = math.hypot(lbs[1], lbs[2])

    print("=== 正常结果：扭转区域（两骨 ±150°，等权）===")
    print(f"DQS 位置 = ({res.position.x:+.6f}, {res.position.y:+.6f}, {res.position.z:+.6f})")
    print(f"DQS 法向 = ({res.normal.x:+.6f}, {res.normal.y:+.6f}, {res.normal.z:+.6f})")
    print(f"DQS 切线 = ({res.tangent.x:+.6f}, {res.tangent.y:+.6f}, {res.tangent.z:+.6f})")
    print(f"DQS 半径（应≈1，不塌陷）: {dqs_radius:.6f}")
    print(f"LBS 位置 = ({lbs[0]:+.6f}, {lbs[1]:+.6f}, {lbs[2]:+.6f})")
    print(f"LBS 半径（塌陷对照）     : {lbs_radius:.6f}")

    # ---- 边界：实际触发一个拒绝 ----
    print()
    print("=== 拒绝边界：负权重 ===")
    try:
        skin(vertices, bones, [[(0, 1.2), (1, -0.2)]])
        print("（意外）未触发拒绝")
    except WeightError as exc:
        print(f"已拒绝 -> WeightError: {exc}")


if __name__ == "__main__":
    main()
