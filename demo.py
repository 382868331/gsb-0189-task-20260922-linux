"""dqskin 演示：扭转区域双四元数蒙皮 vs 线性混合、权重敏感度查询，以及错误输入拒绝。

固定输入、确定性计算、离线运行（约 1 秒内完成）。
"""

from __future__ import annotations

import math

from dqskin import (
    Bone,
    Quat,
    RigidTransform,
    Vec3,
    WeightError,
    skin,
    skin_weight_sensitivity,
)


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
    vertices = [((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))]
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

    # ---- 权重敏感度：三骨骼（旋转+平移混合，不等权）----
    print()
    print("=== 权重敏感度：三骨骼（旋转+平移，不等权 0.5/0.3/0.2）===")
    bones3 = [
        Bone("b0", None, ident,
             RigidTransform(quat_axis_angle((0, 0, 1), 0.9), Vec3(1.0, 0.0, 0.0))),
        Bone("b1", None, ident,
             RigidTransform(quat_axis_angle((0, 1, 0), -0.7), Vec3(0.0, 2.0, 1.0))),
        Bone("b2", None, ident,
             RigidTransform(quat_axis_angle((1, 0, 0), 0.4), Vec3(-1.0, 0.5, 2.0))),
    ]
    pos3 = (0.3, -0.7, 1.1)
    w3 = [(0, 0.5), (1, 0.3), (2, 0.2)]
    sens = skin_weight_sensitivity(pos3, bones3, w3)
    p = sens.position
    print(f"位置 = ({p.x:+.6f}, {p.y:+.6f}, {p.z:+.6f})（与 skin 输出一致）")
    print("雅可比 ∂position/∂w_j（对未归一化原始权重）:")
    for j, col in enumerate(sens.jacobian):
        print(f"  J[{j}] = ({col.x:+.6f}, {col.y:+.6f}, {col.z:+.6f})")

    # 中心差分独立核对（安全邻域，h = 1e-6）
    h = 1e-6
    max_err = 0.0
    for j in range(len(w3)):
        wp = [list(e) for e in w3]
        wm = [list(e) for e in w3]
        wp[j][1] += h
        wm[j][1] -= h
        pp = skin([(pos3, (0, 0, 1))], bones3, [wp])[0].position
        pm = skin([(pos3, (0, 0, 1))], bones3, [wm])[0].position
        for k in range(3):
            cd = (pp[k] - pm[k]) / (2.0 * h)
            max_err = max(max_err, abs(cd - sens.jacobian[j][k]))
    print(f"中心差分核对：最大偏差 {max_err:.3e}（应 < 1e-6）")

    # 全体权重同倍缩放方向的方向导数 Σ w_j·J[j] 应为零
    euler = [sum(w3[j][1] * sens.jacobian[j][k] for j in range(3)) for k in range(3)]
    print(f"缩放方向导数 Σ w_j·J[j] = ({euler[0]:+.2e}, {euler[1]:+.2e}, {euler[2]:+.2e})（应≈0）")

    # ---- 切换边界：等权扭转顶点不可微 ----
    print()
    print("=== 切换边界：等权（0.5/0.5）扭转顶点 ===")
    edge = skin_weight_sensitivity((0.0, 1.0, 0.0), bones, [(0, 0.5), (1, 0.5)])
    print(f"differentiable = {edge.differentiable}")
    print(f"原因: {edge.nondifferentiable_reason}")
    ep = edge.position
    print(f"位置仍返回 = ({ep.x:+.6f}, {ep.y:+.6f}, {ep.z:+.6f})")


if __name__ == "__main__":
    main()
