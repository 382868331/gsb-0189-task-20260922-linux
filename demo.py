"""dqskin 演示：扭转区域双四元数蒙皮 vs 线性混合、错误输入拒绝，
以及权重敏感度（位置对未归一化权重的解析局部导数）。

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
    weight_jacobian,
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


def _fmt(v):
    return f"({v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f})"


def demo_dqs_vs_lbs():
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
    print(f"DQS 位置 = {_fmt(res.position)}")
    print(f"DQS 法向 = {_fmt(res.normal)}")
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


def demo_sensitivity():
    """权重敏感度：三骨不等权（平移+旋转混合，含父子层级），解析导数
    对 skin 黑箱中心差分，并验证缩放零方向。"""
    print()
    print("=== 权重敏感度：三骨不等权（旋转+平移混合）===")
    ident = RigidTransform(Quat(1, 0, 0, 0), Vec3(0, 0, 0))
    bones = [
        Bone("b0", None, ident,
             RigidTransform(quat_axis_angle((0, 0, 1), 0.5), Vec3(1.0, 0.2, -0.3))),
        Bone("b1", 0, ident,
             RigidTransform(quat_axis_angle((1, 0, 1), -0.8), Vec3(0.0, 2.0, 0.5))),
        Bone("b2", 1, ident,
             RigidTransform(quat_axis_angle((0, 1, 0), 1.1), Vec3(-0.4, 0.0, 1.0))),
    ]
    vertex = ((0.4, -0.7, 0.9), (0.0, 0.0, 1.0))
    row = [(0, 0.5), (1, 0.3), (2, 0.2)]
    pos0 = skin([vertex], bones, [row])[0].position
    print(f"当前蒙皮位置 = {_fmt(pos0)}")

    sens = weight_jacobian(vertex, bones, row)
    assert sens.differentiable, sens.reason

    h = 1e-6
    worst = 0.0
    for d, (bidx, w) in zip(sens.derivatives, row):
        rp = [(b, w + (h if b == bidx else 0.0)) for b, w in row]
        rm = [(b, w - (h if b == bidx else 0.0)) for b, w in row]
        pp = skin([vertex], bones, [rp])[0].position
        pm = skin([vertex], bones, [rm])[0].position
        cd = tuple((pp[i] - pm[i]) / (2 * h) for i in range(3))
        worst = max(worst, max(abs(d.d_position[i] - cd[i]) for i in range(3)))
        print(f"∂p/∂w[{bidx}] 解析 = {_fmt(d.d_position)}   中心差分 = {_fmt(cd)}")
    print(f"解析 vs 中心差分最大偏差: {worst:.2e}（h={h:g}）")

    # 缩放零方向：Σ wₖ ∂p/∂wₖ 必须为零向量
    null_axis = [0.0, 0.0, 0.0]
    for d, (_, w) in zip(sens.derivatives, row):
        for i in range(3):
            null_axis[i] += w * d.d_position[i]
    print(f"缩放零方向 Σ wₖ∂p/∂wₖ = {_fmt(null_axis)}（应为零向量）")

    # 一阶预测：权重 w0 增大 0.02、w2 减小 0.02，位置线性外推 vs 实际
    moved = [(0, 0.52), (1, 0.3), (2, 0.18)]
    pos1 = skin([vertex], bones, [moved])[0].position
    pred = tuple(
        pos0[i] + 0.02 * sens.derivatives[0].d_position[i]
        - 0.02 * sens.derivatives[2].d_position[i]
        for i in range(3)
    )
    print(f"权重微调后实际位置 = {_fmt(pos1)}")
    print(f"一阶预测位置       = {_fmt(pred)}（偏差 {max(abs(pos1[i]-pred[i]) for i in range(3)):.2e}）")

    # ---- 不可微边界：权重平局 ----
    print()
    print("=== 敏感度边界 1：最大权重平局 ===")
    tie = weight_jacobian(vertex, bones, [(0, 0.5), (1, 0.5), (2, 0.0)])
    print(f"differentiable = {tie.differentiable}")
    print(f"reason = {tie.reason}")

    # ---- 不可微边界：对齐点积为零（恒等骨 vs 绕 y 180° 骨）----
    print()
    print("=== 敏感度边界 2：对齐点积为零（翻转切换）===")
    flip_bones = [
        Bone("a", None, ident, ident),
        Bone("b", None, ident,
             RigidTransform(quat_axis_angle((0, 1, 0), math.pi), Vec3(0, 0, 0))),
    ]
    flip = weight_jacobian(vertex, flip_bones, [(0, 0.7), (1, 0.3)])
    print(f"differentiable = {flip.differentiable}")
    print(f"reason = {flip.reason}")


def main() -> None:
    demo_dqs_vs_lbs()
    demo_sensitivity()


if __name__ == "__main__":
    main()
