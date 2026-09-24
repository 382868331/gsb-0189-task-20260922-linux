"""dqskin 演示：扭转区域双四元数蒙皮 vs 线性混合、局部姿态 -> 全局姿态
-> 蒙皮的新入口，以及错误输入拒绝。

固定输入、确定性计算、离线运行（约 1 秒内完成）。
"""

from __future__ import annotations

import math

from dqskin import (
    Bone,
    HierarchyError,
    PoseNode,
    Quat,
    RigidTransform,
    Vec3,
    WeightError,
    compute_global_pose,
    skin,
    skin_with_pose,
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

    # ==== 新能力：局部姿态 + 父节点 -> 拓扑排序求全局姿态 -> 旧蒙皮入口 ====
    print()
    print("=== 局部姿态入口：三层骨架（root -> mid -> leaf）===")
    ident_q = Quat(1.0, 0.0, 0.0, 0.0)

    def shift_x(d):
        return RigidTransform(ident_q, Vec3(d, 0.0, 0.0))

    # 绑定姿态：三节骨架沿 x 轴各平移 1（leaf 绑定世界位置 (2,0,0)）
    bind_pose = [
        PoseNode("root", None, shift_x(0.0)),
        PoseNode("mid", "root", shift_x(1.0)),
        PoseNode("leaf", "mid", shift_x(1.0)),
    ]
    # 当前姿态：root 绕 z 轴转 90°，子节点局部变换不变
    current_pose = [
        PoseNode("root", None,
                 RigidTransform(quat_axis_angle((0, 0, 1), math.pi / 2), Vec3(0, 0, 0))),
        PoseNode("mid", "root", shift_x(1.0)),
        PoseNode("leaf", "mid", shift_x(1.0)),
    ]

    globals_cur = compute_global_pose(current_pose)
    print("拓扑序（父先于子）:", " -> ".join(globals_cur))
    for name, g in globals_cur.items():
        t = g.translation
        print(f"  {name:4s} 全局位置 = ({t.x:+.6f}, {t.y:+.6f}, {t.z:+.6f})")

    # 蒙皮：leaf 末端顶点随父节点旋转从 (2,0,0) 到 (0,2,0)
    res = skin_with_pose(
        [((2.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        bind_pose,
        current_pose,
        [[(2, 1.0)]],
    )
    p = res[0].position
    print(f"leaf 末端顶点蒙皮结果 = ({p.x:+.6f}, {p.y:+.6f}, {p.z:+.6f})")

    # ---- 边界：输入顺序打乱（子先于父），拓扑排序后结果一致 ----
    print()
    print("=== 边界：输入顺序打乱（leaf 最先）===")
    shuffled = [bind_pose[2], bind_pose[0], bind_pose[1]]
    g2 = compute_global_pose(shuffled)
    t = g2["leaf"].translation
    print(f"打乱后拓扑序: {' -> '.join(g2)}；leaf 绑定全局位置 = "
          f"({t.x:+.6f}, {t.y:+.6f}, {t.z:+.6f})")

    # ---- 拒绝：重复绑定与环 ----
    print()
    print("=== 拒绝边界：重复绑定与环 ===")
    try:
        compute_global_pose([PoseNode("a", None, bind_pose[0].local),
                             PoseNode("a", None, bind_pose[0].local)])
        print("（意外）未触发拒绝")
    except HierarchyError as exc:
        print(f"已拒绝 -> HierarchyError: {exc}")
    try:
        compute_global_pose([PoseNode("a", "b", bind_pose[0].local),
                             PoseNode("b", "a", bind_pose[0].local)])
        print("（意外）未触发拒绝")
    except HierarchyError as exc:
        print(f"已拒绝 -> HierarchyError: {exc}")


if __name__ == "__main__":
    main()
