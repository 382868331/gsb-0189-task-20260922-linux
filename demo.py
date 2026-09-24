"""dqskin 演示。

固定输入、确定性计算、离线运行（1 秒内完成）。包含：
1. 命名骨架：拓扑排序求全局姿态，父节点旋转带动三层骨架的子节点位置；
2. 扭转区域双四元数蒙皮 vs 线性混合（旧能力）；
3. 四元数规范化边界：normalize='normalize' 接受非单位书写；
4. 实际触发的拒绝：环、非法父节点、重复绑定、负权重。
"""

from __future__ import annotations

import math

from dqskin import (
    Bone,
    HierarchyError,
    InvalidInputError,
    NonRigidTransformError,
    PoseNode,
    QUAT_NORMALIZE_NORMALIZE,
    Quat,
    RigidTransform,
    Vec3,
    WeightError,
    resolve_global_poses,
    skin,
    skin_named,
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

    # ---- 场景 1（新能力）：命名三层骨架，乱序输入，父旋转带动子节点 ----
    # 绑定姿态全恒等；当前姿态：root 绕 z +90°，spine 局部偏移 (0,2,0)，
    # hand 相对 spine 再偏移 (0,1,0)。手算 Rz90 把 +y 送到 -x：
    # spine 世界位置 (-2,0,0)，hand 世界位置 (-3,0,0)。
    print("=== 正常结果：三层骨架（输入乱序，Kahn 拓扑排序）父旋转带动子节点 ===")
    binds = [
        PoseNode("hand", "spine", ident),
        PoseNode("root", None, ident),
        PoseNode("spine", "root", ident),
    ]
    currents = [
        PoseNode("hand", "spine", RigidTransform(quat_axis_angle((0, 0, 1), 0.0),
                                                 Vec3(0.0, 1.0, 0.0))),
        PoseNode("root", None,
                 RigidTransform(quat_axis_angle((0, 0, 1), math.pi / 2),
                                Vec3(0.0, 0.0, 0.0))),
        PoseNode("spine", "root", RigidTransform(quat_axis_angle((0, 0, 1), 0.0),
                                                 Vec3(0.0, 2.0, 0.0))),
    ]
    pose = resolve_global_poses(currents)
    print(f"拓扑序: {pose.order}")
    for name in pose.order:
        t = pose.transforms[name].translation
        print(f"  {name:6s} 世界平移 = ({t.x:+.4f}, {t.y:+.4f}, {t.z:+.4f})")

    vertices = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # hand 端顶点
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),   # spine 端顶点，法向 +x
    ]
    weights = [[("hand", 1.0)], [("spine", 1.0)]]
    results = skin_named(vertices, binds, currents, weights)
    print(f"hand  顶点位置（手算期望 (-3,0,0)）: "
          f"({results[0].position.x:+.4f}, {results[0].position.y:+.4f}, "
          f"{results[0].position.z:+.4f})")
    print(f"spine 顶点位置（手算期望 (-2,0,0)）: "
          f"({results[1].position.x:+.4f}, {results[1].position.y:+.4f}, "
          f"{results[1].position.z:+.4f})")
    print(f"spine 法向 +x 经 Rz90（期望 (0,1,0)）: "
          f"({results[1].normal.x:+.4f}, {results[1].normal.y:+.4f}, "
          f"{results[1].normal.z:+.4f})")

    # ---- 场景 2（旧能力）：两段骨骼反向扭转，DQS vs LBS ----
    print()
    print("=== 正常结果：扭转区域（两骨 ±150°，等权）DQS vs LBS ===")
    twist = math.radians(150.0)
    bones = [
        Bone("upper", None, ident,
             RigidTransform(quat_axis_angle((1, 0, 0), twist), Vec3(0, 0, 0))),
        Bone("lower", None, ident,
             RigidTransform(quat_axis_angle((1, 0, 0), -twist), Vec3(0, 0, 0))),
    ]
    twist_verts = [((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))]
    twist_weights = [[(0, 0.5), (1, 0.5)]]

    (res,) = skin(twist_verts, bones, twist_weights)
    dqs_radius = math.hypot(res.position.y, res.position.z)

    lbs = lbs_point(
        [(bones[0].current.rotation, (0, 0, 0)),
         (bones[1].current.rotation, (0, 0, 0))],
        [0.5, 0.5],
        (0.0, 1.0, 0.0),
    )
    lbs_radius = math.hypot(lbs[1], lbs[2])
    print(f"DQS 位置 = ({res.position.x:+.6f}, {res.position.y:+.6f}, {res.position.z:+.6f})")
    print(f"DQS 半径（应≈1，不塌陷）: {dqs_radius:.6f}")
    print(f"LBS 位置 = ({lbs[0]:+.6f}, {lbs[1]:+.6f}, {lbs[2]:+.6f})")
    print(f"LBS 半径（塌陷对照）     : {lbs_radius:.6f}")

    # ---- 边界 1：非单位四元数在两种规范化规则下的不同处理 ----
    print()
    print("=== 边界：四元数规范化规则固定（reject / normalize）===")
    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    doubled = Quat(2 * c, 0.0, 0.0, 2 * s)  # 范数 2，方向是绕 z 60°
    loose = [PoseNode("r", None, RigidTransform(doubled, Vec3(0.0, 0.0, 0.0)))]
    try:
        resolve_global_poses(loose)
        print("（意外）reject 规则接受了范数 2 的四元数")
    except NonRigidTransformError as exc:
        print(f"默认 reject -> NonRigidTransformError: {exc}")
    pose_n = resolve_global_poses(loose, normalize=QUAT_NORMALIZE_NORMALIZE)
    qn = pose_n.transforms["r"].rotation
    nrm = math.sqrt(sum(v * v for v in qn))
    print(f"normalize 后四元数范数 = {nrm:.10f}（方向即绕 z 60°）")
    try:
        resolve_global_poses([PoseNode("r", None, ident)], normalize="quiet")
    except InvalidInputError as exc:
        print(f"未知规则 -> InvalidInputError: {exc}")

    # ---- 边界 2：实际触发层级拒绝 ----
    print()
    print("=== 实际错误：环 / 非法父节点 / 重复绑定 ===")
    for label, bad in (
        ("两节点环", [PoseNode("a", "b", ident), PoseNode("b", "a", ident)]),
        ("非法父节点", [PoseNode("a", "ghost", ident)]),
        ("重复绑定", [PoseNode("a", None, ident), PoseNode("a", None, ident)]),
    ):
        try:
            resolve_global_poses(bad)
            print(f"{label}: （意外）未触发拒绝")
        except HierarchyError as exc:
            print(f"{label} -> HierarchyError: {exc}")

    # ---- 边界 3：实际触发负权重拒绝（旧入口） ----
    print()
    print("=== 实际错误：负权重 ===")
    try:
        skin(twist_verts, bones, [[(0, 1.2), (1, -0.2)]])
        print("（意外）未触发拒绝")
    except WeightError as exc:
        print(f"已拒绝 -> WeightError: {exc}")


if __name__ == "__main__":
    main()
