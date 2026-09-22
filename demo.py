"""固定输入演示：双四元数蒙皮（dqskin）。

运行：python demo.py（约 0.1 秒，纯标准库，无需网络）。

演示两部分，全部真实计算、无预录结果：

1. 正常结果 —— 两骨“扭转区域”（candy-wrapper 区）：两根骨骼绕同一轴
   反向各转 85°（姿态相差 170°），关节处顶点等权受二者影响。对比
   dqskin 的双四元数结果与 demo 内联的独立矩阵线性混合（LBS）参考：
   DLB 保持刚体性质（径向长度/法向长度不缩），LBS 在大扭转角下塌缩。
2. 实际触发的错误边界 —— 负权输入被 WeightError 拒绝（含可定位前缀），
   零总权同样被拒绝。
"""

from __future__ import annotations

import math

from dqskin import (
    Bone,
    RigidTransform,
    SkinningError,
    skin,
)


# --------------------------------------------------------------------------- #
# demo 内联的独立 LBS（矩阵线性混合）参考，仅用于和 dqskin 对比。
# 刻意用最朴素的 4x4 矩阵代码现场计算，不调用 dqskin。
# --------------------------------------------------------------------------- #


def axis_angle_matrix(axis, angle, translation=(0.0, 0.0, 0.0)):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / n, ay / n, az / n
    c, s, C = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
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


def inverse_rigid(m):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            out[i][j] = m[j][i]
        out[i][3] = -sum(m[j][i] * m[j][3] for j in range(3))
    out[3][3] = 1.0
    return tuple(tuple(r) for r in out)


def lbs_blend_one(vertices, rel_matrices, weight_lists):
    """独立 LBS：M* = (sum w_i T_i) / sum w_i，再作用到位置/法向。"""
    def pairs(w):
        return w.items() if isinstance(w, dict) else w

    out = []
    for v, wl in zip(vertices, weight_lists):
        total = sum(w for _, w in pairs(wl))
        m = [[0.0] * 4 for _ in range(4)]
        for bi, w in pairs(wl):
            t = rel_matrices[bi]
            for r in range(4):
                for c in range(4):
                    m[r][c] += w * t[r][c] / total
        mt = tuple(tuple(r) for r in m)
        out.append((
            mt[0][0] * v[0] + mt[0][1] * v[1] + mt[0][2] * v[2] + mt[0][3],
            mt[1][0] * v[0] + mt[1][1] * v[1] + mt[1][2] * v[2] + mt[1][3],
            mt[2][0] * v[0] + mt[2][1] * v[1] + mt[2][2] * v[2] + mt[2][3],
        ))
    return out


def _fmt(v):
    return f"({v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f})"


def section_twist_region():
    print("=" * 72)
    print("正常结果：两骨扭转区域 candy-wrapper（绕共同关节轴反向扭转，50/50）")
    print("=" * 72)

    bind_i = axis_angle_matrix((1, 0, 0), 0.0)

    # 关节平面 x=0 上的固定采样：中心点 + 三个径向点（构成“截面环”）。
    vertices = [(0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, -0.6, 0.8)]
    # 法向取各点的横向方向（轴向法向不受扭转尺度影响，无法暴露差异）。
    normals = [(1.0, 0.0, 0.0),
               (0.0, 1.0, 0.0),
               (0.0, 0.0, 1.0),
               (0.0, -0.6, 0.8)]
    weights = [{0: 0.5, 1: 0.5} for _ in vertices]

    print(f"{'扭转角(每骨)':>12} | {'DLB 环半径(dqskin)':>18} | "
          f"{'LBS 环半径(独立参考)':>20} | 说明")
    for deg in (15.0, 45.0, 85.0):
        rot_plus = axis_angle_matrix((1, 0, 0), math.radians(deg))
        rot_minus = axis_angle_matrix((1, 0, 0), math.radians(-deg))
        bones = [
            Bone(RigidTransform(bind_i), RigidTransform(rot_plus)),
            Bone(RigidTransform(bind_i), RigidTransform(rot_minus)),
        ]
        result = skin(vertices, bones, weights, normals=normals)
        lbs_vertices = lbs_blend_one(vertices, [rot_plus, rot_minus], weights)

        # 截面环半径：三个径向点到关节中心的距离。
        d_center = result.vertices[0]
        l_center = lbs_vertices[0]
        d_radii = [
            math.sqrt(sum((result.vertices[k][j] - d_center[j]) ** 2
                          for j in range(3)))
            for k in range(1, 4)
        ]
        l_radii = [
            math.sqrt(sum((lbs_vertices[k][j] - l_center[j]) ** 2
                          for j in range(3)))
            for k in range(1, 4)
        ]
        note = "小扭转，差异轻微" if deg <= 15.0 else (
            "LBS 开始明显塌缩" if deg <= 45.0 else "candy-wrapper 塌缩"
        )
        print(f"{deg:>10.0f}°  | {d_radii[0]:>18.6f} | {l_radii[0]:>20.6f} "
              f"| {note}（环半径离散: DLB {max(d_radii)-min(d_radii):.2e}, "
              f"LBS {max(l_radii)-min(l_radii):.2e}）")

    # 85° 情形的逐点明细，便于核对真实输出。
    rot_plus = axis_angle_matrix((1, 0, 0), math.radians(85.0))
    rot_minus = axis_angle_matrix((1, 0, 0), math.radians(-85.0))
    bones = [
        Bone(RigidTransform(bind_i), RigidTransform(rot_plus)),
        Bone(RigidTransform(bind_i), RigidTransform(rot_minus)),
    ]
    result = skin(vertices, bones, weights, normals=normals)
    lbs_vertices = lbs_blend_one(vertices, [rot_plus, rot_minus], weights)
    lbs_normals = lbs_blend_one(normals, [rot_plus, rot_minus], weights)
    d_nlen = math.sqrt(sum(c * c for c in result.normals[1]))
    l_nlen = math.sqrt(sum(c * c for c in lbs_normals[1]))
    print()
    print(f"85° 逐点明细（顶点 1 法向长度：输入 1.0 -> "
          f"DLB {d_nlen:.6f}，LBS {l_nlen:.6f}）：")
    print(f"{'顶点(输入)':>20} | {'DLB 位置(dqskin)':>22} | "
          f"{'LBS 位置(独立参考)':>22}")
    for v, dv, lv in zip(vertices, result.vertices, lbs_vertices):
        print(f"{_fmt(v):>20} | {_fmt(dv):>22} | {_fmt(lv):>22}")
    print()
    print("解释：两骨姿态相差 170°，符号修正后 DLB 沿短弧混合得单位旋转，")
    print("截面环刚体保持；LBS 平均矩阵给截面带来 cos85° ≈ 0.0872 的尺度，")
    print("环被压塌（法向长度同步收缩，而 DLB 法向长度保持 1.0）。")


def section_rejections():
    print()
    print("=" * 72)
    print("错误输入边界（真实触发，非预录）")
    print("=" * 72)

    bind_i = axis_angle_matrix((1, 0, 0), 0.0)
    bones = [
        Bone(RigidTransform(bind_i), RigidTransform(bind_i)),
        Bone(RigidTransform(bind_i), RigidTransform(bind_i)),
    ]
    vertices = [(1.0, 0.0, 0.0)]

    # 1) 负权：必须被拒绝并定位到 weights[0][?].weight。
    bad_weights = [[(0, 0.6), (1, -0.6)]]
    try:
        skin(vertices, bones, bad_weights)
    except SkinningError as exc:
        print(f"[负权] 捕获 {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - 演示护栏
        raise AssertionError("negative weight was not rejected")

    # 2) 零总权：必须被拒绝。
    try:
        skin(vertices, bones, [{0: 0.0, 1: 0.0}])
    except SkinningError as exc:
        print(f"[零总权] 捕获 {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - 演示护栏
        raise AssertionError("zero total weight was not rejected")

    # 3) 非刚体（剪切矩阵）：在构造 RigidTransform 时即被拒绝。
    sheared = [list(r) for r in bind_i]
    sheared[0][1] = 0.5
    try:
        RigidTransform(tuple(tuple(r) for r in sheared))
    except SkinningError as exc:
        print(f"[非刚体] 捕获 {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - 演示护栏
        raise AssertionError("non-rigid matrix was not rejected")


if __name__ == "__main__":
    section_twist_region()
    section_rejections()
