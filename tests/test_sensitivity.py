"""dqskin 权重敏感度查询（skin_weight_sensitivity）单元测试。

解析导数的核对方式：在安全邻域内对 skin 输出做中心差分（独立于一阶导
代码路径），另验证缩放方向导数为零（Euler 齐次性）、单骨导数为零等解析性质。
边界用例：权重平局、对齐点积近零（近对消）、零总权重。
"""

from __future__ import annotations

import math
import unittest

from dqskin import (
    ALIGN_DOT_TOL,
    Bone,
    InvalidInputError,
    Quat,
    RigidTransform,
    Vec3,
    WeightError,
    WeightSensitivity,
    skin,
    skin_weight_sensitivity,
)


def quat_axis_angle(axis, angle):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    h = angle / 2.0
    s = math.sin(h) / n
    return Quat(math.cos(h), ax * s, ay * s, az * s)


IDENT = RigidTransform(Quat(1.0, 0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))


def rigid(axis, angle, t=(0.0, 0.0, 0.0)):
    return RigidTransform(quat_axis_angle(axis, angle), Vec3(*t))


def three_bone_case():
    """平移 + 旋转混合 + 不等权的三骨骼案例（含非平凡绑定姿态）。"""
    bones = [
        Bone("b0", None, IDENT, rigid((0, 0, 1), 0.9, (1.0, 0.0, 0.0))),
        Bone("b1", None, rigid((1, 0, 0), 0.2, (0.1, 0.0, 0.0)),
             rigid((0, 1, 0), -0.7, (0.0, 2.0, 1.0))),
        Bone("b2", None, IDENT, rigid((1, 0, 0), 0.4, (-1.0, 0.5, 2.0))),
    ]
    position = (0.3, -0.7, 1.1)
    weights = [(0, 0.5), (1, 0.3), (2, 0.2)]
    return position, bones, weights


def skinned_position(position, bones, weights):
    """用 skin 求同一顶点位置（差分用；法向固定不影响位置）。"""
    return skin([(position, (0.0, 0.0, 1.0))], bones, [weights])[0].position


def central_difference(position, bones, weights, h=1e-6):
    """对每个权重条目做中心差分，返回与 weights 等长的 Vec3 列表。"""
    cols = []
    for j in range(len(weights)):
        wp = [list(e) for e in weights]
        wm = [list(e) for e in weights]
        wp[j][1] += h
        wm[j][1] -= h
        pp = skinned_position(position, bones, wp)
        pm = skinned_position(position, bones, wm)
        cols.append(tuple((a - b) / (2.0 * h) for a, b in zip(pp, pm)))
    return cols


class TestSensitivityAnalytic(unittest.TestCase):
    """可微区域：解析导数 vs 中心差分 / 解析性质。"""

    def test_position_matches_skin_exactly(self):
        position, bones, weights = three_bone_case()
        sens = skin_weight_sensitivity(position, bones, weights)
        expected = skinned_position(position, bones, weights)
        self.assertEqual(tuple(sens.position), tuple(expected))

    def test_result_contract(self):
        position, bones, weights = three_bone_case()
        sens = skin_weight_sensitivity(position, bones, weights)
        self.assertIsInstance(sens, WeightSensitivity)
        self.assertTrue(sens.differentiable)
        self.assertIsNone(sens.nondifferentiable_reason)
        self.assertEqual(len(sens.jacobian), len(weights))
        for col in sens.jacobian:
            self.assertIsInstance(col, Vec3)

    def test_jacobian_matches_central_difference(self):
        position, bones, weights = three_bone_case()
        sens = skin_weight_sensitivity(position, bones, weights)
        cd = central_difference(position, bones, weights)
        for j, (analytic, numeric) in enumerate(zip(sens.jacobian, cd)):
            for a, b in zip(analytic, numeric):
                self.assertAlmostEqual(a, b, delta=1e-6,
                                       msg=f"jacobian[{j}] 与中心差分不符")

    def test_uniform_scaling_directional_derivative_is_zero(self):
        # 导数针对未归一化输入：沿全体权重同倍缩放方向 w 的方向导数
        # Σ_j w_j·J[j] 必须为零（位置是权重的零次齐次函数）。
        position, bones, weights = three_bone_case()
        sens = skin_weight_sensitivity(position, bones, weights)
        for k in range(3):
            dir_der = sum(w * sens.jacobian[j][k] for j, (_, w) in enumerate(weights))
            self.assertAlmostEqual(dir_der, 0.0, delta=1e-9)

    def test_single_bone_jacobian_is_zero(self):
        # 单骨：位置不依赖权重数值，导数恒为零
        bones = [Bone("b", None, IDENT, rigid((0, 0, 1), 1.1, (1.0, 2.0, 3.0)))]
        sens = skin_weight_sensitivity((0.5, 0.5, 0.5), bones, [(0, 2.5)])
        self.assertTrue(sens.differentiable)
        for comp in sens.jacobian[0]:
            self.assertAlmostEqual(comp, 0.0, delta=1e-9)

    def test_zero_weight_entry_right_derivative(self):
        # 零权重项给出右导数：用前向差分（负侧会违反非负约束）松散核对
        position, bones, _ = three_bone_case()
        weights = [(0, 0.5), (1, 0.3), (2, 0.0)]
        sens = skin_weight_sensitivity(position, bones, weights)
        self.assertTrue(sens.differentiable)
        h = 1e-7
        wp = [list(e) for e in weights]
        wp[2][1] += h
        p0 = skinned_position(position, bones, weights)
        p1 = skinned_position(position, bones, wp)
        for k in range(3):
            forward = (p1[k] - p0[k]) / h
            self.assertAlmostEqual(forward, sens.jacobian[2][k], delta=1e-4)

    def test_weight_scaling_leaves_position_and_direction_zero(self):
        # 权重整体缩放：position 不变，且缩放方向导数仍为零
        position, bones, weights = three_bone_case()
        base = skin_weight_sensitivity(position, bones, weights)
        scaled = skin_weight_sensitivity(
            position, bones, [(b, 100.0 * w) for b, w in weights]
        )
        for a, b in zip(base.position, scaled.position):
            self.assertAlmostEqual(a, b, delta=1e-12)
        for k in range(3):
            dir_der = sum(
                100.0 * w * scaled.jacobian[j][k]
                for j, (_, w) in enumerate(weights)
            )
            self.assertAlmostEqual(dir_der, 0.0, delta=1e-9)


class TestSensitivityBoundaries(unittest.TestCase):
    """切换边界：标记不可微并给出原因；position 仍与 skin 一致。"""

    def assert_nondiff(self, sens, keyword, position, bones, weights):
        self.assertIsInstance(sens, WeightSensitivity)
        self.assertFalse(sens.differentiable)
        self.assertIsNone(sens.jacobian)
        self.assertIn(keyword, sens.nondifferentiable_reason)
        # position 仍然返回，且与 skin 输出一致
        self.assertEqual(
            tuple(sens.position), tuple(skinned_position(position, bones, weights))
        )

    def test_weight_tie_at_max(self):
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.8)),
            Bone("b", None, IDENT, rigid((0, 1, 0), -0.5, (1.0, 0.0, 0.0))),
        ]
        weights = [(0, 0.5), (1, 0.5)]
        sens = skin_weight_sensitivity((1.0, 0.0, 0.0), bones, weights)
        self.assert_nondiff(sens, "并列", (1.0, 0.0, 0.0), bones, weights)

    def test_near_tie_is_still_differentiable(self):
        # 切换边界是严格并列；近并列（基准局部不变）仍可微
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.8)),
            Bone("b", None, IDENT, rigid((0, 1, 0), -0.5, (1.0, 0.0, 0.0))),
        ]
        weights = [(0, 0.5), (1, 0.5 - 1e-9)]
        sens = skin_weight_sensitivity((1.0, 0.0, 0.0), bones, weights)
        self.assertTrue(sens.differentiable)

    def test_alignment_dot_near_zero(self):
        # 骨骼 1 相对基准（恒等）旋转恰好 180°：对齐点积 ≈ 0（近对消）
        bones = [
            Bone("ref", None, IDENT, IDENT),
            Bone("flip", None, IDENT, rigid((0, 0, 1), math.pi)),
        ]
        weights = [(0, 0.6), (1, 0.4)]
        sens = skin_weight_sensitivity((1.0, 2.0, 3.0), bones, weights)
        self.assert_nondiff(sens, "点积", (1.0, 2.0, 3.0), bones, weights)

    def test_alignment_dot_above_threshold_ok(self):
        # 179°：点积 ≈ cos(89.5°) ≈ 0.0087 > ALIGN_DOT_TOL，仍可微
        bones = [
            Bone("ref", None, IDENT, IDENT),
            Bone("far", None, IDENT, rigid((0, 0, 1), math.radians(179.0))),
        ]
        weights = [(0, 0.6), (1, 0.4)]
        sens = skin_weight_sensitivity((1.0, 2.0, 3.0), bones, weights)
        self.assertTrue(sens.differentiable)
        self.assertLess(abs(math.cos(math.radians(89.5))), 0.01)
        self.assertGreater(abs(math.cos(math.radians(89.5))), ALIGN_DOT_TOL)

    def test_near_cancellation_flagged_by_norm_gate(self):
        # 近对消：两骨骼在基准（恒等）附近 180° 处、向量部相互抵消，
        # 混合实部相对范数 ≈ w_max/Σw。默认阈值下可微（该闸门是防御性的，
        # 相对范数恒 >= 1/k），用显式阈值验证闸门本身会触发。
        angle = 2.0 * math.acos(0.01)  # 实部 = 0.01，远离 ALIGN_DOT_TOL
        bones = [
            Bone("ref", None, IDENT, IDENT),
            Bone("px", None, IDENT, rigid((1, 0, 0), angle)),
            Bone("nx", None, IDENT, rigid((-1, 0, 0), angle)),
        ]
        weights = [(0, 0.34), (1, 0.33), (2, 0.33)]
        ok = skin_weight_sensitivity((0.1, 0.2, 0.3), bones, weights)
        self.assertTrue(ok.differentiable)
        sens = skin_weight_sensitivity(
            (0.1, 0.2, 0.3), bones, weights, min_blend_norm=0.6
        )
        self.assert_nondiff(sens, "范数", (0.1, 0.2, 0.3), bones, weights)

    def test_zero_total_weight_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            skin_weight_sensitivity((0.0, 0.0, 0.0), bones, [(0, 0.0)])
        with self.assertRaises(WeightError):
            skin_weight_sensitivity((0.0, 0.0, 0.0), bones, [])

    def test_invalid_inputs_match_skin_errors(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):  # 负权
            skin_weight_sensitivity((0.0, 0.0, 0.0), bones, [(0, -0.5)])
        with self.assertRaises(WeightError):  # 下标越界
            skin_weight_sensitivity((0.0, 0.0, 0.0), bones, [(1, 1.0)])
        with self.assertRaises(InvalidInputError):  # NaN 位置
            skin_weight_sensitivity((float("nan"), 0.0, 0.0), bones, [(0, 1.0)])
        with self.assertRaises(InvalidInputError):  # bool 权重
            skin_weight_sensitivity((0.0, 0.0, 0.0), bones, [(0, True)])
        with self.assertRaises(InvalidInputError):  # 非法阈值
            skin_weight_sensitivity(
                (0.0, 0.0, 0.0), bones, [(0, 1.0)], align_dot_tol=-1.0
            )


if __name__ == "__main__":
    unittest.main()
