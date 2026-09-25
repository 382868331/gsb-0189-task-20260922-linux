"""weight_jacobian 权重敏感度测试。

核对方式（相互独立，均不调用被测解析导数本身）：
- 平移型混合：直接手算解析期望 ∂p/∂wₖ = (tₖ − t̄)/W（同向旋转时 DQS
  平移按归一化权重线性），不经过 skin；
- 一般情形（旋转混合、不等权三骨、层级绑定修正）：对公开黑箱 skin
  在安全邻域做中心差分，与解析导数逐项比对（中心差分是题面指定的
  独立核对，不调用 weight_jacobian）；
- 边界：权重平局 / 对齐点积为零 / 实部近抵消，必须返回
  differentiable=False 并给出原因；
- 非法输入沿用 skin 同一套校验（负权、零总权等）。
"""

from __future__ import annotations

import math
import unittest

from dqskin import (
    Bone,
    Quat,
    RigidTransform,
    Vec3,
    WeightError,
    InvalidInputError,
    skin,
    weight_jacobian,
    WEIGHT_MARGIN_TOL,
    SIGN_DOT_TOL,
    REAL_NORM_MIN,
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


def assert_vec_close(tc, actual, expected, tol=1e-7):
    tc.assertAlmostEqual(actual[0], expected[0], delta=tol)
    tc.assertAlmostEqual(actual[1], expected[1], delta=tol)
    tc.assertAlmostEqual(actual[2], expected[2], delta=tol)


def cd_jacobian(bones, vertex, idxs, weights, h=1e-6):
    """对公开 skin 做中心差分：∂position/∂wₖ。安全邻域内符号/归一化
    分支保持不变（调用方保证远离切换边界）。"""
    def pos(wv):
        row = [(idxs[j], wv[j]) for j in range(len(wv))]
        return skin([vertex], bones, [row])[0].position

    out = []
    for k in range(len(weights)):
        wp = list(weights)
        wm = list(weights)
        wp[k] += h
        wm[k] -= h
        pp, pm = pos(wp), pos(wm)
        out.append(
            Vec3(
                (pp.x - pm.x) / (2 * h),
                (pp.y - pm.y) / (2 * h),
                (pp.z - pm.z) / (2 * h),
            )
        )
    return out


def fd_one_sided(bones, vertex, idxs, weights, k, h=1e-6):
    """单侧前向差分（供 wₖ=0 的方向导数核对）。"""
    def pos(wv):
        row = [(idxs[j], wv[j]) for j in range(len(wv))]
        return skin([vertex], bones, [row])[0].position

    wp = list(weights)
    wp[k] += h
    pp, p0 = pos(wp), pos(weights)
    return Vec3((pp.x - p0.x) / h, (pp.y - p0.y) / h, (pp.z - p0.z) / h)


class TestTranslationBlendAnalytic(unittest.TestCase):
    """同向旋转、仅平移不同：平移线性，导数有手算闭式。"""

    def test_derivative_formula(self):
        R = rigid((0, 0, 1), 0.6)
        t1, t2, t3 = (2.0, 0.0, -1.0), (0.0, 3.0, 1.0), (-1.0, 1.0, 2.0)
        bones = [
            Bone("a", None, IDENT, RigidTransform(R.rotation, Vec3(*t1))),
            Bone("b", None, IDENT, RigidTransform(R.rotation, Vec3(*t2))),
            Bone("c", None, IDENT, RigidTransform(R.rotation, Vec3(*t3))),
        ]
        wv = [0.5, 0.3, 0.2]
        vertex = ((0.4, -0.9, 1.1), (0.0, 0.0, 1.0))
        sens = weight_jacobian(vertex, bones, list(enumerate(wv)))
        self.assertTrue(sens.differentiable, sens.reason)
        W = sum(wv)
        tbar = tuple(sum(wv[i] * (t1, t2, t3)[i][j] for i in range(3)) / W
                     for j in range(3))
        for k, d in enumerate(sens.derivatives):
            self.assertEqual(d.bone_index, k)
            tk = (t1, t2, t3)[k]
            expected = tuple((tk[j] - tbar[j]) / W for j in range(3))
            assert_vec_close(self, d.d_position, expected, tol=1e-9)

    def test_single_bone_derivative_zero(self):
        bones = [Bone("b", None, IDENT, rigid((0, 1, 0), 0.9, (1.0, 2.0, 3.0)))]
        for scale in (1.0, 137.0):
            sens = weight_jacobian(((0.3, 0.2, -0.5), (1.0, 0.0, 0.0)),
                                   bones, [(0, scale)])
            self.assertTrue(sens.differentiable)
            (d,) = sens.derivatives
            assert_vec_close(self, d.d_position, (0.0, 0.0, 0.0), tol=1e-12)


class TestCentralDifferences(unittest.TestCase):
    """中心差分独立核对：旋转混合、平移+旋转、不等权三骨。"""

    def test_two_bone_rotation_blend(self):
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.8)),
            Bone("b", None, IDENT, rigid((0, 1, 0), -0.5, (1.0, 0.0, 0.2))),
        ]
        wv = [0.7, 0.3]
        vertex = ((0.6, -0.4, 0.8), (0.0, 1.0, 0.0))
        sens = weight_jacobian(vertex, bones, [(0, wv[0]), (1, wv[1])])
        self.assertTrue(sens.differentiable, sens.reason)
        cd = cd_jacobian(bones, vertex, [0, 1], wv)
        for d, c in zip(sens.derivatives, cd):
            assert_vec_close(self, d.d_position, c, tol=1e-7)

    def test_three_bone_unequal_with_hierarchy(self):
        # 含父子层级与非平凡绑定姿态：三骨不等权，旋转与平移都参与。
        binds = [
            rigid((0, 0, 1), 0.2, (0.0, 1.0, 0.0)),
            rigid((1, 0, 0), -0.3, (1.0, 0.0, 0.0)),
            rigid((0, 1, 0), 0.4, (0.0, 0.0, 1.0)),
        ]
        curs = [
            rigid((0, 0, 1), 1.1, (0.5, 0.0, 0.0)),
            rigid((1, 0, 1), 0.7, (0.0, 0.5, 0.0)),
            rigid((0, 1, 0), -0.9, (-0.2, 0.1, 0.5)),
        ]
        bones = [
            Bone("b0", None, binds[0], curs[0]),
            Bone("b1", 0, binds[1], curs[1]),
            Bone("b2", 1, binds[2], curs[2]),
        ]
        wv = [0.55, 0.3, 0.15]
        vertex = ((0.4, -0.8, 1.3), (0.0, 0.0, 1.0))
        row = [(2, wv[2]), (0, wv[0]), (1, wv[1])]  # 故意打乱顺序
        sens = weight_jacobian(vertex, bones, row)
        self.assertTrue(sens.differentiable, sens.reason)
        # 导数顺序与 bone_index 必须跟随输入行
        self.assertEqual([d.bone_index for d in sens.derivatives], [2, 0, 1])
        cd = cd_jacobian(bones, vertex, [2, 0, 1], [wv[2], wv[0], wv[1]])
        for d, c in zip(sens.derivatives, cd):
            assert_vec_close(self, d.d_position, c, tol=1e-7)

    def test_unscaled_weights_and_scale_null(self):
        # 未归一化权重（和不为 1）：导数数值随总权变化，但缩放零方向恒为零。
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.9, (1.0, 0.0, 0.0))),
            Bone("b", None, IDENT, rigid((0, 1, 0), -0.6, (0.0, 2.0, 0.5))),
        ]
        vertex = ((0.3, 0.7, -0.2), (1.0, 0.0, 0.0))
        for scale in (0.013, 1.0, 42.0):
            wv = [0.3 * scale, 0.7 * scale]
            sens = weight_jacobian(vertex, bones, [(0, wv[0]), (1, wv[1])])
            self.assertTrue(sens.differentiable, sens.reason)
            cd = cd_jacobian(bones, vertex, [0, 1], wv, h=max(1e-8, 1e-6 * scale))
            for d, c in zip(sens.derivatives, cd):
                assert_vec_close(self, d.d_position, c, tol=1e-6)
            # Σₖ wₖ ∂p/∂wₖ = 0（全体权重同倍缩放的方向导数）
            for axis in range(3):
                null = sum(wv[k] * sens.derivatives[k].d_position[axis]
                           for k in range(2))
                self.assertAlmostEqual(null, 0.0, delta=1e-9)

    def test_derivative_is_dqs_not_lbs(self):
        # 防冒充：存在旋转混合时，解析 DQS 导数必须匹配 skin 的中心差分，
        # 且与朴素 LBS 权重导数（(Rₖp+tₖ−p')/W）明显不同。
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 1.1, (0.5, 0.0, 0.0))),
            Bone("b", None, IDENT, rigid((0, 1, 0), -0.8, (0.0, 0.5, 0.3))),
        ]
        wv = [0.62, 0.38]
        vertex = ((0.7, -0.5, 0.4), (0.0, 1.0, 0.0))
        sens = weight_jacobian(vertex, bones, [(0, wv[0]), (1, wv[1])])
        self.assertTrue(sens.differentiable, sens.reason)
        cd = cd_jacobian(bones, vertex, [0, 1], wv)
        for d, c in zip(sens.derivatives, cd):
            assert_vec_close(self, d.d_position, c, tol=1e-7)

        # LBS 闭式导数（独立手算：四元数旋转矩阵展开）
        p = vertex[0]
        p_out = skin([vertex], bones, [[(0, wv[0]), (1, wv[1])]])[0].position

        def rot(q, v):
            w, x, y, z = q
            R = [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
            return tuple(sum(R[i][j] * v[j] for j in range(3)) for i in range(3))

        W = sum(wv)
        lbs_jac = []
        for k in range(2):
            rt = bones[k].current
            qp = tuple(rot(rt.rotation, p)[i] + rt.translation[i] for i in range(3))
            lbs_jac.append(tuple((qp[i] - p_out[i]) / W for i in range(3)))
        for d, lj in zip(sens.derivatives, lbs_jac):
            gap = max(abs(d.d_position[i] - lj[i]) for i in range(3))
            self.assertGreater(gap, 1e-3, "DQS 导数与 LBS 导数异常相同，疑似冒充")

    def test_zero_weight_entry_one_sided(self):
        # 零权重骨骼：解析导数应等于 w 离开 0 的单侧方向导数。
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.5, (1.0, 0.0, 0.0))),
            Bone("b", None, IDENT, rigid((1, 0, 0), 0.9, (0.0, 2.0, 0.0))),
        ]
        wv = [0.8, 0.0]
        vertex = ((0.5, -0.3, 0.7), (0.0, 0.0, 1.0))
        sens = weight_jacobian(vertex, bones, [(0, wv[0]), (1, wv[1])])
        self.assertTrue(sens.differentiable, sens.reason)
        d_zero = sens.derivatives[1].d_position
        fd = fd_one_sided(bones, vertex, [0, 1], wv, 1, h=1e-7)
        assert_vec_close(self, d_zero, fd, tol=1e-5)
        # 非零项中心差分（只扰动 k=0；k=1 保持为 0，不触碰负权边界）
        h = 1e-6
        def pos0(w0):
            return skin([vertex], bones, [[(0, w0), (1, 0.0)]])[0].position
        pp, pm = pos0(0.8 + h), pos0(0.8 - h)
        cd0 = Vec3((pp.x - pm.x) / (2 * h),
                   (pp.y - pm.y) / (2 * h),
                   (pp.z - pm.z) / (2 * h))
        assert_vec_close(self, sens.derivatives[0].d_position, cd0, tol=1e-6)


class TestNondifferentiableBoundaries(unittest.TestCase):
    """切换/退化边界：必须标 nondifferentiable 并给原因。"""

    def test_weight_tie(self):
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.4)),
            Bone("b", None, IDENT, rigid((0, 1, 0), 0.4)),
        ]
        sens = weight_jacobian(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                               bones, [(0, 0.5), (1, 0.5)])
        self.assertFalse(sens.differentiable)
        self.assertEqual(sens.derivatives, ())
        self.assertIn("最大权重并列", sens.reason)
        self.assertIn("不可微", sens.reason)

    def test_weight_tie_unscaled(self):
        # 平局判定基于归一化权重：0.3/0.3/0.3 是平局，0.3/0.3/0.4 不是
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.3)),
            Bone("b", None, IDENT, rigid((0, 1, 0), 0.3)),
            Bone("c", None, IDENT, rigid((1, 0, 0), 0.3)),
        ]
        sens = weight_jacobian(((1.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                               bones, [(0, 0.3), (1, 0.3), (2, 0.4)])
        self.assertTrue(sens.differentiable, sens.reason)
        sens_tie = weight_jacobian(
            ((1.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            bones, [(0, 0.3), (1, 0.3), (2, 0.3)],
        )
        self.assertFalse(sens_tie.differentiable)
        self.assertIn("最大权重并列", sens_tie.reason)

    def test_sign_dot_zero(self):
        # 恒等旋转与绕 y 180°：实部 (1,0,0,0) 与 (0,0,1,0) 点积恰为零。
        bones = [
            Bone("a", None, IDENT, IDENT),
            Bone("b", None, IDENT, rigid((0, 1, 0), math.pi)),
        ]
        sens = weight_jacobian(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                               bones, [(0, 0.7), (1, 0.3)])
        self.assertFalse(sens.differentiable)
        self.assertIn("对齐点积", sens.reason)

    def test_near_cancellation_real_norm(self):
        # 32 骨近抵消构造（用 MAX_BONES 上限达到该分支可达域）：
        # 基准骨与其余骨的旋转角均为 θ=π-0.01（半角 π/2-ε），故四元数
        # 标量 c=sin(ε)≈0.005、向量部模 s=cos(ε)≈1。基准骨轴沿 x，
        # 其余 31 骨的轴均匀铺在 yz 平面（与 x 垂直）。此时：
        #   - 基准唯一：margin ≈ 7.7e-4 > WEIGHT_MARGIN_TOL；
        #   - 对齐点积 dot = c² ≈ 2.5e-5 > SIGN_DOT_TOL，无一翻转；
        #   - 31 个横向单位向量之和恰为零（31 次单位根），
        #     混合实部范数 ≈ α0 ≈ 0.0324 < REAL_NORM_MIN。
        nb = 32
        eps = 0.005
        c, s = math.sin(eps), math.cos(eps)
        bones = [Bone("ref", None, IDENT,
                      RigidTransform(Quat(c, s, 0.0, 0.0), Vec3(0, 0, 0)))]
        for j in range(nb - 1):
            phi = 2.0 * math.pi * j / (nb - 1)
            bones.append(Bone(f"t{j}", None, IDENT, RigidTransform(
                Quat(c, 0.0, s * math.cos(phi), s * math.sin(phi)),
                Vec3(0, 0, 0))))
        a0 = 0.032
        rest = (1.0 - a0) / (nb - 1)
        row = [(0, a0)] + [(i, rest) for i in range(1, nb)]
        sens = weight_jacobian(((0.2, 0.3, -0.4), (1.0, 0.0, 0.0)),
                               bones, row)
        self.assertFalse(sens.differentiable)
        self.assertIn("混合实部范数", sens.reason)
        # 同构配置但远离抵消（基准骨主导）时应可微，确认不是误杀
        row_safe = [(0, 0.8)] + [(i, 0.2 / (nb - 1)) for i in range(1, nb)]
        sens_safe = weight_jacobian(((0.2, 0.3, -0.4), (1.0, 0, 0)),
                                    bones, row_safe)
        self.assertTrue(sens_safe.differentiable, sens_safe.reason)
        cd = cd_jacobian(bones, ((0.2, 0.3, -0.4), (1, 0, 0)),
                         [r[0] for r in row_safe], [r[1] for r in row_safe])
        assert_vec_close(self, sens_safe.derivatives[0].d_position,
                         cd[0], tol=1e-7)

    def test_thresholds_are_explicit_constants(self):
        self.assertGreater(WEIGHT_MARGIN_TOL, 0.0)
        self.assertGreater(SIGN_DOT_TOL, 0.0)
        self.assertGreater(REAL_NORM_MIN, SIGN_DOT_TOL)


class TestValidation(unittest.TestCase):
    """weight_jacobian 复用 skin 校验，非法输入不返回敏感度结果。"""

    def test_negative_weight_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            weight_jacobian(((0, 0, 0), (0, 0, 1)), bones, [(0, -0.1)])

    def test_zero_total_weight_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            weight_jacobian(((0, 0, 0), (0, 0, 1)), bones, [(0, 0.0)])

    def test_empty_row_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            weight_jacobian(((0, 0, 0), (0, 0, 1)), bones, [])

    def test_bad_vertex_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(InvalidInputError):
            weight_jacobian(((0, 0, 0), (0, 0, 0)), bones, [(0, 1.0)])

    def test_duplicate_bone_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT), Bone("c", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            weight_jacobian(((0, 0, 0), (0, 0, 1)),
                            bones, [(0, 0.5), (0, 0.5)])

    def test_safe_case_does_not_mutate_inputs(self):
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), 0.4)),
            Bone("b", None, IDENT, rigid((0, 1, 0), 0.4)),
        ]
        vertex = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        row = [(0, 0.7), (1, 0.3)]
        snapshot = repr(row)
        sens = weight_jacobian(vertex, bones, row)
        self.assertTrue(sens.differentiable)
        self.assertEqual(repr(row), snapshot)


if __name__ == "__main__":
    unittest.main()
