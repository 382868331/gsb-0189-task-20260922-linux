"""dqskin 单元测试。

期望值来源：
- 解析手算（单骨刚体、同向骨骼线性混合、绑定姿态恒等、180° 附近混合）；
- tests/refimpl.py 中与 dqskin 完全独立的矩阵实现（交叉验证）。
任何期望值都不通过调用 dqskin 自身得到。
"""

from __future__ import annotations

import math
import unittest

from dqskin import (
    Bone,
    HierarchyError,
    InvalidInputError,
    NonRigidTransformError,
    Quat,
    RigidTransform,
    SkinResult,
    Vec3,
    WeightError,
    skin,
)

import refimpl


def quat_axis_angle(axis, angle):
    """轴角 -> 单位四元数 (w, x, y, z)。标准定义，测试侧独立书写。"""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    h = angle / 2.0
    s = math.sin(h) / n
    return Quat(math.cos(h), ax * s, ay * s, az * s)


IDENT = RigidTransform(Quat(1.0, 0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))


def rigid(axis, angle, t=(0.0, 0.0, 0.0)):
    return RigidTransform(quat_axis_angle(axis, angle), Vec3(*t))


def assert_vec_close(tc, actual, expected, tol=1e-9):
    tc.assertAlmostEqual(actual[0], expected[0], delta=tol)
    tc.assertAlmostEqual(actual[1], expected[1], delta=tol)
    tc.assertAlmostEqual(actual[2], expected[2], delta=tol)


class TestSingleBoneAnalytic(unittest.TestCase):
    """单骨：结果必须是解析刚体变换。"""

    def test_rotation_90_about_z(self):
        bones = [Bone("b", None, IDENT, rigid((0, 0, 1), math.pi / 2))]
        verts = [((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
        weights = [[(0, 1.0)]]
        (res,) = skin(verts, bones, weights)
        assert_vec_close(self, res.position, (0.0, 1.0, 0.0))
        assert_vec_close(self, res.normal, (0.0, 1.0, 0.0))

    def test_rotation_plus_translation(self):
        # 绕 y 轴 180°，再平移 (1, 2, 3)
        bones = [Bone("b", None, IDENT, rigid((0, 1, 0), math.pi, (1, 2, 3)))]
        verts = [((1.0, 0.0, 1.0), (0.0, 0.0, 1.0))]
        (res,) = skin(verts, bones, [[(0, 1.0)]])
        assert_vec_close(self, res.position, (0.0, 2.0, 2.0))
        assert_vec_close(self, res.normal, (0.0, 0.0, -1.0))

    def test_translation_does_not_affect_normal(self):
        bones = [Bone("b", None, IDENT, rigid((0, 0, 1), 0.0, (5.0, -3.0, 2.0)))]
        (res,) = skin([((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))], bones, [[(0, 1.0)]])
        assert_vec_close(self, res.position, (5.0, -3.0, 2.0))
        assert_vec_close(self, res.normal, (0.0, 1.0, 0.0))

    def test_bind_correction(self):
        # 绑定姿态含旋转，当前姿态相同 -> 恒等；当前多转 90° -> 净 90°
        bind = rigid((0, 0, 1), math.pi / 2)
        cur = rigid((0, 0, 1), math.pi)
        bones = [Bone("b", None, bind, cur)]
        (res,) = skin([((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))], bones, [[(0, 1.0)]])
        # M = cur * bind^{-1} = 绕 z 净转 90°
        assert_vec_close(self, res.position, (0.0, 1.0, 0.0))
        assert_vec_close(self, res.normal, (-1.0, 0.0, 0.0))

    def test_bind_pose_unchanged(self):
        # current == bind -> 顶点不动（绑定姿态不动性）
        b0 = rigid((1, 2, 3), 0.7, (1.0, -2.0, 0.5))
        b1 = rigid((0, 1, 0), -1.1, (0.0, 3.0, 1.0))
        bones = [
            Bone("root", None, b0, b0),
            Bone("child", 0, b1, b1),
        ]
        verts = [
            ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0)),
            ((-4.0, 0.5, 2.0), (1.0, 0.0, 0.0)),
        ]
        weights = [[(0, 0.3), (1, 0.7)], [(1, 1.0)]]
        for res, (p, n) in zip(skin(verts, bones, weights), verts):
            assert_vec_close(self, res.position, p)
            assert_vec_close(self, res.normal, n)


class TestCoDirectionalBones(unittest.TestCase):
    """同向骨骼：旋转相同的骨骼混合，旋转不变、平移按权重线性。"""

    def test_same_rotation_linear_translation(self):
        R = rigid((0, 0, 1), math.pi / 3)
        t1 = Vec3(2.0, 0.0, 0.0)
        t2 = Vec3(0.0, 4.0, 0.0)
        bones = [
            Bone("a", None, IDENT, RigidTransform(R.rotation, t1)),
            Bone("b", None, IDENT, RigidTransform(R.rotation, t2)),
        ]
        w1, w2 = 0.25, 0.75
        (res,) = skin(
            [((1.0, 1.0, 1.0), (0.0, 0.0, 1.0))],
            bones,
            [[(0, w1), (1, w2)]],
        )
        # 期望：旋转 R，平移 w1*t1 + w2*t2（解析线性参考）
        expected_R = refimpl.axis_angle_matrix((0, 0, 1), math.pi / 3)
        expected_t = (w1 * 2.0 + w2 * 0.0, w1 * 0.0 + w2 * 4.0, 0.0)
        expected_p = tuple(
            refimpl.mat_vec(expected_R, (1.0, 1.0, 1.0))[i] + expected_t[i]
            for i in range(3)
        )
        assert_vec_close(self, res.position, expected_p)
        assert_vec_close(
            self, res.normal, refimpl.mat_vec(expected_R, (0.0, 0.0, 1.0))
        )

    def test_identical_transforms_exact(self):
        # 两骨蒙皮变换完全相同 -> 混合结果等于该变换本身
        m = rigid((1, 1, 0), 0.9, (1.0, 2.0, 3.0))
        bones = [Bone("a", None, IDENT, m), Bone("b", None, IDENT, m)]
        p, n = (0.5, -1.0, 2.0), (0.0, 1.0, 0.0)
        (res,) = skin([(p, n)], bones, [[(0, 0.4), (1, 0.6)]])
        RT = refimpl.rigid_from_axis_angle((1, 1, 0), 0.9, (1.0, 2.0, 3.0))
        assert_vec_close(self, res.position, refimpl.rigid_apply(RT, p))
        assert_vec_close(self, res.normal, refimpl.rigid_rotate(RT, n))


class TestNear180(unittest.TestCase):
    """180° 附近：符号翻转与最短路径混合。"""

    def test_antipodal_same_rotation(self):
        # q 与 -q 表示同一旋转；混合必须仍等于该旋转（符号基准翻转生效），
        # 否则实部相互抵消、结果退化。
        q = quat_axis_angle((0, 0, 1), 2.0)
        mq = Quat(-q.w, -q.x, -q.y, -q.z)
        t = Vec3(1.0, 0.0, 0.0)
        bones = [
            Bone("a", None, IDENT, RigidTransform(q, t)),
            Bone("b", None, IDENT, RigidTransform(mq, t)),
        ]
        p, n = (1.0, 2.0, 3.0), (0.0, 0.0, 1.0)
        (res,) = skin([(p, n)], bones, [[(0, 0.5), (1, 0.5)]])
        RT = refimpl.rigid_from_axis_angle((0, 0, 1), 2.0, (1.0, 0.0, 0.0))
        assert_vec_close(self, res.position, refimpl.rigid_apply(RT, p))
        assert_vec_close(self, res.normal, refimpl.rigid_rotate(RT, n))

    def test_blend_across_180_takes_short_path(self):
        # 两骨分别绕 z 转 179° 与 181°（即 -179°），等权混合应得 ~180°，
        # 而不是 ~0°（长路径）。
        a1 = math.radians(179.0)
        a2 = math.radians(181.0)
        bones = [
            Bone("a", None, IDENT, rigid((0, 0, 1), a1)),
            Bone("b", None, IDENT, rigid((0, 0, 1), a2)),
        ]
        (res,) = skin([((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))], bones, [[(0, 1.0), (1, 1.0)]])
        # 期望：绕 z 转 180°，点 (1,0,0) -> (-1,0,0)，法向 (0,1,0) -> (0,-1,0)
        assert_vec_close(self, res.position, (-1.0, 0.0, 0.0), tol=1e-6)
        assert_vec_close(self, res.normal, (0.0, -1.0, 0.0), tol=1e-6)

    def test_exact_180_single_bone(self):
        bones = [Bone("b", None, IDENT, rigid((0, 1, 0), math.pi))]
        (res,) = skin([((3.0, 1.0, -2.0), (1.0, 0.0, 0.0))], bones, [[(0, 1.0)]])
        assert_vec_close(self, res.position, (-3.0, 1.0, 2.0))
        assert_vec_close(self, res.normal, (-1.0, 0.0, 0.0))


class TestWeightScalingInvariance(unittest.TestCase):
    """权重缩放不变：同一顶点所有权乘同一正常数，结果不变。"""

    def test_scaling_invariant(self):
        m1 = rigid((0, 0, 1), 1.0, (1.0, 0.0, 0.0))
        m2 = rigid((0, 1, 0), -0.6, (0.0, 2.0, 1.0))
        bones = [Bone("a", None, IDENT, m1), Bone("b", None, IDENT, m2)]
        vert = [((0.3, 0.7, -0.2), (1.0, 0.0, 0.0))]
        base = skin(vert, bones, [[(0, 0.3), (1, 0.7)]])[0]
        for scale in (1e-3, 2.0, 137.5):
            scaled = skin(vert, bones, [[(0, 0.3 * scale), (1, 0.7 * scale)]])[0]
            assert_vec_close(self, scaled.position, base.position, tol=1e-9)
            assert_vec_close(self, scaled.normal, base.normal, tol=1e-9)

    def test_zero_weight_bone_ignored(self):
        m1 = rigid((0, 0, 1), 0.5)
        m2 = rigid((1, 0, 0), 2.0, (9.0, 9.0, 9.0))
        bones = [Bone("a", None, IDENT, m1), Bone("b", None, IDENT, m2)]
        vert = [((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        with_zero = skin(vert, bones, [[(0, 1.0), (1, 0.0)]])[0]
        without = skin(vert, bones, [[(0, 1.0)]])[0]
        assert_vec_close(self, with_zero.position, without.position)
        assert_vec_close(self, with_zero.normal, without.normal)


class TestHierarchy(unittest.TestCase):
    """骨骼树：世界变换沿父链累乘。"""

    def test_parent_motion_propagates(self):
        # 绑定全恒等；当前：父绕 z 转 90°，子局部平移 (1,0,0)。
        # 子的世界蒙皮变换 = R90z * T(1,0,0)：原点 -> (0,1,0)
        bones = [
            Bone("root", None, IDENT, rigid((0, 0, 1), math.pi / 2)),
            Bone("child", 0, IDENT, rigid((0, 0, 1), 0.0, (1.0, 0.0, 0.0))),
        ]
        (res,) = skin([((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))], bones, [[(1, 1.0)]])
        assert_vec_close(self, res.position, (0.0, 1.0, 0.0))
        assert_vec_close(self, res.normal, (0.0, 0.0, 1.0))

    def test_deep_chain_matches_matrix_reference(self):
        # 三层链 + 非平凡绑定姿态，与独立矩阵参考交叉验证
        binds = [
            rigid((0, 0, 1), 0.2, (0.0, 1.0, 0.0)),
            rigid((1, 0, 0), -0.3, (1.0, 0.0, 0.0)),
            rigid((0, 1, 0), 0.4, (0.0, 0.0, 1.0)),
        ]
        curs = [
            rigid((0, 0, 1), 1.1, (0.5, 0.0, 0.0)),
            rigid((1, 0, 0), 0.7, (0.0, 0.5, 0.0)),
            rigid((0, 1, 0), -0.9, (0.0, 0.0, 0.5)),
        ]
        bones = [
            Bone("b0", None, binds[0], curs[0]),
            Bone("b1", 0, binds[1], curs[1]),
            Bone("b2", 1, binds[2], curs[2]),
        ]
        p, n = (0.4, -0.8, 1.3), (0.0, 0.0, 1.0)
        w = [(0, 0.2), (1, 0.5), (2, 0.3)]
        (res,) = skin([(p, n)], bones, [w])

        # 独立参考：矩阵世界变换 -> M_i = W_cur_i * W_bind_i^{-1}
        def world(locals_):
            ws = []
            for i, loc in enumerate(locals_):
                T = refimpl.rigid_from_axis_angle(*_axis_angle_of(loc))
                ws.append(T if i == 0 else refimpl.rigid_compose(ws[_parent[i]], T))
            return ws

        _parent = [None, 0, 1]
        wb = world(binds)
        wc = world(curs)
        Ms = [
            refimpl.rigid_compose(wc[i], refimpl.rigid_inverse(wb[i]))
            for i in range(3)
        ]
        # DQ 混合的独立矩阵近似不可直接解析，故仅对单骨顶点做精确交叉验证，
        # 多骨顶点验证其为单位刚体结果（法向长度守恒、位置有限）。
        (res_single,) = skin([(p, n)], bones, [[(2, 1.0)]])
        assert_vec_close(self, res_single.position, refimpl.rigid_apply(Ms[2], p))
        assert_vec_close(self, res_single.normal, refimpl.rigid_rotate(Ms[2], n))
        nlen = math.sqrt(sum(c * c for c in res.normal))
        self.assertAlmostEqual(nlen, 1.0, delta=1e-9)
        self.assertTrue(all(math.isfinite(c) for c in res.position))

    def test_forest_multiple_roots(self):
        # 森林：两棵独立树均合法
        bones = [
            Bone("r1", None, IDENT, rigid((0, 0, 1), 0.5)),
            Bone("r2", None, IDENT, rigid((0, 0, 1), -0.5)),
        ]
        (res,) = skin([((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))], bones, [[(1, 1.0)]])
        expected = refimpl.mat_vec(
            refimpl.axis_angle_matrix((0, 0, 1), -0.5), (1.0, 0.0, 0.0)
        )
        assert_vec_close(self, res.position, expected)


def _axis_angle_of(rt):
    """从测试构造器还原轴角（仅供参考实现使用）。"""
    q = rt.rotation
    angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
    s = math.sin(angle / 2.0)
    if abs(s) < 1e-12:
        axis = (1.0, 0.0, 0.0)
        angle = 0.0
    else:
        axis = (q.x / s, q.y / s, q.z / s)
    return axis, angle, (rt.translation.x, rt.translation.y, rt.translation.z)


class TestRejections(unittest.TestCase):
    """非法输入必须抛出可定位错误，且不返回部分结果。"""

    def test_non_rigid_rejected(self):
        bad = RigidTransform(Quat(2.0, 0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
        bones = [Bone("b", None, IDENT, bad)]
        with self.assertRaises(NonRigidTransformError) as cm:
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 1.0)]])
        self.assertIn("bones[0].current.rotation", str(cm.exception))

    def test_cycle_rejected(self):
        bones = [
            Bone("a", 1, IDENT, IDENT),
            Bone("b", 0, IDENT, IDENT),
        ]
        with self.assertRaises(HierarchyError) as cm:
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 1.0)]])
        self.assertIn("环", str(cm.exception))

    def test_self_loop_rejected(self):
        bones = [Bone("a", 0, IDENT, IDENT)]
        with self.assertRaises(HierarchyError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 1.0)]])

    def test_negative_weight_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError) as cm:
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, -0.5)]])
        self.assertIn("weights[0][0]", str(cm.exception))

    def test_zero_total_weight_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 0.0)]])

    def test_nan_inf_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(InvalidInputError) as cm:
            skin([((float("nan"), 0, 0), (0, 0, 1))], bones, [[(0, 1.0)]])
        self.assertIn("vertices[0].position", str(cm.exception))
        bad_t = RigidTransform(Quat(1.0, 0.0, 0.0, 0.0), Vec3(float("inf"), 0.0, 0.0))
        with self.assertRaises(InvalidInputError):
            skin([((0, 0, 0), (0, 0, 1))], [Bone("b", None, IDENT, bad_t)], [[(0, 1.0)]])

    def test_bool_rejected_as_int_and_number(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(InvalidInputError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(True, 1.0)]])
        with self.assertRaises(InvalidInputError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, True)]])
        with self.assertRaises(InvalidInputError):
            skin([((0, 0, 0), (0, 0, 1))], [Bone("b", True, IDENT, IDENT)], [[(0, 1.0)]])

    def test_too_many_bones(self):
        bones = [Bone(i, None, IDENT, IDENT) for i in range(33)]
        with self.assertRaises(HierarchyError) as cm:
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 1.0)]])
        self.assertIn("32", str(cm.exception))
        # 32 根合法
        ok = skin([((0, 0, 0), (0, 0, 1))], bones[:32], [[(31, 1.0)]])
        self.assertEqual(len(ok), 1)

    def test_parent_out_of_range(self):
        bones = [Bone("b", 5, IDENT, IDENT)]
        with self.assertRaises(HierarchyError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 1.0)]])

    def test_weight_bone_index_out_of_range(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(1, 1.0)]])

    def test_duplicate_bone_index(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(WeightError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 0.5), (0, 0.5)]])

    def test_weights_length_mismatch(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(InvalidInputError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[(0, 1.0)], [(0, 1.0)]])

    def test_empty_inputs(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(InvalidInputError):
            skin([], bones, [])
        with self.assertRaises(HierarchyError):
            skin([((0, 0, 0), (0, 0, 1))], [], [[]])
        with self.assertRaises(WeightError):
            skin([((0, 0, 0), (0, 0, 1))], bones, [[]])

    def test_zero_normal_rejected(self):
        bones = [Bone("b", None, IDENT, IDENT)]
        with self.assertRaises(InvalidInputError):
            skin([((0, 0, 0), (0, 0, 0))], bones, [[(0, 1.0)]])

    def test_no_partial_results_on_failure(self):
        # 第二个顶点非法 -> 整体抛错，不返回任何结果
        bones = [Bone("b", None, IDENT, IDENT)]
        verts = [((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (0, 0, 1))]
        weights = [[(0, 1.0)], [(0, -1.0)]]
        with self.assertRaises(WeightError):
            skin(verts, bones, weights)


class TestOutputContract(unittest.TestCase):
    def test_result_types_and_count(self):
        bones = [Bone("b", None, IDENT, rigid((0, 0, 1), 0.3))]
        verts = [((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0))]
        res = skin(verts, bones, [[(0, 1.0)]] * 3)
        self.assertEqual(len(res), 3)
        for r in res:
            self.assertIsInstance(r, SkinResult)
            self.assertIsInstance(r.position, Vec3)
            self.assertIsInstance(r.normal, Vec3)

    def test_inputs_not_mutated(self):
        bones = [Bone("b", None, IDENT, rigid((0, 0, 1), 0.3))]
        verts = [((1, 0, 0), (0, 1, 0))]
        weights = [[(0, 1.0)]]
        snapshot = (repr(verts), repr(bones), repr(weights))
        skin(verts, bones, weights)
        self.assertEqual(snapshot, (repr(verts), repr(bones), repr(weights)))

    def test_normal_length_preserved(self):
        m = rigid((1, 2, 3), 1.3, (4.0, 5.0, 6.0))
        bones = [Bone("b", None, IDENT, m)]
        (res,) = skin([((0, 0, 0), (0, 0, 2.0))], bones, [[(0, 1.0)]])
        nlen = math.sqrt(sum(c * c for c in res.normal))
        self.assertAlmostEqual(nlen, 2.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
