"""功能正确性测试。

期望值来源（独立参考，见 tests/_reference.py，全程不 import dqskin）：

* 单骨：直接用罗德里格斯公式与 4x4 矩阵乘/刚体逆推出的解析刚体结果；
* 同向骨骼：经典矩阵线性混合蒙皮（LBS）——与双四元数完全不同的代码路径；
* 近 180 度最短路径：轴角解析中点。

误差口径：常规场景断言绝对误差 <= 1e-9（输入量级 O(1)）；
近 180 度混合断言 <= 1e-8；绑定姿态不动断言 <= 1e-12。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqskin import (  # noqa: E402
    Bone,
    RigidTransform,
    skin,
)

from tests import _reference as ref  # noqa: E402

I = ref.axis_angle_matrix((0.0, 0.0, 1.0), 0.0)
VEC3_TOL = 1e-9
NEAR_180_TOL = 1e-8
BIND_POSE_TOL = 1e-12


def _bones_from_spec(spec):
    return [
        Bone(RigidTransform(b), RigidTransform(c), parent=p)
        for (b, c, p) in spec
    ]


def _dict_weights(weight_lists):
    return [{i: w for i, w in wl} for wl in weight_lists]


def _assert_vec3(testcase, got, expected, tol=VEC3_TOL):
    for k, name in enumerate("xyz"):
        testcase.assertAlmostEqual(
            got[k], expected[k], delta=tol,
            msg=f"component {name}: got {got[k]!r}, expected {expected[k]!r}",
        )


class SingleBoneAnalyticTests(unittest.TestCase):
    """单骨蒙皮必须等于解析刚体变换 current_world * inv(bind_world)。"""

    def _check(self, bind, current, points, normals=None):
        bones = _bones_from_spec([(bind, current, None)])
        wl = [[(0, 1.0)] for _ in points]
        result = skin(points, bones, _dict_weights(wl), normals=normals)
        rel = ref.mat_mul(current, ref.inverse_rigid(bind))
        for i, p in enumerate(points):
            _assert_vec3(self, result.vertices[i], ref.mat_vec(rel, p))
            if normals is not None:
                _assert_vec3(self, result.normals[i], ref.mat_dir(rel, normals[i]))
                # 刚体旋转保持法向长度（输入未必单位化）。
                n_in = normals[i]
                n_out = result.normals[i]
                self.assertAlmostEqual(
                    math.sqrt(sum(c * c for c in n_out)),
                    math.sqrt(sum(c * c for c in n_in)),
                    delta=1e-12,
                )

    def test_identity(self):
        pts = [(1.0, 2.0, 3.0), (-4.0, 0.5, -7.0)]
        self._check(I, I, pts)

    def test_quarter_turn_about_z(self):
        cur = ref.axis_angle_matrix((0.0, 0.0, 1.0), math.pi / 2)
        pts = [(1.0, 0.0, 0.0), (0.0, 1.0, 2.0), (-3.0, 2.0, -1.0)]
        ns = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
        self._check(I, cur, pts, ns)

    def test_rotation_plus_translation(self):
        cur = ref.axis_angle_matrix(
            (0.0, 1.0, 0.0), 0.7, translation=(2.0, -3.0, 5.0)
        )
        pts = [(1.5, -0.5, 2.0), (0.0, 0.0, 0.0)]
        self._check(I, cur, pts)

    def test_axis_111_120_degrees(self):
        cur = ref.axis_angle_matrix((1.0, 1.0, 1.0), 2 * math.pi / 3)
        pts = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 2.0, 3.0)]
        ns = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self._check(I, cur, pts, ns)

    def test_exact_180_about_general_axis(self):
        # 180 度正是“trace 开根号”类转换的退化点；这里必须仍然精确。
        cur = ref.axis_angle_matrix((1.0, 1.0, 1.0), math.pi)
        pts = [(1.0, 0.0, 0.0), (-2.0, 3.0, 1.5)]
        self._check(I, cur, pts)
        cur2 = ref.axis_angle_matrix(
            (1.0, 2.0, 3.0), math.pi, translation=(1.0, -1.0, 2.0)
        )
        self._check(I, cur2, pts)

    def test_near_180_single_bone(self):
        for delta in (1e-3, 1e-6, 1e-9):
            with self.subTest(delta=delta):
                cur = ref.axis_angle_matrix(
                    (0.0, 0.0, 1.0), math.pi - delta
                )
                self._check(I, cur, [(1.0, 0.5, -0.25)])
                cur = ref.axis_angle_matrix(
                    (0.0, 0.0, 1.0), -(math.pi - delta)
                )
                self._check(I, cur, [(1.0, 0.5, -0.25)])

    def test_non_identity_bind(self):
        bind = ref.axis_angle_matrix(
            (1.0, 0.0, 0.0), 0.5, translation=(1.0, 2.0, 3.0)
        )
        current = ref.axis_angle_matrix(
            (0.0, 1.0, 0.0), 1.1, translation=(-2.0, 0.0, 4.0)
        )
        pts = [(2.0, -1.0, 0.5), (0.0, 0.0, 0.0)]
        ns = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self._check(bind, current, pts, ns)

    def test_translation_does_not_move_normals(self):
        cur = ref.axis_angle_matrix(
            (0.0, 0.0, 1.0), 0.3, translation=(10.0, -20.0, 30.0)
        )
        n = (0.6, -0.8, 0.0)
        result = skin(
            [(0.0, 0.0, 0.0)],
            _bones_from_spec([(I, cur, None)]),
            [{0: 1.0}],
            normals=[n],
        )
        _assert_vec3(self, result.normals[0], ref.mat_dir(cur, n))


class HierarchyTests(unittest.TestCase):
    """父链累乘世界变换的正确性（含非平凡 bind 与 current 局部链）。"""

    def test_chain_single_influences(self):
        # 根 -> 骨1 -> 骨2；每根骨骼的 bind/current 局部都含旋转和平移。
        b0 = ref.axis_angle_matrix((0, 0, 1), 0.4, (1.0, 0.0, 0.0))
        b1 = ref.axis_angle_matrix((0, 1, 0), 0.2, (0.0, 2.0, 0.0))
        b2 = ref.axis_angle_matrix((1, 0, 0), 0.6, (0.0, 0.0, -1.0))
        c0 = ref.axis_angle_matrix((0, 0, 1), 0.9, (1.5, 0.0, 0.0))
        c1 = ref.axis_angle_matrix((0, 1, 0), -0.3, (0.0, 1.5, 0.0))
        c2 = ref.axis_angle_matrix((1, 1, 1), 0.2, (0.0, 0.0, -0.5))
        spec = [(b0, c0, None), (b1, c1, 0), (b2, c2, 1)]
        pts = [(1.0, -2.0, 3.0), (0.5, 0.5, 0.5), (-1.0, 4.0, 0.0)]
        wl = [[(i, 1.0)] for i in range(3)]
        result = skin(pts, _bones_from_spec(spec), _dict_weights(wl))
        rels = ref.lbs_relative_matrices(spec)
        for i, p in enumerate(pts):
            _assert_vec3(self, result.vertices[i], ref.mat_vec(rels[i], p))

    def test_shuffled_index_order(self):
        # 子骨骼排在根之前（索引 0 是子，parent=1；索引 1 是根）。
        b0 = ref.axis_angle_matrix((1, 0, 0), 0.2, (0.0, 0.0, 4.0))
        c0 = ref.axis_angle_matrix((0, 0, 1), -0.5, (3.0, 0.0, 0.0))
        b1 = ref.axis_angle_matrix((0, 0, 1), 0.3, (1.0, 2.0, 3.0))
        c1 = ref.axis_angle_matrix((0, 1, 0), 0.7, (2.0, -1.0, 0.0))
        shuffled = [(b1, c1, 1), (b0, c0, None)]
        pts = [(1.5, -2.0, 0.7), (0.0, 0.0, 0.0)]
        # 每顶点只受一根骨骼影响：直接对独立的逐骨世界累乘相对矩阵参考。
        wl = [[(0, 1.0)], [(1, 1.0)]]
        ns = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
        result = skin(
            pts, _bones_from_spec(shuffled), _dict_weights(wl), normals=ns
        )
        rels = ref.lbs_relative_matrices(shuffled)
        for i, p in enumerate(pts):
            _assert_vec3(self, result.vertices[i], ref.mat_vec(rels[i], p))
            _assert_vec3(self, result.normals[i], ref.mat_dir(rels[i], ns[i]))

    def test_forest_multiple_roots(self):
        b0 = ref.axis_angle_matrix((0, 0, 1), 0.3, (1.0, 0.0, 0.0))
        b1 = ref.axis_angle_matrix((0, 1, 0), 0.5, (0.0, 2.0, 0.0))
        c0 = ref.axis_angle_matrix((0, 0, 1), 0.8, (2.0, 0.0, 0.0))
        c1 = ref.axis_angle_matrix((0, 1, 0), 0.1, (0.0, -1.0, 0.0))
        spec = [(b0, c0, None), (b1, c1, None)]
        pts = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
        wl = [[(0, 1.0)], [(1, 1.0)]]
        result = skin(pts, _bones_from_spec(spec), _dict_weights(wl))
        rels = ref.lbs_relative_matrices(spec)
        for i, p in enumerate(pts):
            _assert_vec3(self, result.vertices[i], ref.mat_vec(rels[i], p))


class CoalitionedBonesTests(unittest.TestCase):
    """同向（同旋转、不同平移）骨骼：DLB 必须等于解析线性结果与 LBS 参考。"""

    def _spec(self, rot, translations):
        # bind 均为单位，current 为同一旋转 + 各自平移。
        return [
            (I, ref.axis_angle_matrix(rot[0], rot[1], t), None)
            for t in translations
        ]

    def test_two_bones_midpoint_translation(self):
        spec = self._spec(
            ((0.0, 0.0, 1.0), 0.0),
            [(0.0, 0.0, 0.0), (4.0, 2.0, 0.0)],
        )
        pts = [(1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]
        wl = [[(0, 1.0), (1, 1.0)] for _ in pts]
        result = skin(pts, _bones_from_spec(spec), _dict_weights(wl))
        lbs_v, _ = ref.linear_blend_skin(pts, spec, wl)
        for i, p in enumerate(pts):
            _assert_vec3(self, result.vertices[i], lbs_v[i])
            # 解析：中点平移
            _assert_vec3(self, result.vertices[i], (p[0] + 2.0, p[1] + 1.0, 0.0))

    def test_shared_rotation_weighted_translations(self):
        rot = ((0.0, 1.0, 0.0), 0.8)
        t0, t1 = (1.0, 0.0, -2.0), (-3.0, 2.0, 1.0)
        spec = self._spec(rot, [t0, t1])
        pts = [(1.0, -1.0, 2.0), (0.0, 0.0, 0.0), (2.0, 3.0, -1.0)]
        for a, b in ((0.3, 0.7), (0.1, 0.9), (2.0, 3.0)):
            with self.subTest(a=a, b=b):
                wl = [[(0, a), (1, b)] for _ in pts]
                result = skin(
                    pts, _bones_from_spec(spec), _dict_weights(wl),
                    normals=[(1.0, 0.0, 0.0)] * len(pts),
                )
                lbs_v, lbs_n = ref.linear_blend_skin(
                    pts, spec, wl, normals=[(1.0, 0.0, 0.0)] * len(pts)
                )
                for i in range(len(pts)):
                    _assert_vec3(self, result.vertices[i], lbs_v[i])
                    _assert_vec3(self, result.normals[i], lbs_n[i])

    def test_three_bones_same_rotation(self):
        rot = ((1.0, 1.0, 1.0), 0.5)
        spec = self._spec(
            rot,
            [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0)],
        )
        pts = [(1.0, 2.0, -1.0)]
        wl = [[(0, 0.25), (1, 0.5), (2, 0.25)]]
        result = skin(pts, _bones_from_spec(spec), _dict_weights(wl))
        lbs_v, _ = ref.linear_blend_skin(pts, spec, wl)
        _assert_vec3(self, result.vertices[0], lbs_v[0])


class Near180BlendTests(unittest.TestCase):
    """近 180 度：单骨精确、两骨最短路径混合（符号修正）。"""

    def test_plus_minus_near_pi_takes_short_arc(self):
        # +179.99..° 与 -179.99..° 绕同一轴：作为刚体姿态几乎相同（差 ~0.01°），
        # 但四元数表示分别位于实部正/负两侧。符号修正后等权混合必须给出
        # 绕该轴 180° 的旋转（最短路径），而不是 LBS 那种绕回单位姿态的结果。
        for delta in (1e-4, 1e-7):
            with self.subTest(delta=delta):
                m0 = ref.axis_angle_matrix(
                    (0.0, 0.0, 1.0), math.pi - delta,
                    translation=(0.0, 0.0, 0.0),
                )
                m1 = ref.axis_angle_matrix(
                    (0.0, 0.0, 1.0), -(math.pi - delta),
                    translation=(4.0, 0.0, 0.0),
                )
                # 两根独立骨骼（同为根）：作为姿态它们只差 2δ 的短弧，
                # 但四元数实部符号相反——正是符号修正的用武之地。
                spec = [(I, m0, None), (I, m1, None)]
                pts = [(1.0, 0.0, 0.0), (0.0, 1.0, 2.0)]
                wl = [[(0, 1.0), (1, 1.0)] for _ in pts]
                result = skin(pts, _bones_from_spec(spec), _dict_weights(wl))
                # 独立闭式期望：
                #   s=sin(δ/2), c=cos(δ/2)；实部混合恰为绕 z 轴 π；
                #   双部（含被翻转的 m1 平移 (4,0,0)）给出平移 (2c, -2s, 0)。
                s = math.sin(delta / 2.0)
                c = math.cos(delta / 2.0)

                def expected(p):
                    x, y, z = p
                    return (-x + 2 * c, -y - 2 * s, z)

                for i, p in enumerate(pts):
                    _assert_vec3(
                        self, result.vertices[i],
                        expected(p), tol=NEAR_180_TOL,
                    )

    def test_exact_antipodal_rotations_are_deterministic(self):
        # 两骨分别为绕 z 轴 π 与绕 x 轴 π：实部四元数内积恰为 0
        # （旋转流形上的对跖点，最短路径不唯一）。实现不得崩溃或给 NaN，
        # 结果必须等于“锚点规则”下的确定混合：q = (z+x)/|z+x|
        # = (0,1,0,1)/sqrt(2)，即绕轴 (1,0,1)/sqrt(2) 转 π。
        mz = ref.axis_angle_matrix((0.0, 0.0, 1.0), math.pi)
        mx = ref.axis_angle_matrix((1.0, 0.0, 0.0), math.pi)
        spec = [(I, mz, None), (I, mx, None)]
        pts = [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
        wl = [[(0, 0.5), (1, 0.5)] for _ in pts]
        r1 = skin(pts, _bones_from_spec(spec), _dict_weights(wl))
        axis = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
        expected = ref.axis_angle_matrix(axis, math.pi)
        for i, p in enumerate(pts):
            _assert_vec3(self, r1.vertices[i], ref.mat_vec(expected, p))
        # 确定性：重复调用逐位一致；交换权重主从（bone1 为锚点）仍合法确定。
        r2 = skin(pts, _bones_from_spec(spec), _dict_weights(wl))
        self.assertEqual(r1.vertices, r2.vertices)
        flipped_w = [[(0, 0.4), (1, 0.6)] for _ in pts]
        r3 = skin(pts, _bones_from_spec(spec), _dict_weights(flipped_w))
        for row in r3.vertices:
            for c in row:
                self.assertTrue(math.isfinite(c))

    def test_continuous_across_180_single_bone(self):
        pts = [(1.0, 0.3, -0.2), (0.0, 0.0, 0.0)]
        bones_pm = {}
        for tag, angle in (("plus", math.pi - 1e-10),
                           ("minus", -(math.pi - 1e-10))):
            spec = [(I, ref.axis_angle_matrix((0, 0, 1), angle), None)]
            bones_pm[tag] = skin(
                pts, _bones_from_spec(spec),
                [[(0, 1.0)] for _ in pts],
            )
        for i, p in enumerate(pts):
            # +π-ε 与 -π+ε 作用于向量只差 2ε 量级
            a = bones_pm["plus"].vertices[i]
            b = bones_pm["minus"].vertices[i]
            for k in range(3):
                self.assertAlmostEqual(a[k], b[k], delta=1e-8)


class WeightScalingInvarianceTests(unittest.TestCase):
    """权重整体缩放不改变结果（实现内部归一化，契约允许任意正尺度）。"""

    def test_scale_invariant(self):
        spec = [
            (I, ref.axis_angle_matrix((0, 0, 1), 0.5, (1.0, 0.0, 0.0)), None),
            (I, ref.axis_angle_matrix((0, 1, 0), 0.9, (0.0, 2.0, 0.0)), 0),
            (I, ref.axis_angle_matrix((1, 0, 0), 0.2, (0.0, 0.0, 3.0)), 1),
        ]
        pts = [(1.0, -1.0, 2.0), (0.5, 0.5, 0.5)]
        base = [[(0, 0.2), (1, 0.5), (2, 0.3)] for _ in pts]
        scales = (1.0, 2.5, 100.0, 0.01, 7.0)
        results = []
        for s in scales:
            wl = [[(i, w * s) for i, w in vw] for vw in base]
            results.append(
                skin(pts, _bones_from_spec(spec), _dict_weights(wl)).vertices
            )
        for vi in range(len(pts)):
            for k in range(3):
                values = [r[vi][k] for r in results]
                spread = max(values) - min(values)
                self.assertAlmostEqual(spread, 0.0, delta=1e-10)

    def test_sequence_weight_form_equals_dict_form(self):
        spec = [
            (I, ref.axis_angle_matrix((0, 0, 1), 0.5, (1.0, 0.0, 0.0)), None),
            (I, ref.axis_angle_matrix((0, 1, 0), 0.9, (0.0, 2.0, 0.0)), 0),
        ]
        pts = [(1.0, 1.0, 1.0)]
        d = skin(pts, _bones_from_spec(spec), [{0: 0.4, 1: 0.6}]).vertices
        s = skin(pts, _bones_from_spec(spec), [[(0, 0.4), (1, 0.6)]]).vertices
        _assert_vec3(self, d[0], s[0], tol=1e-12)


class BindPoseTests(unittest.TestCase):
    """绑定姿态（current == bind）下任何层级、任何权重都不动。"""

    def test_rest_pose_identity(self):
        m0 = ref.axis_angle_matrix((0, 0, 1), 0.7, (1.0, 2.0, 3.0))
        m1 = ref.axis_angle_matrix((0, 1, 0), 0.4, (-2.0, 1.0, 0.0))
        m2 = ref.axis_angle_matrix((1, 0, 0), 1.1, (0.0, -3.0, 2.0))
        spec = [(m0, m0, None), (m1, m1, 0), (m2, m2, 1)]
        pts = [(1.0, -2.0, 3.0), (0.0, 0.0, 0.0), (-4.0, 5.0, -6.0)]
        ns = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        wl = [
            [(0, 1.0)],
            [(0, 0.3), (1, 0.3), (2, 0.4)],
            [(2, 0.9)],
        ]
        result = skin(
            pts, _bones_from_spec(spec), _dict_weights(wl), normals=ns
        )
        for i, p in enumerate(pts):
            _assert_vec3(self, result.vertices[i], p, tol=BIND_POSE_TOL)
            _assert_vec3(self, result.normals[i], ns[i], tol=BIND_POSE_TOL)


class RoundTripAndDeterminismTests(unittest.TestCase):
    """补充性质：往返一致（不作主要期望值来源）与确定性。"""

    def test_roundtrip_swap_bind_current(self):
        b0 = ref.axis_angle_matrix((1, 0, 0), 0.6, (1.0, 0.0, 0.0))
        b1 = ref.axis_angle_matrix((0, 1, 0), 0.3, (0.0, 1.0, 0.0))
        c0 = ref.axis_angle_matrix((0, 0, 1), 0.9, (2.0, 0.0, 0.0))
        c1 = ref.axis_angle_matrix((1, 1, 1), 0.2, (0.0, 0.0, 2.0))
        pts = [(1.0, -1.0, 2.0), (0.0, 0.0, 0.0)]
        wl = [[(0, 0.6), (1, 0.4)] for _ in pts]
        forward = skin(
            pts,
            _bones_from_spec([(b0, c0, None), (b1, c1, 0)]),
            _dict_weights(wl),
        ).vertices
        back = skin(
            forward,
            _bones_from_spec([(c0, b0, None), (c1, b1, 0)]),
            _dict_weights(wl),
        ).vertices
        for i, p in enumerate(pts):
            _assert_vec3(self, back[i], p, tol=1e-8)

    def test_deterministic(self):
        spec = [
            (I, ref.axis_angle_matrix((0, 0, 1), 0.7, (1.0, 0.0, 0.0)), None),
            (I, ref.axis_angle_matrix((0, 1, 0), 0.4, (0.0, 1.0, 0.0)), 0),
        ]
        pts = [(1.0, 2.0, 3.0)]
        w = [{0: 0.3, 1: 0.7}]
        r1 = skin(pts, _bones_from_spec(spec), w)
        r2 = skin(pts, _bones_from_spec(spec), w)
        self.assertEqual(r1.vertices, r2.vertices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
