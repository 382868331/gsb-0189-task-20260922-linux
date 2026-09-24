"""命名骨架 / 拓扑排序 / 全局姿态 / skin_named 的单元测试。

期望值来源：
- 解析手算（三层链父旋转带动子节点、归一化四元数、180° 父旋转）；
- tests/refimpl.py 中与 dqskin 完全独立的矩阵实现（全局姿态交叉验证）。
任何期望值都不通过调用被测实现自身得到。
"""

from __future__ import annotations

import math
import unittest

from dqskin import (
    Bone,
    GlobalPose,
    HierarchyError,
    InvalidInputError,
    NonRigidTransformError,
    PoseNode,
    QUAT_NORMALIZE_NORMALIZE,
    QUAT_NORMALIZE_REJECT,
    Quat,
    RigidTransform,
    Vec3,
    WeightError,
    resolve_global_poses,
    skin,
    skin_named,
)

import refimpl


IDENT = RigidTransform(Quat(1.0, 0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))


def quat_axis_angle(axis, angle):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az)
    h = angle / 2.0
    s = math.sin(h) / n
    return Quat(math.cos(h), ax * s, ay * s, az * s)


def rigid(axis, angle, t=(0.0, 0.0, 0.0)):
    return RigidTransform(quat_axis_angle(axis, angle), Vec3(*t))


def assert_vec_close(tc, actual, expected, tol=1e-9):
    tc.assertAlmostEqual(actual[0], expected[0], delta=tol)
    tc.assertAlmostEqual(actual[1], expected[1], delta=tol)
    tc.assertAlmostEqual(actual[2], expected[2], delta=tol)


def matrix_of(rt: RigidTransform):
    """从 RigidTransform 还原 (R, t) 参考量（仅测试侧，不调用 dqskin 运算）。"""
    q = rt.rotation
    angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
    s = math.sin(angle / 2.0)
    if abs(s) < 1e-12:
        axis = (1.0, 0.0, 0.0)
        angle = 0.0
    else:
        axis = (q.x / s, q.y / s, q.z / s)
    return refimpl.rigid_from_axis_angle(
        axis, angle, (rt.translation.x, rt.translation.y, rt.translation.z)
    )


class TestTopologicalOrder(unittest.TestCase):
    """Kahn 拓扑排序：父必先于子，同层按输入顺序，结果确定。"""

    def test_shuffled_input_sorted_parent_first(self):
        nodes = [
            PoseNode("hand", "spine", IDENT),
            PoseNode("root", None, IDENT),
            PoseNode("spine", "root", IDENT),
        ]
        pose = resolve_global_poses(nodes)
        self.assertIsInstance(pose, GlobalPose)
        self.assertEqual(pose.order, ("root", "spine", "hand"))
        self.assertEqual(set(pose.transforms), {"root", "spine", "hand"})

    def test_sibling_order_follows_input_order(self):
        nodes = [
            PoseNode("b", "r", IDENT),
            PoseNode("a", "r", IDENT),
            PoseNode("r", None, IDENT),
        ]
        pose = resolve_global_poses(nodes)
        self.assertEqual(pose.order, ("r", "b", "a"))

    def test_forest_multiple_roots(self):
        nodes = [
            PoseNode("r1", None, rigid((0, 0, 1), 0.5)),
            PoseNode("r2", None, rigid((0, 0, 1), -0.5)),
        ]
        pose = resolve_global_poses(nodes)
        self.assertEqual(pose.order, ("r1", "r2"))
        self.assertEqual(len(pose.transforms), 2)

    def test_tuple_and_dict_nodes_accepted(self):
        pose = resolve_global_poses(
            [
                ("root", None, IDENT),
                {"name": "child", "parent": "root", "local": IDENT},
            ]
        )
        self.assertEqual(pose.order, ("root", "child"))

    def test_integer_names_accepted(self):
        pose = resolve_global_poses([PoseNode(7, None, IDENT)])
        self.assertEqual(pose.order, (7,))


class TestGlobalPoseAnalytic(unittest.TestCase):
    """三层骨架：父节点旋转后子节点位置（手算）。"""

    def three_layer_nodes(self, root_local=None, spine_local=None, hand_local=None):
        # 绑定全恒等；当前：root 绕 z +90°，spine 局部偏移 (0,2,0)，
        # hand 相对 spine 再偏移 (0,1,0)。
        return [
            PoseNode("root", None, root_local if root_local is not None
                     else rigid((0, 0, 1), math.pi / 2)),
            PoseNode("spine", "root", spine_local if spine_local is not None
                     else rigid((0, 0, 1), 0.0, (0.0, 2.0, 0.0))),
            PoseNode("hand", "spine", hand_local if hand_local is not None
                     else rigid((0, 0, 1), 0.0, (0.0, 1.0, 0.0))),
        ]

    def test_parent_rotation_carries_children(self):
        # 手算：Rz90 把 +y 送到 -x。
        # spine 世界平移 = Rz90(0,2,0) = (-2,0,0)
        # hand  世界平移 = Rz90(0,3,0) = (-3,0,0)
        pose = resolve_global_poses(self.three_layer_nodes())
        self.assertAlmostEqual(pose.transforms["spine"].translation.x, -2.0, delta=1e-9)
        self.assertAlmostEqual(pose.transforms["spine"].translation.y, 0.0, delta=1e-9)
        self.assertAlmostEqual(pose.transforms["hand"].translation.x, -3.0, delta=1e-9)
        self.assertAlmostEqual(pose.transforms["hand"].translation.y, 0.0, delta=1e-9)
        self.assertAlmostEqual(pose.transforms["hand"].translation.z, 0.0, delta=1e-9)

    def test_skin_named_three_layer_child_position(self):
        nodes = self.three_layer_nodes()
        binds = [PoseNode(n.name, n.parent, IDENT) for n in nodes]
        # 顶点在原点，法向 +z，完全受 hand 影响 -> 世界平移 (-3,0,0)，法向不变
        (res,) = skin_named(
            [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
            binds,
            nodes,
            [[("hand", 1.0)]],
        )
        assert_vec_close(self, res.position, (-3.0, 0.0, 0.0))
        assert_vec_close(self, res.normal, (0.0, 0.0, 1.0))

    def test_child_local_geometry_rotated_by_parent(self):
        # spine 处的局部点 (0,2,0)：经父 Rz90 -> (-2,0,0)
        nodes = self.three_layer_nodes()
        binds = [PoseNode(n.name, n.parent, IDENT) for n in nodes]
        (res,) = skin_named(
            [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))],
            binds,
            nodes,
            [[("spine", 1.0)]],
        )
        assert_vec_close(self, res.position, (-2.0, 0.0, 0.0))
        # 法向 +x 经 Rz90 -> +y
        assert_vec_close(self, res.normal, (0.0, 1.0, 0.0))

    def test_parent_180_rotation_flips_child_offset(self):
        # 父绕 y 180°，子局部偏移 (1,0,0) -> 世界 (-1,0,0)
        nodes = [
            PoseNode("root", None, rigid((0, 1, 0), math.pi)),
            PoseNode("child", "root", rigid((0, 1, 0), 0.0, (1.0, 0.0, 0.0))),
        ]
        binds = [PoseNode("root", None, IDENT), PoseNode("child", "root", IDENT)]
        (res,) = skin_named(
            [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
            binds,
            nodes,
            [[("child", 1.0)]],
        )
        assert_vec_close(self, res.position, (-1.0, 0.0, 0.0))
        assert_vec_close(self, res.normal, (0.0, 1.0, 0.0))

    def test_global_poses_match_matrix_reference(self):
        # 三层链 + 各层局部旋转与平移，与独立矩阵参考逐层交叉验证
        locs = [
            rigid((0, 0, 1), 0.6, (0.0, 1.0, 0.0)),
            rigid((1, 0, 0), -0.4, (1.0, 0.0, 0.0)),
            rigid((0, 1, 0), 0.9, (0.0, 0.0, 1.0)),
        ]
        nodes = [
            PoseNode("a", None, locs[0]),
            PoseNode("b", "a", locs[1]),
            PoseNode("c", "b", locs[2]),
        ]
        pose = resolve_global_poses(nodes)

        T0 = matrix_of(locs[0])
        T1 = refimpl.rigid_compose(T0, matrix_of(locs[1]))
        T2 = refimpl.rigid_compose(T1, matrix_of(locs[2]))
        expected = {"a": T0, "b": T1, "c": T2}
        p = (0.37, -1.23, 0.91)
        for name, T in expected.items():
            got = pose.transforms[name]
            assert_vec_close(self, _apply_rt(got, p), refimpl.rigid_apply(T, p))

    def test_bind_pose_invariance(self):
        # current == bind（同一组命名节点）-> 顶点不动
        locs = [
            rigid((0, 0, 1), 0.6, (0.0, 1.0, 0.0)),
            rigid((1, 0, 0), -0.4, (1.0, 0.0, 0.0)),
            rigid((0, 1, 0), 0.9, (0.0, 0.0, 1.0)),
        ]
        nodes = [
            PoseNode("a", None, locs[0]),
            PoseNode("b", "a", locs[1]),
            PoseNode("c", "b", locs[2]),
        ]
        verts = [
            ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0)),
            ((-4.0, 0.5, 2.0), (1.0, 0.0, 0.0)),
        ]
        weights = [[("a", 0.3), ("c", 0.7)], [("b", 1.0)]]
        for res, (p, n) in zip(skin_named(verts, nodes, nodes, weights), verts):
            assert_vec_close(self, res.position, p)
            assert_vec_close(self, res.normal, n)


def _apply_rt(rt: RigidTransform, p):
    """测试侧独立四元数作用（与 core._qrot 无关的直接展开，用于交叉核对取值）。"""
    # 这里直接用 refimpl 的 Rodrigues 矩阵，避免借用被测运算。
    T = matrix_of(rt)
    return refimpl.rigid_apply(T, p)


class TestEquivalentToLegacySkin(unittest.TestCase):
    """命名入口拍平后必须与旧下标入口（同一层级数学）给出一致结果，
    并与独立矩阵参考一致。"""

    def test_named_matches_index_skeleton_and_matrix_ref(self):
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
        legacy_bones = [
            Bone("b0", None, binds[0], curs[0]),
            Bone("b1", 0, binds[1], curs[1]),
            Bone("b2", 1, binds[2], curs[2]),
        ]
        named_binds = [
            PoseNode("b0", None, binds[0]),
            PoseNode("b1", "b0", binds[1]),
            PoseNode("b2", "b1", binds[2]),
        ]
        named_curs = [
            PoseNode("b2", "b1", curs[2]),  # 故意乱序：拓扑排序负责还原
            PoseNode("b0", None, curs[0]),
            PoseNode("b1", "b0", curs[1]),
        ]
        p, n = (0.4, -0.8, 1.3), (0.0, 0.0, 1.0)
        w_named = [[("b0", 0.2), ("b1", 0.5), ("b2", 0.3)]]
        legacy = skin([(p, n)], legacy_bones, [[(0, 0.2), (1, 0.5), (2, 0.3)]])
        named = skin_named([(p, n)], named_binds, named_curs, w_named)
        assert_vec_close(self, named[0].position, legacy[0].position)
        assert_vec_close(self, named[0].normal, legacy[0].normal)

        # 单骨顶点与矩阵参考精确核对
        (single,) = skin_named([(p, n)], named_binds, named_curs, [[("b2", 1.0)]])
        # 参考：世界变换矩阵逐层累乘
        def world(locals_):
            ws = []
            parents = [None, 0, 1]
            for i, loc in enumerate(locals_):
                T = matrix_of(loc)
                ws.append(T if parents[i] is None else refimpl.rigid_compose(ws[parents[i]], T))
            return ws

        Ms = [
            refimpl.rigid_compose(world(curs)[i], refimpl.rigid_inverse(world(binds)[i]))
            for i in range(3)
        ]
        assert_vec_close(self, single.position, refimpl.rigid_apply(Ms[2], p))
        assert_vec_close(self, single.normal, refimpl.rigid_rotate(Ms[2], n))


class TestQuaternionNormalization(unittest.TestCase):
    """规范化规则固定：reject（默认）与 normalize（q/‖q‖），零四元数拒绝。"""

    def test_default_mode_is_reject(self):
        bad = Quat(2.0, 0.0, 0.0, 0.0)
        nodes = [PoseNode("r", None, RigidTransform(bad, Vec3(0, 0, 0)))]
        with self.assertRaises(NonRigidTransformError):
            resolve_global_poses(nodes)
        with self.assertRaises(NonRigidTransformError):
            resolve_global_poses(nodes, normalize=QUAT_NORMALIZE_REJECT)

    def test_explicit_reject_string_equal_to_constant(self):
        self.assertEqual(QUAT_NORMALIZE_REJECT, "reject")
        self.assertEqual(QUAT_NORMALIZE_NORMALIZE, "normalize")

    def test_normalize_mode_divides_by_norm(self):
        # q = 2 * (cos30 + k sin30)，范数 2，归一化后等于绕 z 60°
        c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
        q2 = Quat(2 * c, 0.0, 0.0, 2 * s)
        nodes = [
            PoseNode("r", None, RigidTransform(q2, Vec3(0, 0, 0))),
        ]
        pose = resolve_global_poses(nodes, normalize=QUAT_NORMALIZE_NORMALIZE)
        qn = pose.transforms["r"].rotation
        self.assertAlmostEqual(qn.w, c, delta=1e-12)
        self.assertAlmostEqual(qn.z, s, delta=1e-12)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in qn)), 1.0, delta=1e-12)
        # 作用于 (1,0,0)：绕 z 60° -> (0.5, √3/2, 0)
        (res,) = skin_named(
            [((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))],
            [PoseNode("r", None, IDENT)],
            nodes,
            [[("r", 1.0)]],
            normalize=QUAT_NORMALIZE_NORMALIZE,
        )
        assert_vec_close(self, res.position, (0.5, math.sqrt(3) / 2, 0.0))

    def test_normalize_within_tol_kept_unit(self):
        # 范数 1+5e-7（合法范围内）在 normalize 模式下也被拉回单位
        q = Quat(1.0 + 5e-7, 0.0, 0.0, 0.0)
        pose = resolve_global_poses(
            [PoseNode("r", None, RigidTransform(q, Vec3(0, 0, 0)))],
            normalize=QUAT_NORMALIZE_NORMALIZE,
        )
        qn = pose.transforms["r"].rotation
        self.assertAlmostEqual(qn.w, 1.0, delta=1e-12)

    def test_zero_quaternion_rejected_in_normalize_mode(self):
        q = Quat(0.0, 0.0, 0.0, 0.0)
        nodes = [PoseNode("r", None, RigidTransform(q, Vec3(0, 0, 0)))]
        with self.assertRaises(InvalidInputError) as cm:
            resolve_global_poses(nodes, normalize=QUAT_NORMALIZE_NORMALIZE)
        self.assertIn("零四元数", str(cm.exception))

    def test_unknown_normalize_mode_rejected(self):
        with self.assertRaises(InvalidInputError):
            resolve_global_poses([PoseNode("r", None, IDENT)], normalize="quiet")

    def test_normalize_mode_applies_to_bind_and_current(self):
        c, s = math.cos(math.pi / 8), math.sin(math.pi / 8)
        q2 = Quat(2 * c, 0.0, 0.0, 2 * s)  # 范数 2
        binds = [PoseNode("r", None, RigidTransform(q2, Vec3(0, 0, 0)))]
        curs = [PoseNode("r", None, RigidTransform(q2, Vec3(0, 0, 0)))]
        # 绑定与当前是同一（非单位书写的）姿态 -> 归一化后抵消，顶点不动
        verts = [((0.7, -0.2, 1.1), (0.0, 1.0, 0.0))]
        (res,) = skin_named(
            verts, binds, curs, [[("r", 1.0)]],
            normalize=QUAT_NORMALIZE_NORMALIZE,
        )
        assert_vec_close(self, res.position, verts[0][0])
        assert_vec_close(self, res.normal, verts[0][1])

    def test_nan_quaternion_rejected_even_in_normalize_mode(self):
        q = Quat(float("nan"), 0.0, 0.0, 1.0)
        nodes = [PoseNode("r", None, RigidTransform(q, Vec3(0, 0, 0)))]
        with self.assertRaises(InvalidInputError):
            resolve_global_poses(nodes, normalize=QUAT_NORMALIZE_NORMALIZE)


class TestHierarchyRejections(unittest.TestCase):
    """拒绝环、非法父节点、重复绑定；失败不留半更新状态。"""

    def test_two_node_cycle_rejected(self):
        nodes = [
            PoseNode("a", "b", IDENT),
            PoseNode("b", "a", IDENT),
        ]
        with self.assertRaises(HierarchyError) as cm:
            resolve_global_poses(nodes)
        self.assertIn("环", str(cm.exception))

    def test_three_node_cycle_rejected(self):
        nodes = [
            PoseNode("a", "c", IDENT),
            PoseNode("b", "a", IDENT),
            PoseNode("c", "b", IDENT),
        ]
        with self.assertRaises(HierarchyError) as cm:
            resolve_global_poses(nodes)
        self.assertIn("环", str(cm.exception))

    def test_self_parent_is_a_cycle(self):
        with self.assertRaises(HierarchyError) as cm:
            resolve_global_poses([PoseNode("a", "a", IDENT)])
        self.assertIn("环", str(cm.exception))

    def test_unknown_parent_rejected(self):
        nodes = [PoseNode("a", "ghost", IDENT)]
        with self.assertRaises(HierarchyError) as cm:
            resolve_global_poses(nodes)
        self.assertIn("非法父节点", str(cm.exception))
        self.assertIn("ghost", str(cm.exception))

    def test_duplicate_binding_rejected(self):
        nodes = [
            PoseNode("a", None, IDENT),
            PoseNode("a", None, IDENT),
        ]
        with self.assertRaises(HierarchyError) as cm:
            resolve_global_poses(nodes)
        self.assertIn("重复绑定", str(cm.exception))

    def test_none_name_rejected(self):
        with self.assertRaises(InvalidInputError):
            resolve_global_poses([PoseNode(None, None, IDENT)])

    def test_none_parent_name_in_pair_rejected(self):
        with self.assertRaises(InvalidInputError):
            resolve_global_poses([(None, None, IDENT)])

    def test_empty_and_too_many_nodes(self):
        with self.assertRaises(HierarchyError):
            resolve_global_poses([])
        many = [PoseNode(f"n{i}", None, IDENT) for i in range(33)]
        with self.assertRaises(HierarchyError) as cm:
            resolve_global_poses(many)
        self.assertIn("32", str(cm.exception))
        # 32 个节点合法
        pose = resolve_global_poses(many[:32])
        self.assertEqual(len(pose.order), 32)

    def test_bind_current_name_sets_must_match(self):
        binds = [PoseNode("a", None, IDENT), PoseNode("b", "a", IDENT)]
        curs = [PoseNode("a", None, IDENT), PoseNode("c", "a", IDENT)]
        with self.assertRaises(HierarchyError) as cm:
            skin_named(
                [((0, 0, 0), (0, 0, 1))], binds, curs,
                [[("a", 1.0)]],
            )
        self.assertIn("不一致", str(cm.exception))

    def test_unknown_bone_name_in_weights_rejected(self):
        nodes = [PoseNode("a", None, IDENT)]
        with self.assertRaises(WeightError) as cm:
            skin_named(
                [((0, 0, 0), (0, 0, 1))], nodes, nodes,
                [[("ghost", 1.0)]],
            )
        self.assertIn("未知骨骼名", str(cm.exception))

    def test_duplicate_bone_name_in_weight_row_rejected(self):
        nodes = [PoseNode("a", None, IDENT)]
        with self.assertRaises(WeightError):
            skin_named(
                [((0, 0, 0), (0, 0, 1))], nodes, nodes,
                [[("a", 0.5), ("a", 0.5)]],
            )

    def test_negative_weight_named_rejected(self):
        nodes = [
            PoseNode("a", None, IDENT),
            PoseNode("b", None, IDENT),
        ]
        with self.assertRaises(WeightError) as cm:
            skin_named(
                [((0, 0, 0), (0, 0, 1))], nodes, nodes,
                [[("a", 1.2), ("b", -0.2)]],
            )
        self.assertIn("weights[0][1]", str(cm.exception))

    def test_no_partial_state_on_failure(self):
        # 纯函数：调用失败前后输入快照不变，且不返回任何结果
        good = [PoseNode("a", None, IDENT)]
        bad_cycle = [PoseNode("a", "b", IDENT), PoseNode("b", "a", IDENT)]
        snap_good = repr(good)
        snap_bad = repr(bad_cycle)
        with self.assertRaises(HierarchyError):
            skin_named([((0, 0, 0), (0, 0, 1))], good, bad_cycle, [[("a", 1.0)]])
        self.assertEqual(repr(good), snap_good)
        self.assertEqual(repr(bad_cycle), snap_bad)

    def test_global_pose_failure_does_not_leak(self):
        # resolve 失败时不返回 GlobalPose（不是返回部分字典）
        nodes = [PoseNode("a", "x", IDENT), PoseNode("b", None, IDENT)]
        with self.assertRaises(HierarchyError):
            resolve_global_poses(nodes)


class TestLegacyEntryUnchanged(unittest.TestCase):
    """旧 skin 入口与下标语义继续可用（旧测试之外补一条冒烟）。"""

    def test_legacy_index_parent_still_works(self):
        bones = [
            Bone("root", None, IDENT, rigid((0, 0, 1), math.pi / 2)),
            Bone("child", 0, IDENT, rigid((0, 0, 1), 0.0, (1.0, 0.0, 0.0))),
        ]
        (res,) = skin(
            [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))], bones, [[(1, 1.0)]]
        )
        assert_vec_close(self, res.position, (0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
