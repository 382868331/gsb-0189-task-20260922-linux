"""姿态节点入口（局部变换 + 父节点 -> 全局姿态 -> 蒙皮）的单元测试。

期望值来源：
- 解析手算（三层骨架、父节点旋转后的子节点位置、单节点、拓扑换拓扑）；
- tests/refimpl.py 中与 dqskin 完全独立的矩阵实现（交叉验证全局姿态）。
任何期望值都不通过调用 dqskin 自身得到。
"""

from __future__ import annotations

import math
import unittest

from dqskin import (
    HierarchyError,
    InvalidInputError,
    NonRigidTransformError,
    PoseNode,
    Quat,
    RigidTransform,
    Vec3,
    WeightError,
    compute_global_pose,
    skin_with_pose,
)

import refimpl


def quat_axis_angle(axis, angle):
    """轴角 -> 单位四元数 (w, x, y, z)。标准定义，测试侧独立书写。"""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    h = angle / 2.0
    s = math.sin(h) / n
    return Quat(math.cos(h), ax * s, ay * s, az * s)


def quat_to_matrix(q):
    """单位四元数 -> 3x3 旋转矩阵。标准展开式，测试侧独立书写。"""
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


IDENT_Q = Quat(1.0, 0.0, 0.0, 0.0)
IDENT = RigidTransform(IDENT_Q, Vec3(0.0, 0.0, 0.0))


def T(x, y, z):
    """纯平移局部变换。"""
    return RigidTransform(IDENT_Q, Vec3(x, y, z))


def rigid(axis, angle, t=(0.0, 0.0, 0.0)):
    return RigidTransform(quat_axis_angle(axis, angle), Vec3(*t))


def assert_vec_close(tc, actual, expected, tol=1e-9):
    tc.assertAlmostEqual(actual[0], expected[0], delta=tol)
    tc.assertAlmostEqual(actual[1], expected[1], delta=tol)
    tc.assertAlmostEqual(actual[2], expected[2], delta=tol)


def assert_mat_close(tc, actual, expected, tol=1e-9):
    for i in range(3):
        for j in range(3):
            tc.assertAlmostEqual(actual[i][j], expected[i][j], delta=tol)


def axis_angle_of(rt):
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


# 三层骨架（root -> mid -> leaf），局部均为沿 x 平移 1；
# 当前姿态 root 绕 z 转 90°。手算全局结果：
#   root: R90z, t=(0,0,0)
#   mid : R90z, t=(0,1,0)
#   leaf: R90z, t=(0,2,0)
def make_three_layer(angle=math.pi / 2):
    bind = [
        PoseNode("root", None, T(0.0, 0.0, 0.0)),
        PoseNode("mid", "root", T(1.0, 0.0, 0.0)),
        PoseNode("leaf", "mid", T(1.0, 0.0, 0.0)),
    ]
    current = [
        PoseNode("root", None, rigid((0, 0, 1), angle)),
        PoseNode("mid", "root", T(1.0, 0.0, 0.0)),
        PoseNode("leaf", "mid", T(1.0, 0.0, 0.0)),
    ]
    return bind, current


class TestGlobalPoseAnalytic(unittest.TestCase):
    """三层骨架：拓扑排序与全局姿态的解析手算校验。"""

    def test_three_layer_parent_rotation(self):
        # 输入顺序故意打乱（子先于父），拓扑排序必须仍得父先于子
        nodes = [
            PoseNode("leaf", "mid", T(1.0, 0.0, 0.0)),
            PoseNode("root", None, rigid((0, 0, 1), math.pi / 2)),
            PoseNode("mid", "root", T(1.0, 0.0, 0.0)),
        ]
        g = compute_global_pose(nodes)
        self.assertEqual(list(g), ["root", "mid", "leaf"])

        s2 = math.sqrt(0.5)
        for name in ("root", "mid", "leaf"):
            q = g[name].rotation
            # 手算：三层旋转均为 R90z = (cos45°, 0, 0, sin45°)
            self.assertAlmostEqual(q.w, s2, delta=1e-12)
            self.assertAlmostEqual(q.x, 0.0, delta=1e-12)
            self.assertAlmostEqual(q.y, 0.0, delta=1e-12)
            self.assertAlmostEqual(q.z, s2, delta=1e-12)
        # 手算平移：root (0,0,0)；mid = R90z·(1,0,0) = (0,1,0)；
        # leaf = R90z·(1,0,0) + (0,1,0) = (0,2,0)
        assert_vec_close(self, g["root"].translation, (0.0, 0.0, 0.0))
        assert_vec_close(self, g["mid"].translation, (0.0, 1.0, 0.0))
        assert_vec_close(self, g["leaf"].translation, (0.0, 2.0, 0.0))

    def test_child_position_after_parent_rotation(self):
        # 端到端：父节点旋转后子节点位置（skin_with_pose -> 旧入口）
        bind, current = make_three_layer()
        verts = [
            ((2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # leaf 末端（绑定世界位置）
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # mid 关节
            ((0.5, 0.0, 0.0), (0.0, 0.0, 1.0)),  # root 中段
        ]
        weights = [[(2, 1.0)], [(1, 1.0)], [(0, 1.0)]]
        res = skin_with_pose(verts, bind, current, weights)
        # 手算：M_leaf = M_mid = M_root = 绕原点 R90z（平移分量相消为 0）
        # (2,0,0)->(0,2,0)；(1,0,0)->(0,1,0)；(0.5,0,0)->(0,0.5,0)；法向不变
        assert_vec_close(self, res[0].position, (0.0, 2.0, 0.0))
        assert_vec_close(self, res[1].position, (0.0, 1.0, 0.0))
        assert_vec_close(self, res[2].position, (0.0, 0.5, 0.0))
        for r in res:
            assert_vec_close(self, r.normal, (0.0, 0.0, 1.0))

    def test_weight_index_follows_bind_input_order(self):
        # weights 下标对应 bind_pose 的输入顺序（此处输入顺序非拓扑序）
        bind, current = make_three_layer()
        shuffled_bind = [bind[2], bind[0], bind[1]]  # leaf, root, mid
        # 下标 0 = leaf：顶点 (2,0,0) 应到 (0,2,0)
        (res,) = skin_with_pose(
            [((2.0, 0.0, 0.0), (0.0, 0.0, 1.0))], shuffled_bind, current, [[(0, 1.0)]]
        )
        assert_vec_close(self, res.position, (0.0, 2.0, 0.0))
        # 下标 2 = mid：顶点 (1,0,0) 应到 (0,1,0)
        (res,) = skin_with_pose(
            [((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))], shuffled_bind, current, [[(2, 1.0)]]
        )
        assert_vec_close(self, res.position, (0.0, 1.0, 0.0))


class TestGlobalPoseCrossCheck(unittest.TestCase):
    """与独立矩阵参考（refimpl）交叉验证全局姿态。"""

    def test_three_layer_nontrivial_chain(self):
        locals_ = [
            rigid((0, 0, 1), 0.2, (0.0, 1.0, 0.0)),
            rigid((1, 0, 0), -0.3, (1.0, 0.0, 0.0)),
            rigid((0, 1, 0), 0.4, (0.0, 0.0, 1.0)),
        ]
        nodes = [
            PoseNode("b0", None, locals_[0]),
            PoseNode("b1", "b0", locals_[1]),
            PoseNode("b2", "b1", locals_[2]),
        ]
        g = compute_global_pose(nodes)

        # 独立参考：矩阵刚体复合 W = W_parent · L
        Ts = [refimpl.rigid_from_axis_angle(*axis_angle_of(loc)) for loc in locals_]
        W = [Ts[0]]
        W.append(refimpl.rigid_compose(W[0], Ts[1]))
        W.append(refimpl.rigid_compose(W[1], Ts[2]))

        for i, name in enumerate(("b0", "b1", "b2")):
            R_expected, t_expected = W[i]
            assert_mat_close(self, quat_to_matrix(g[name].rotation), R_expected)
            assert_vec_close(self, g[name].translation, t_expected)

    def test_normalization_rule_on_deep_chain(self):
        # 固定规范化规则：32 层链每步单位化，全局旋转保持严格单位且正确
        axis = (1.0, 2.0, 3.0)
        angle = 0.37
        n = 32
        nodes = [
            PoseNode(i, None if i == 0 else i - 1, rigid(axis, angle))
            for i in range(n)
        ]
        g = compute_global_pose(nodes)
        q = g[n - 1].rotation
        norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
        self.assertAlmostEqual(norm, 1.0, delta=1e-12)

        # 独立参考：矩阵连乘（等价于绕同轴转 n*angle）
        R_step = refimpl.axis_angle_matrix(axis, angle)
        R = R_step
        for _ in range(n - 1):
            R = refimpl.mat_mul(R, R_step)
        assert_mat_close(self, quat_to_matrix(q), R, tol=1e-9)
        # 解析关系：R = axis_angle(axis, 32*0.37)
        assert_mat_close(
            self, quat_to_matrix(q), refimpl.axis_angle_matrix(axis, n * angle), tol=1e-9
        )


class TestTopologyProperties(unittest.TestCase):
    """拓扑性质：森林、父先于子（小规模枚举）。"""

    def test_forest_parent_before_child(self):
        nodes = [
            PoseNode("c1", "r1", IDENT),
            PoseNode("r2", None, IDENT),
            PoseNode("g1", "c1", IDENT),
            PoseNode("r1", None, IDENT),
            PoseNode("c2", "r2", IDENT),
        ]
        g = compute_global_pose(nodes)
        order = list(g)
        self.assertEqual(sorted(order), ["c1", "c2", "g1", "r1", "r2"])
        pos = {name: i for i, name in enumerate(order)}
        for nd in nodes:  # 枚举每条父子边：父必先于子
            if nd.parent is not None:
                self.assertLess(pos[nd.parent], pos[nd.name])

    def test_single_node(self):
        rt = rigid((0, 0, 1), 0.5, (1.0, 2.0, 3.0))
        g = compute_global_pose([PoseNode("only", None, rt)])
        self.assertEqual(list(g), ["only"])
        self.assertEqual(g["only"], rt)

    def test_single_node_skin_hand_computed(self):
        # 单节点端到端：bind 恒等，current 绕 z 转 90° 再平移 (1,2,3)
        cur = rigid((0, 0, 1), math.pi / 2, (1.0, 2.0, 3.0))
        (res,) = skin_with_pose(
            [((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
            [PoseNode("a", None, IDENT)],
            [PoseNode("a", None, cur)],
            [[(0, 1.0)]],
        )
        # 手算：R90z·(1,0,0) + (1,2,3) = (1,3,3)；法向 R90z·(0,1,0) = (-1,0,0)
        assert_vec_close(self, res.position, (1.0, 3.0, 3.0))
        assert_vec_close(self, res.normal, (-1.0, 0.0, 0.0))

    def test_different_topology_bind_and_current(self):
        # 绑定/当前父结构允许不同：child 在当前姿态中脱离 root 成为根
        bind = [
            PoseNode("root", None, IDENT),
            PoseNode("child", "root", T(1.0, 0.0, 0.0)),
        ]
        current = [
            PoseNode("root", None, IDENT),
            PoseNode("child", None, rigid((0, 0, 1), math.pi / 2, (1.0, 0.0, 0.0))),
        ]
        (res,) = skin_with_pose(
            [((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))], bind, current, [[(1, 1.0)]]
        )
        # 手算：W_bind_child = T(1,0,0)；W_cur_child = (R90z, (1,0,0))；
        # M = W_cur·W_bind⁻¹：t = R90z·(-1,0,0)+(1,0,0) = (1,-1,0)；
        # p=(1,0,0) -> R90z·(1,0,0)+(1,-1,0) = (1,0,0)（顶点不动）
        assert_vec_close(self, res.position, (1.0, 0.0, 0.0))
        assert_vec_close(self, res.normal, (0.0, 0.0, 1.0))

    def test_bone_limit_inherited_from_skin(self):
        # 32 节点链合法；33 节点由旧入口的 MAX_BONES 拒绝
        ident_pose = [PoseNode(i, None if i == 0 else i - 1, IDENT) for i in range(32)]
        ok = skin_with_pose(
            [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))], ident_pose, ident_pose, [[(31, 1.0)]]
        )
        self.assertEqual(len(ok), 1)
        too_many = [PoseNode(i, None if i == 0 else i - 1, IDENT) for i in range(33)]
        with self.assertRaises(HierarchyError) as cm:
            skin_with_pose(
                [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))], too_many, too_many, [[(32, 1.0)]]
            )
        self.assertIn("32", str(cm.exception))


class TestPoseRejections(unittest.TestCase):
    """非法姿态输入必须抛出可定位错误，且不返回部分结果。"""

    def test_duplicate_binding_rejected(self):
        nodes = [PoseNode("a", None, IDENT), PoseNode("a", None, IDENT)]
        with self.assertRaises(HierarchyError) as cm:
            compute_global_pose(nodes)
        self.assertIn("重复绑定", str(cm.exception))
        self.assertIn("pose", str(cm.exception))

    def test_unknown_parent_rejected(self):
        nodes = [PoseNode("a", "ghost", IDENT)]
        with self.assertRaises(HierarchyError) as cm:
            compute_global_pose(nodes)
        self.assertIn("pose[0].parent", str(cm.exception))

    def test_self_loop_rejected(self):
        with self.assertRaises(HierarchyError) as cm:
            compute_global_pose([PoseNode("a", "a", IDENT)])
        self.assertIn("自环", str(cm.exception))

    def test_cycle_rejected(self):
        two = [PoseNode("a", "b", IDENT), PoseNode("b", "a", IDENT)]
        with self.assertRaises(HierarchyError) as cm:
            compute_global_pose(two)
        self.assertIn("环", str(cm.exception))
        three = [
            PoseNode("a", "c", IDENT),
            PoseNode("b", "a", IDENT),
            PoseNode("c", "b", IDENT),
        ]
        with self.assertRaises(HierarchyError):
            compute_global_pose(three)

    def test_non_rigid_local_rejected(self):
        bad = RigidTransform(Quat(1.0, 0.5, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
        with self.assertRaises(NonRigidTransformError) as cm:
            compute_global_pose([PoseNode("a", None, bad)])
        self.assertIn("pose[0].local.rotation", str(cm.exception))

    def test_nan_inf_rejected(self):
        bad = RigidTransform(IDENT_Q, Vec3(float("nan"), 0.0, 0.0))
        with self.assertRaises(InvalidInputError) as cm:
            compute_global_pose([PoseNode("a", None, bad)])
        self.assertIn("pose[0].local.translation", str(cm.exception))

    def test_unhashable_name_rejected(self):
        with self.assertRaises(InvalidInputError):
            compute_global_pose([PoseNode(["a"], None, IDENT)])
        with self.assertRaises(InvalidInputError):
            compute_global_pose([PoseNode("a", ["b"], IDENT)])

    def test_empty_pose_rejected(self):
        with self.assertRaises(HierarchyError):
            compute_global_pose([])

    def test_name_set_mismatch_rejected(self):
        bind = [PoseNode("a", None, IDENT), PoseNode("b", "a", IDENT)]
        current = [PoseNode("a", None, IDENT), PoseNode("c", "a", IDENT)]
        with self.assertRaises(HierarchyError) as cm:
            skin_with_pose(
                [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))], bind, current, [[(0, 1.0)]]
            )
        self.assertIn("不一致", str(cm.exception))

    def test_error_path_prefixes(self):
        # skin_with_pose 的错误路径带 bind_pose / current_pose 前缀
        bad = RigidTransform(Quat(2.0, 0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
        with self.assertRaises(NonRigidTransformError) as cm:
            skin_with_pose(
                [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
                [PoseNode("a", None, bad)],
                [PoseNode("a", None, IDENT)],
                [[(0, 1.0)]],
            )
        self.assertIn("bind_pose[0].local.rotation", str(cm.exception))
        with self.assertRaises(HierarchyError) as cm:
            skin_with_pose(
                [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
                [PoseNode("a", None, IDENT)],
                [PoseNode("a", "a", IDENT)],
                [[(0, 1.0)]],
            )
        self.assertIn("current_pose[0].parent", str(cm.exception))

    def test_no_partial_results_on_failure(self):
        # 姿态合法但权重非法 -> 整体抛错，不返回任何结果
        bind, current = make_three_layer()
        with self.assertRaises(WeightError):
            skin_with_pose(
                [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))], bind, current, [[(0, -1.0)]]
            )


class TestPoseInputFormsAndPurity(unittest.TestCase):
    def test_tuple_and_dict_forms(self):
        rt = rigid((0, 0, 1), 0.25, (1.0, 0.0, 0.0))
        a = compute_global_pose([PoseNode("x", None, rt)])
        b = compute_global_pose([("x", None, rt)])
        c = compute_global_pose([{"name": "x", "parent": None, "local": rt}])
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_inputs_not_mutated(self):
        bind, current = make_three_layer()
        verts = [((2.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
        weights = [[(2, 1.0)]]
        snapshot = (repr(bind), repr(current), repr(verts), repr(weights))
        skin_with_pose(verts, bind, current, weights)
        compute_global_pose(bind)
        self.assertEqual(snapshot, (repr(bind), repr(current), repr(verts), repr(weights)))

    def test_returned_globals_independent_per_call(self):
        # 纯函数：两次调用互不影响，返回值可独立使用
        nodes = [PoseNode("a", None, T(1.0, 0.0, 0.0))]
        g1 = compute_global_pose(nodes)
        g2 = compute_global_pose(nodes)
        self.assertEqual(g1, g2)
        self.assertIsNot(g1, g2)


if __name__ == "__main__":
    unittest.main()
