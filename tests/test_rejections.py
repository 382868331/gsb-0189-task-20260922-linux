"""错误语义与输入边界测试。

覆盖题面明确的拒绝项：非刚体、循环骨骼、负权、零总权、NaN/Infinity、
bool 冒充整数、越界/形状错误、骨骼数上限、不可定位输入等。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqskin import (  # noqa: E402
    MAX_BONES,
    Bone,
    RigidTransform,
    RigidTransformError,
    SkeletonError,
    SkinningError,
    VertexDataError,
    WeightError,
    skin,
)

from tests import _reference as ref  # noqa: E402

I = ref.axis_angle_matrix((0.0, 0.0, 1.0), 0.0)


def _bone(bind=I, current=I, parent=None):
    return Bone(RigidTransform(bind), RigidTransform(current), parent=parent)


def _locatable(exception, needle):
    """错误消息必须包含可定位前缀片段。"""
    return needle in str(exception)


class RigidMatrixRejectionTests(unittest.TestCase):
    def test_wrong_shape_rows(self):
        with self.assertRaises(RigidTransformError) as cm:
            RigidTransform(tuple(tuple((1.0, 0.0, 0.0, 0.0))
                                 for _ in range(3)))
        self.assertTrue(_locatable(cm.exception, "RigidTransform.matrix"))

    def test_wrong_shape_cols(self):
        with self.assertRaises(RigidTransformError) as cm:
            RigidTransform(((1.0, 0.0, 0.0),) * 4)
        self.assertTrue(_locatable(cm.exception, "[0]"))

    def test_last_row_bad(self):
        m = [list(r) for r in I]
        m[3] = [0.0, 0.0, 0.0, 0.5]
        with self.assertRaises(RigidTransformError):
            RigidTransform(tuple(tuple(r) for r in m))

    def test_nonorthonormal_shear(self):
        m = [list(r) for r in I]
        m[0][1] = 0.5  # 剪切
        with self.assertRaises(RigidTransformError) as cm:
            RigidTransform(tuple(tuple(r) for r in m))
        self.assertIn("orthonormal", str(cm.exception))

    def test_scale_rejected(self):
        m = [list(r) for r in I]
        m[0][0] = 2.0
        with self.assertRaises(RigidTransformError):
            RigidTransform(tuple(tuple(r) for r in m))

    def test_reflection_rejected(self):
        m = [list(r) for r in I]
        m[0][0] = -1.0  # det = -1
        with self.assertRaises(RigidTransformError) as cm:
            RigidTransform(tuple(tuple(r) for r in m))
        self.assertIn("determinant", str(cm.exception))

    def test_nan_inf_in_matrix(self):
        for bad in (math.nan, math.inf, -math.inf):
            m = [list(r) for r in I]
            m[1][2] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(SkinningError) as cm:
                    RigidTransform(tuple(tuple(r) for r in m))
                self.assertTrue(
                    _locatable(cm.exception, "[1][2]"),
                    msg=str(cm.exception),
                )

    def test_near_orthogonal_but_outside_contract_tol(self):
        # 5e-9 偏差超过输入刚体容差 1e-9，必须拒绝（不用宽容差放行）。
        m = [list(r) for r in I]
        m[0][1] = 5e-9
        with self.assertRaises(RigidTransformError):
            RigidTransform(tuple(tuple(r) for r in m))

    def test_bool_element_is_typeerror(self):
        m = [list(r) for r in I]
        m[0][0] = True
        with self.assertRaises(TypeError):
            RigidTransform(tuple(tuple(r) for r in m))


class SkeletonRejectionTests(unittest.TestCase):
    def test_zero_bones(self):
        with self.assertRaises(SkeletonError) as cm:
            skin([(0.0, 0.0, 0.0)], [], [{}])
        self.assertTrue(_locatable(cm.exception, "bones"))

    def test_too_many_bones(self):
        bones = [_bone() for _ in range(MAX_BONES + 1)]
        with self.assertRaises(SkeletonError) as cm:
            skin([(0.0, 0.0, 0.0)], bones, [{0: 1.0}])
        self.assertIn(str(MAX_BONES), str(cm.exception))

    def test_max_bones_accepted(self):
        # 上限本身可用（小规模验证，不做压力测试）。
        bones = [_bone() for _ in range(MAX_BONES)]
        result = skin([(1.0, 2.0, 3.0)], bones, [{0: 1.0}])
        self.assertEqual(result.vertices[0], (1.0, 2.0, 3.0))

    def test_parent_out_of_range(self):
        bones = [_bone(parent=5)]
        with self.assertRaises(SkeletonError) as cm:
            skin([(0.0, 0.0, 0.0)], bones, [{0: 1.0}])
        self.assertTrue(_locatable(cm.exception, "bones[0].parent"))

    def test_self_parent(self):
        bones = [_bone(parent=0)]
        with self.assertRaises(SkeletonError) as cm:
            skin([(0.0, 0.0, 0.0)], bones, [{0: 1.0}])
        self.assertIn("own parent", str(cm.exception))

    def test_direct_cycle(self):
        bones = [_bone(parent=1), _bone(parent=0)]
        with self.assertRaises(SkeletonError) as cm:
            skin([(0.0, 0.0, 0.0)], bones, [{0: 1.0}])
        self.assertIn("cycle", str(cm.exception))

    def test_indirect_cycle(self):
        bones = [_bone(parent=2), _bone(parent=0), _bone(parent=1)]
        with self.assertRaises(SkeletonError) as cm:
            skin([(0.0, 0.0, 0.0)], bones, [{0: 1.0}])
        self.assertIn("cycle", str(cm.exception))
        # 路径可定位：至少标出环上的一个 bones[i]
        self.assertIn("bones[", str(cm.exception))

    def test_parent_bool_rejected(self):
        # bool 冒充父索引：构造期即 TypeError（bool 是 int 子类，必须显式拒绝）。
        with self.assertRaises(TypeError) as cm:
            Bone(RigidTransform(I), RigidTransform(I), parent=True)
        self.assertIn("bool", str(cm.exception))

    def test_bone_wrong_type(self):
        with self.assertRaises(TypeError) as cm:
            skin([(0.0, 0.0, 0.0)], [("not a bone",)], [{0: 1.0}])
        self.assertIn("bones[0]", str(cm.exception))


class BoneConstructionTests(unittest.TestCase):
    def test_bind_wrong_type(self):
        with self.assertRaises(TypeError) as cm:
            Bone(I, RigidTransform(I))
        self.assertIn("Bone.bind", str(cm.exception))

    def test_current_wrong_type(self):
        with self.assertRaises(TypeError) as cm:
            Bone(RigidTransform(I), I)
        self.assertIn("Bone.current", str(cm.exception))

    def test_parent_bool_at_construction(self):
        with self.assertRaises(TypeError):
            Bone(RigidTransform(I), RigidTransform(I), parent=True)

    def test_parent_float_at_construction(self):
        with self.assertRaises(TypeError):
            Bone(RigidTransform(I), RigidTransform(I), parent=1.5)

    def test_parent_negative_at_construction(self):
        with self.assertRaises(SkeletonError) as cm:
            Bone(RigidTransform(I), RigidTransform(I), parent=-1)
        self.assertIn("Bone.parent", str(cm.exception))


class VertexRejectionTests(unittest.TestCase):
    def setUp(self):
        self.bones = [_bone()]

    def test_wrong_component_count(self):
        with self.assertRaises(VertexDataError) as cm:
            skin([(1.0, 2.0)], self.bones, [{0: 1.0}])
        self.assertTrue(_locatable(cm.exception, "vertices[0]"))

    def test_nan_vertex(self):
        with self.assertRaises(SkinningError) as cm:
            skin([(1.0, math.nan, 0.0)], self.bones, [{0: 1.0}])
        self.assertTrue(_locatable(cm.exception, "vertices[0][1]"))

    def test_inf_vertex(self):
        with self.assertRaises(SkinningError):
            skin([(1.0, math.inf, 0.0)], self.bones, [{0: 1.0}])

    def test_bool_vertex_component(self):
        with self.assertRaises(TypeError) as cm:
            skin([(1.0, True, 0.0)], self.bones, [{0: 1.0}])
        self.assertIn("bool", str(cm.exception))

    def test_empty_vertices(self):
        with self.assertRaises(VertexDataError):
            skin([], self.bones, [])

    def test_normals_length_mismatch(self):
        with self.assertRaises(VertexDataError) as cm:
            skin(
                [(0.0, 0.0, 0.0)], self.bones, [{0: 1.0}],
                normals=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            )
        self.assertIn("normals", str(cm.exception))

    def test_normals_nan(self):
        with self.assertRaises(SkinningError):
            skin(
                [(0.0, 0.0, 0.0)], self.bones, [{0: 1.0}],
                normals=[(1.0, math.nan, 0.0)],
            )


class WeightRejectionTests(unittest.TestCase):
    def setUp(self):
        self.bones = [_bone(), _bone(parent=0)]

    def test_negative_weight(self):
        with self.assertRaises(WeightError) as cm:
            skin([(0.0, 0.0, 0.0)], self.bones, [{0: -0.1, 1: 1.0}])
        self.assertIn("negative weight", str(cm.exception))
        self.assertTrue(_locatable(cm.exception, "weights[0]"))

    def test_zero_total_weight(self):
        with self.assertRaises(WeightError) as cm:
            skin([(0.0, 0.0, 0.0)], self.bones, [{0: 0.0, 1: 0.0}])
        self.assertIn("total weight", str(cm.exception))

    def test_empty_weight_entry(self):
        with self.assertRaises(WeightError):
            skin([(0.0, 0.0, 0.0)], self.bones, [{}])

    def test_bone_index_out_of_range(self):
        with self.assertRaises(WeightError) as cm:
            skin([(0.0, 0.0, 0.0)], self.bones, [{7: 1.0}])
        self.assertTrue(_locatable(cm.exception, "weights[0]"))

    def test_weights_length_mismatch(self):
        with self.assertRaises(WeightError) as cm:
            skin(
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                self.bones,
                [{0: 1.0}],
            )
        self.assertIn("each vertex", str(cm.exception))

    def test_nan_weight(self):
        with self.assertRaises(SkinningError) as cm:
            skin([(0.0, 0.0, 0.0)], self.bones, [{0: math.nan}])
        self.assertTrue(_locatable(cm.exception, "weights[0]"))

    def test_inf_weight(self):
        with self.assertRaises(SkinningError):
            skin([(0.0, 0.0, 0.0)], self.bones, [{0: math.inf}])

    def test_bool_weight(self):
        with self.assertRaises(TypeError) as cm:
            skin([(0.0, 0.0, 0.0)], self.bones, [{0: True}])
        self.assertIn("bool", str(cm.exception))

    def test_bool_bone_index(self):
        with self.assertRaises(TypeError) as cm:
            skin([(0.0, 0.0, 0.0)], self.bones, [[(True, 1.0)]])
        self.assertIn("bool", str(cm.exception))

    def test_duplicate_bone_index(self):
        with self.assertRaises(WeightError) as cm:
            skin(
                [(0.0, 0.0, 0.0)], self.bones,
                [[(0, 0.5), (0, 0.5)]],
            )
        self.assertIn("duplicate", str(cm.exception))

    def test_malformed_pair(self):
        with self.assertRaises(WeightError):
            skin([(0.0, 0.0, 0.0)], self.bones, [[(0,)]])

    def test_negative_weight_at_specific_vertex(self):
        # 多顶点时错误必须定位到具体顶点。
        with self.assertRaises(WeightError) as cm:
            skin(
                [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                self.bones,
                [{0: 1.0}, {0: -1.0}],
            )
        self.assertIn("weights[1]", str(cm.exception))


class ErrorHierarchyTests(unittest.TestCase):
    def test_all_are_value_errors(self):
        for exc in (
            RigidTransformError,
            SkeletonError,
            VertexDataError,
            WeightError,
        ):
            self.assertTrue(issubclass(exc, SkinningError))
            self.assertTrue(issubclass(exc, ValueError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
