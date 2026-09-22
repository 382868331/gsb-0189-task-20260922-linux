"""固定种子的随机化属性测试（确定性，离线，毫秒级）。

四类性质（期望值均来自 tests/_reference.py 的独立参考，不调用被测核心
生成期望值）：

1. 随机单骨（含 180 度附近）== 解析 current*inverse(bind)；
2. 随机同向骨骼 DLB == 独立 LBS 参考；
3. 随机多层级绑定姿态不动；
4. DLB 结果是刚体：同一权重下任意点对距离保持（等距性）。
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqskin import Bone, RigidTransform, skin  # noqa: E402
from tests import _reference as ref  # noqa: E402

TOL = 1e-9

I_M = ref.axis_angle_matrix((0.0, 0.0, 1.0), 0.0)


def _bone(b, c, p=None):
    return Bone(RigidTransform(b), RigidTransform(c), parent=p)


def _rand_axis(rng):
    v = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _rand_tf(rng, near_pi=False):
    axis = _rand_axis(rng)
    if near_pi:
        angle = rng.choice((-1.0, 1.0)) * (
            math.pi - rng.uniform(0.0, 1e-6)
        )
    else:
        angle = rng.uniform(-math.pi, math.pi)
    t = tuple(rng.uniform(-10.0, 10.0) for _ in range(3))
    return ref.axis_angle_matrix(axis, angle, t)


def _rand_points(rng, count=4, lo=-5.0, hi=5.0):
    return [
        tuple(rng.uniform(lo, hi) for _ in range(3)) for _ in range(count)
    ]


def _max_abs(a, b):
    return max(abs(a[k] - b[k]) for k in range(3))


class RandomizedPropertyTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20260922)

    def test_single_bone_matches_analytic(self):
        worst = 0.0
        for trial in range(60):
            bind = _rand_tf(self.rng)
            current = _rand_tf(
                self.rng, near_pi=(trial % 3 == 0)
            )
            pts = _rand_points(self.rng)
            ns = _rand_points(self.rng, lo=-1.0, hi=1.0)
            result = skin(
                pts, [_bone(bind, current)], [{0: 1.0}] * len(pts),
                normals=ns,
            )
            rel = ref.mat_mul(current, ref.inverse_rigid(bind))
            for i, p in enumerate(pts):
                worst = max(
                    worst, _max_abs(result.vertices[i], ref.mat_vec(rel, p))
                )
                worst = max(
                    worst,
                    _max_abs(result.normals[i], ref.mat_dir(rel, ns[i])),
                )
        self.assertLessEqual(worst, TOL)

    def test_coalitioned_bones_match_lbs_reference(self):
        worst = 0.0
        for _ in range(60):
            axis = _rand_axis(self.rng)
            angle = self.rng.uniform(-math.pi, math.pi)
            n_bones = self.rng.randint(2, 4)
            spec = [
                (I_M,
                 ref.axis_angle_matrix(
                     axis, angle,
                     tuple(self.rng.uniform(-8.0, 8.0) for _ in range(3)),
                 ),
                 None)
                for _ in range(n_bones)
            ]
            pts = _rand_points(self.rng, count=3)
            wl = [
                [(i, self.rng.uniform(0.05, 3.0)) for i in range(n_bones)]
                for _ in pts
            ]
            result = skin(
                pts,
                [_bone(b, c, p) for b, c, p in spec],
                [dict(w) for w in wl],
            )
            lbs_v, _ = ref.linear_blend_skin(pts, spec, wl)
            for i in range(len(pts)):
                worst = max(worst, _max_abs(result.vertices[i], lbs_v[i]))
        self.assertLessEqual(worst, TOL)

    def test_rest_pose_deep_chain(self):
        worst = 0.0
        for _ in range(40):
            depth = self.rng.randint(2, 6)
            ms = [_rand_tf(self.rng) for _ in range(depth)]
            spec = [
                (ms[i], ms[i], None if i == 0 else i - 1)
                for i in range(depth)
            ]
            pts = _rand_points(self.rng)
            wl = []
            for _ in pts:
                ws = [self.rng.uniform(0.0, 2.0) for _ in range(depth)]
                ws[0] += 0.05  # 保证总权 > 0
                wl.append([(i, w) for i, w in enumerate(ws)])
            result = skin(
                pts,
                [_bone(b, c, p) for b, c, p in spec],
                [dict(w) for w in wl],
            )
            for i, p in enumerate(pts):
                worst = max(worst, _max_abs(result.vertices[i], p))
        self.assertLessEqual(worst, 1e-12)

    def test_blended_transform_is_an_isometry(self):
        """DLB 混合结果必须是刚体：同权重下任意两点距离保持。"""
        worst = 0.0
        for _ in range(80):
            spec = [
                (I_M, _rand_tf(self.rng), None),
                (I_M, _rand_tf(self.rng), None),
            ]
            pts = _rand_points(self.rng, count=6)
            weights = {
                0: self.rng.uniform(0.05, 1.0),
                1: self.rng.uniform(0.05, 1.0),
            }
            result = skin(
                pts,
                [_bone(b, c, p) for b, c, p in spec],
                [dict(weights) for _ in pts],
            )
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d0 = math.sqrt(
                        sum((pts[i][k] - pts[j][k]) ** 2 for k in range(3))
                    )
                    d1 = math.sqrt(
                        sum(
                            (result.vertices[i][k] - result.vertices[j][k])
                            ** 2
                            for k in range(3)
                        )
                    )
                    worst = max(worst, abs(d0 - d1))
        self.assertLessEqual(worst, TOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
