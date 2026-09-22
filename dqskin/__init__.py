"""
dqskin — 双四元数骨骼蒙皮（Dual-Quaternion Skinning）与绑定修正。

仅依赖 Python 3.14 标准库。公开接口见模块文档与 README。
"""

from .core import (
    Bone,
    Quat,
    RigidTransform,
    SkinError,
    InvalidInputError,
    HierarchyError,
    NonRigidTransformError,
    WeightError,
    SkinResult,
    Vec3,
    skin,
    QNORM_TOL,
    MAX_BONES,
)

__all__ = [
    "Bone",
    "Quat",
    "RigidTransform",
    "SkinError",
    "InvalidInputError",
    "HierarchyError",
    "NonRigidTransformError",
    "WeightError",
    "SkinResult",
    "Vec3",
    "skin",
    "QNORM_TOL",
    "MAX_BONES",
]
