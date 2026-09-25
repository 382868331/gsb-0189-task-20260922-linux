"""
dqskin — 双四元数骨骼蒙皮（Dual-Quaternion Skinning）、绑定修正与权重敏感度查询。

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
    WeightSensitivity,
    Vec3,
    skin,
    skin_weight_sensitivity,
    QNORM_TOL,
    MAX_BONES,
    ALIGN_DOT_TOL,
    MIN_BLEND_NORM,
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
    "WeightSensitivity",
    "Vec3",
    "skin",
    "skin_weight_sensitivity",
    "QNORM_TOL",
    "MAX_BONES",
    "ALIGN_DOT_TOL",
    "MIN_BLEND_NORM",
]
