"""dqskin —— 双四元数骨骼蒙皮与绑定修正（dual-quaternion skinning）。

标准库实现，无第三方依赖。唯一核心入口：

    from dqskin import skin, Bone, RigidTransform

详见 README。
"""

from __future__ import annotations

from .core import (
    MAX_BONES,
    Bone,
    RigidTransform,
    SkinResult,
    skin,
)
from .errors import (
    RigidTransformError,
    SkeletonError,
    SkinningError,
    VertexDataError,
    WeightError,
)

__all__ = [
    "MAX_BONES",
    "Bone",
    "RigidTransform",
    "SkinResult",
    "skin",
    "SkinningError",
    "RigidTransformError",
    "SkeletonError",
    "VertexDataError",
    "WeightError",
]

__version__ = "1.0.0"
