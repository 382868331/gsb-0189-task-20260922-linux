"""dqskin 的错误类型层级。

所有由本库主动拒绝的输入都继承 :class:`SkinningError`（它是 ``ValueError``
的子类），错误消息一律带可定位前缀（``bones[2].bind``、``vertices[5]``、
``weights[3]`` 等）。布尔冒充整数/数字时抛标准 ``TypeError``。
"""

from __future__ import annotations


class SkinningError(ValueError):
    """所有蒙皮输入契约错误的基类。"""


class RigidTransformError(SkinningError):
    """4x4 变换不是刚体变换（含 NaN/Inf、非正交、反射、末行不为 [0,0,0,1]）。"""


class SkeletonError(SkinningError):
    """骨骼树非法：超过 32 根、父索引越界、有环、多根/不连通。"""


class WeightError(SkinningError):
    """权重非法：负权、零总权、索引越界、形状错误等。"""


class VertexDataError(SkinningError):
    """顶点/法向数据非法：长度不符、非有限数、不是三元向量等。"""
