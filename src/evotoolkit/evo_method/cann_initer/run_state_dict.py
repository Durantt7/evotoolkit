# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANNIniter 运行状态"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


class CANNIniterRunStateDict:
    """CANNIniter 阶段间状态传递（使用 pickle 序列化）"""

    def __init__(
        self,
        task_info: Optional[dict] = None,
        # Phase 0 输出
        signature: Optional[Any] = None,
        compute_pattern: Optional[str] = None,
        strategies: Optional[Dict[str, str]] = None,
        # 并行分支输出
        pybind_src: Optional[str] = None,
        kernel_src: Optional[str] = None,
        tiling_src: Optional[str] = None,
        operator_src: Optional[str] = None,
        # 联合规划
        joint_plan: Optional[dict] = None,
        joint_conversation: Optional[List[dict]] = None,
        # 知识检索结果
        knowledge: Optional[dict] = None,
        # Debug 状态
        debug_history: Optional[List[dict]] = None,
        current_iteration: int = 0,
        is_done: bool = False,
        success: bool = False,
    ):
        # 基础信息
        self.task_info = task_info or {}

        # Phase 0
        self.signature = signature
        self.compute_pattern = compute_pattern
        self.strategies = strategies or {}

        # 并行分支输出
        self.pybind_src = pybind_src
        self.kernel_src = kernel_src
        self.tiling_src = tiling_src
        self.operator_src = operator_src

        # 联合规划
        self.joint_plan = joint_plan or {}
        self.joint_conversation = joint_conversation or []

        # 知识检索
        self.knowledge = knowledge or {}

        # Debug
        self.debug_history = debug_history or []
        self.current_iteration = current_iteration
        self.is_done = is_done
        self.success = success

    def to_pickle(self, file_path: str) -> None:
        """保存状态到 pickle 文件"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_path: str) -> "CANNIniterRunStateDict":
        """从 pickle 文件加载状态"""
        with open(file_path, "rb") as f:
            return pickle.load(f)
