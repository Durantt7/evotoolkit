# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANNIniter 配置类"""

from typing import TYPE_CHECKING, Optional

from evotoolkit.tools.llm import HttpsApi

from .utils import KnowledgeBase

if TYPE_CHECKING:
    from evotoolkit.task.cann_init import CANNInitTask, CANNIniterInterface


class CANNIniterConfig:
    """CANNIniter 配置（独立配置，不继承 BaseConfig）"""

    def __init__(
        self,
        task: "CANNInitTask",
        interface: "CANNIniterInterface",
        output_path: str,
        running_llm: HttpsApi,
        knowledge_base: Optional[KnowledgeBase] = None,
        verbose: bool = True,
        max_debug_iterations: int = 5,
        max_joint_turns: int = 3,
    ):
        self.task = task
        self.interface = interface
        self.output_path = output_path
        self.running_llm = running_llm
        self.knowledge_base = knowledge_base
        self.verbose = verbose
        self.max_debug_iterations = max_debug_iterations
        self.max_joint_turns = max_joint_turns
