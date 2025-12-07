# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANNIniter 工具类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class KnowledgeBase(ABC):
    """
    知识库抽象基类

    两种检索策略：
    - API: 严格模式（不猜，找不到返回 not_found）
    - Example: 宽松模式（可以返回相似的）
    """

    @abstractmethod
    def search_api(self, name: str) -> Dict[str, Any]:
        """
        严格检索 API 文档

        Returns:
            {
                "status": "found" | "not_found" | "ambiguous",
                "api_doc": "完整文档（如果找到）",
                "candidates": ["候选列表（如果模糊）"]
            }
        """
        pass

    @abstractmethod
    def search_operator(self, name: str, top_k: int = 3) -> Dict[str, Any]:
        """
        宽松检索算子示例

        Returns:
            {
                "primary": {"name": "...", "code": "..."},
                "related": [{"name": "...", "reason": "..."}],
                "confidence": "high" | "medium"
            }
        """
        pass

    def get_api_categories(self) -> Dict[str, List[str]]:
        """获取 API 分类列表（可选实现）"""
        return {}

    def get_paradigm_template(self, paradigm: str) -> Optional[str]:
        """获取编程范式模板（可选实现）"""
        return None
