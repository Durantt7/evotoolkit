# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Phase 0: 计算模式识别 Prompt 默认实现"""

from typing import Any


class Phase0PromptMixin:
    """Phase 0 计算模式识别 Prompt"""

    def get_pattern_analysis_prompt(self, python_ref: str, signature: Any) -> str:
        """生成计算模式识别 Prompt"""
        return f"""
分析以下 Python 算子代码的计算模式和实现策略。

## Python Reference 代码
```python
{python_ref}
```

## 输入输出签名
{signature}

## 请分析并返回 JSON 格式：

```json
{{
  "compute_pattern": "element-wise | reduction | matmul | broadcast | other",
  "strategies": {{
    "kernel": "generate",
    "tiling": "default | generate",
    "pybind": "default | generate"
  }},
  "reasoning": "简要说明判断理由"
}}
```

**策略规则**：
- element-wise: tiling=default, pybind=default（输出 shape = 输入 shape，不需要推断）
- 其他所有: tiling=generate, pybind=generate（需要推断输出 shape）

**计算模式判断标准**：
- element-wise: 逐元素操作（+, -, *, /, exp, log, relu...），输出形状 = 输入形状
- reduction: 沿某维度聚合（sum, mean, max, min...），输出形状 < 输入形状
- matmul: 矩阵乘法，输出形状由输入维度决定
- broadcast: 涉及广播的操作
- other: 其他复杂模式
"""
