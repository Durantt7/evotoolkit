# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Pybind 专员 Prompt 默认实现"""

from typing import Any


class PybindPromptMixin:
    """Pybind 专员 Prompt"""

    def get_pybind_prompt(self, signature: Any) -> str:
        """生成 Pybind 代码 Prompt"""
        return f"""
你是 PyTorch C++ 扩展专家。请生成 NPU 算子的 pybind 代码。

## 算子签名
{signature}

## 要求
1. 使用 at::Tensor 作为输入输出类型
2. 正确计算输出 tensor 的 shape
3. 调用 NPU kernel 执行计算
4. 处理好内存分配

## 代码模板参考
```cpp
#include <torch/extension.h>
#include "aclnn_xxx.h"

at::Tensor npu_xxx(const at::Tensor& input, ...) {{
    // 1. 计算输出 shape
    auto output_shape = ...;

    // 2. 创建输出 tensor
    auto output = at::empty(output_shape, input.options());

    // 3. 调用 NPU kernel
    EXEC_NPU_CMD(aclnnXxx, input, ..., output);

    return output;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("npu_xxx", &npu_xxx, "NPU xxx operation");
}}
```

请返回完整的 C++ 代码。
"""
