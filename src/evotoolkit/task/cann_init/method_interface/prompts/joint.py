# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Kernel + Tiling 联合分支 Prompt 默认实现"""

from typing import List


class JointPromptMixin:
    """Kernel + Tiling 联合分支 Prompt"""

    # ==================== 多轮对话 ====================

    def get_tiling_propose_prompt(self, context: dict, conversation: List[dict]) -> str:
        """生成 Tiling 专员提出策略 Prompt"""
        history = "\n".join([f"[{msg['role']}]: {msg['content']}" for msg in conversation])

        return f"""
你是昇腾 Ascend C Tiling 专员。基于以下信息提出 tiling 策略。

## 算子信息
- 计算模式: {context.get('compute_pattern')}
- 签名: {context.get('signature')}

## Python Reference
```python
{context.get('python_ref')}
```

## 对话历史
{history if history else "(这是第一轮，请提出初始方案)"}

## 你的任务
1. 分析计算模式，确定编程范式（vector / cube）
2. 提出 tiling 策略：
   - block_dim: 并行核心数
   - tile_size: 每次处理的数据量
   - 是否使用默认 tiling 模板
3. 说明数据切分方式

## 输出格式
```json
{{
  "paradigm": "vector | cube",
  "use_default_tiling": true | false,
  "tiling_params": {{
    "block_dim": 8,
    "tile_size": "auto | 具体值"
  }},
  "data_partition": "说明数据如何切分",
  "reasoning": "策略理由"
}}
```
"""

    def get_kernel_review_prompt(self, context: dict, conversation: List[dict]) -> str:
        """生成 Kernel 专员评审 Prompt"""
        history = "\n".join([f"[{msg['role']}]: {msg['content']}" for msg in conversation])

        return f"""
你是昇腾 Ascend C Kernel 专员。评审 Tiling 专员的方案。

## 算子信息
- 计算模式: {context.get('compute_pattern')}

## Python Reference
```python
{context.get('python_ref')}
```

## 对话历史
{history}

## 你的任务
1. 评估 tiling 方案是否合理
2. 设计 kernel 数据流：CopyIn → Compute → CopyOut
3. 确定需要的 Ascend C API
4. 确定流水线策略（single_buffer / double_buffer）

## 输出格式
如果接受方案，回复中**必须包含 "accepted"**：
```json
{{
  "accepted": true,
  "kernel_design": {{
    "data_flow": "CopyIn → Compute → CopyOut",
    "pipeline": "double_buffer",
    "key_apis": ["Add", "DataCopy", "AllocTensor"]
  }},
  "retrieval_requests": [
    {{"type": "api", "name": "Add"}},
    {{"type": "example", "name": "add_custom"}}
  ]
}}
```

如果需要修改，说明原因并提出建议。
"""

    # ==================== 代码实现 ====================

    def get_kernel_impl_prompt(self, plan: dict, knowledge: dict, python_ref: str) -> str:
        """生成 Kernel 实现 Prompt"""
        return f"""
你是昇腾 Ascend C Kernel 开发专家。请生成完整的 kernel 代码。

## Python Reference
```python
{python_ref}
```

## 联合规划
{plan}

## 可用知识（API 文档 / 示例代码）
{knowledge}

## 代码要求
1. 使用 Ascend C API
2. 实现标准结构：
   - class KernelXxx: Init, Process, CopyIn, Compute, CopyOut
   - extern "C" __global__ 入口函数
3. 正确处理 tiling 参数
4. 使用流水线优化

## 代码模板
```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelXxx {{
public:
    __aicore__ inline KernelXxx() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {{
        // 初始化 GlobalTensor, 计算 tiling 参数
    }}
    __aicore__ inline void Process() {{
        // 主循环：CopyIn → Compute → CopyOut
    }}
private:
    __aicore__ inline void CopyIn(int32_t progress) {{ /* DataCopy GM -> UB */ }}
    __aicore__ inline void Compute(int32_t progress) {{ /* 计算 */ }}
    __aicore__ inline void CopyOut(int32_t progress) {{ /* DataCopy UB -> GM */ }}

    // 成员变量
    GlobalTensor<half> xGm, yGm, zGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, 2> outQueue;
}};

extern "C" __global__ __aicore__ void xxx_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
    KernelXxx op;
    op.Init(x, y, z, tilingData.totalLength);
    op.Process();
}}
```

请返回完整的 kernel C++ 代码。
"""

    def get_tiling_impl_prompt(self, plan: dict, knowledge: dict) -> str:
        """生成 Tiling 实现 Prompt"""
        return f"""
你是昇腾 Ascend C Host 端开发专家。请生成 tiling 相关代码。

## 联合规划
{plan}

## 可用知识
{knowledge}

## 需要生成两个文件

### 1. tiling.h (Tiling 数据结构)
```cpp
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF(XxxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Xxx, XxxTilingData)
}}
```

### 2. op_host.cpp (Tiling 计算 + InferShape)
```cpp
#include "xxx_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {{
static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
    // 计算 tiling 参数
    XxxTilingData tiling;
    tiling.set_totalLength(...);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), ...);
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ops {{
class Xxx : public OpDef {{
public:
    Xxx(const char* name) : OpDef(name) {{
        // 定义输入输出
        this->Input("x").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16}});
        this->Output("z").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16}});
        // 注册 tiling
        this->SetInferShape(ge::InferShape).SetTiling(optiling::TilingFunc);
    }}
}};
OP_ADD(Xxx);
}}
```

## 返回 JSON 格式
```json
{{
  "host_tiling_src": "完整的 tiling.h 代码",
  "host_operator_src": "完整的 op_host.cpp 代码"
}}
```
"""
