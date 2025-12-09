# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Code Implementation Prompts for Joint Branch

This module contains prompts for the three-stage code generation:
1. Tiling Header (tiling.h) - Tiling data structure definition (NO knowledge needed)
2. Tiling Host (op_host.cpp) - Tiling calculation + InferShape
3. Kernel (op_kernel.cpp) - Kernel implementation

Each stage is generated separately to ensure quality and manageability.
"""

from typing import List, Optional

from evotoolkit.task.cann_init.method_interface.prompts.phase0 import _format_signature


def _format_tiling_fields(tiling_fields: List[dict]) -> str:
    """Format tiling fields for prompt display.

    Args:
        tiling_fields: List of {"name": str, "type": str, "purpose": str}

    Returns:
        Formatted string like:
        - totalLength: uint32_t // total number of elements
        - tileNum: uint32_t // number of tiles
    """
    if not tiling_fields:
        return "(no custom fields specified)"

    lines = []
    for field in tiling_fields:
        if isinstance(field, dict):
            name = field.get("name", "unknown")
            ftype = field.get("type", "uint32_t")
            purpose = field.get("purpose", "")
            lines.append(f"- {name}: {ftype} // {purpose}")
        elif isinstance(field, str):
            lines.append(f"- {field}")
    return "\n".join(lines) if lines else "(none)"


def _format_joint_plan(plan: dict) -> dict:
    """Extract and format joint plan components."""
    return {
        "tiling_strategy": plan.get("tiling_strategy", "default"),
        "tiling_fields": _format_tiling_fields(plan.get("tiling_fields", [])),
        "tiling_execution": plan.get("tiling_execution", "(not specified)"),
        "kernel_pseudocode": plan.get("kernel_pseudocode", "(not specified)"),
        "kernel_design": plan.get("kernel_design", "(not specified)"),
        "tiling_proposal": plan.get("tiling_proposal", "(not specified)"),
    }


class ImplPromptsMixin:
    """Prompts for three-stage code implementation.

    Stage 1: get_tiling_header_prompt() -> tiling.h (NO knowledge needed)
    Stage 2: get_tiling_host_prompt() -> op_host.cpp
    Stage 3: get_kernel_impl_prompt() -> op_kernel.cpp

    For default tiling strategy, only Stage 3 is needed.
    """

    # =========================================================================
    # Stage 1: Tiling Header (tiling.h) - NO knowledge needed
    # =========================================================================

    def get_tiling_header_prompt(
        self,
        plan: dict,
        context: dict,
    ) -> str:
        """Generate prompt for tiling.h (tiling data structure).

        NOTE: This stage does NOT need knowledge - it's pure structure definition.

        Args:
            plan: Joint plan dict from _extract_joint_plan()
            context: Contains 'signature', 'op_name', etc.

        Returns:
            Prompt string for generating tiling.h
        """
        formatted = _format_joint_plan(plan)
        op_name = context.get("op_name", "CustomOp")
        formatted_sig = _format_signature(context.get("signature"))

        return f"""## Role
You are an Ascend C expert generating the **tiling.h** header file.

## Task
Define the tiling data structure based on the joint plan.

## Input

### Operator
- Name: `{op_name}`
- Signature:
{formatted_sig}

### Tiling Fields (from Joint Plan)
{formatted["tiling_fields"]}

### Tiling Execution (reference)
```
{formatted["tiling_execution"]}
```

## Output Format

Output a single code block containing the complete tiling.h file.

### Fixed Structure (DO NOT modify)
```cpp
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF({op_name}TilingData)
    // <VARIABLE: field definitions go here>
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({op_name}, {op_name}TilingData)
}}
```

### Variable Parts
| Part | Format | Example |
|------|--------|---------|
| Field definition | `TILING_DATA_FIELD_DEF(type, name);` | `TILING_DATA_FIELD_DEF(uint32_t, totalLength);` |
| Supported types | `uint32_t`, `int32_t`, `float`, `uint64_t` | - |

## Rules
1. Include ALL fields from "Tiling Fields" section
2. Field names must be valid C++ identifiers (camelCase recommended)
3. One `TILING_DATA_FIELD_DEF` per line, with semicolon
4. No comments needed inside the struct

## Example

Input: Softmax with fields `batchSize`, `featureDim`, `rowsPerCore`

Output:
```cpp
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, featureDim);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, SoftmaxTilingData)
}}
```

Now generate the tiling.h for `{op_name}`:
"""

    # =========================================================================
    # Stage 2: Tiling Host (op_host.cpp)
    # =========================================================================

    def get_tiling_host_prompt(
        self,
        plan: dict,
        context: dict,
        knowledge: str,
        tiling_header: str,
    ) -> str:
        """Generate prompt for op_host.cpp (tiling calculation + InferShape).

        Args:
            plan: Joint plan dict
            context: Contains 'signature', 'op_name', 'python_ref', etc.
            knowledge: Retrieved knowledge context
            tiling_header: Generated tiling.h content from Stage 1

        Returns:
            Prompt string for generating op_host.cpp
        """
        formatted = _format_joint_plan(plan)
        op_name = context.get("op_name", "CustomOp")
        formatted_sig = _format_signature(context.get("signature"))
        python_ref = context.get("python_ref", "")

        return f"""## Role
You are an Ascend C host-side development expert generating the **op_host.cpp** file.

## Task
Implement tiling calculation logic and operator registration.

## Input

### Operator
- Name: `{op_name}`
- Signature:
{formatted_sig}

### Python Reference
```python
{python_ref}
```

### Tiling Execution (from Joint Plan)
```
{formatted["tiling_execution"]}
```

### Tiling Design (from Joint Plan)
{formatted["tiling_proposal"]}

### Generated tiling.h (Stage 1 output)
```cpp
{tiling_header}
```

### Available Knowledge
{knowledge}

## Output Format

Output a single code block containing the complete op_host.cpp file.

### Fixed Structure (DO NOT modify skeleton)
```cpp
#include "{op_name.lower()}_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {{

static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
    // <VARIABLE: tiling calculation logic>
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ops {{
class {op_name} : public OpDef {{
public:
    explicit {op_name}(const char* name) : OpDef(name) {{
        // <VARIABLE: input/output definitions>
        // <VARIABLE: InferShape and Tiling registration>
    }}
}};
OP_ADD({op_name});
}}
```

### Variable Parts

| Part | Pattern | Example |
|------|---------|---------|
| Get shape | `context->GetInputShape(i)->GetStorageShape()` | `auto shape = context->GetInputShape(0)->GetStorageShape();` |
| Get dim | `shape.GetDim(i)` | `uint32_t batchSize = shape.GetDim(0);` |
| Set field | `tiling.set_<fieldName>(value)` | `tiling.set_batchSize(batchSize);` |
| Set block_dim | `context->SetBlockDim(n)` | `context->SetBlockDim(8);` |
| Save tiling | See below | - |
| Define input | `.Input("name").ParamType(REQUIRED).DataType({{...}})` | `.Input("x").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16}})` |
| Define output | `.Output("name").ParamType(REQUIRED).DataType({{...}})` | `.Output("y").ParamType(REQUIRED).DataType({{ge::DT_FLOAT16}})` |
| Register | `.SetInferShape(ge::InferShape).SetTiling(optiling::TilingFunc)` | - |

### Save Tiling Pattern (fixed)
```cpp
tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
```

## Example (Softmax)

```cpp
#include "softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {{

static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t batchSize = shape.GetDim(0);
    uint32_t featureDim = shape.GetDim(1);
    uint32_t coreNum = 8;
    uint32_t rowsPerCore = (batchSize + coreNum - 1) / coreNum;

    SoftmaxTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_featureDim(featureDim);
    tiling.set_rowsPerCore(rowsPerCore);

    context->SetBlockDim(std::min(coreNum, batchSize));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ops {{
class Softmax : public OpDef {{
public:
    explicit Softmax(const char* name) : OpDef(name) {{
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT16, ge::DT_FLOAT}});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT16, ge::DT_FLOAT}});
        this->SetInferShape(ge::InferShape)
            .SetTiling(optiling::TilingFunc);
    }}
}};
OP_ADD(Softmax);
}}
```

Now generate the op_host.cpp for `{op_name}`:
"""

    # =========================================================================
    # Stage 3: Kernel Implementation (op_kernel.cpp)
    # =========================================================================

    def get_kernel_impl_prompt(
        self,
        plan: dict,
        context: dict,
        knowledge: str,
        tiling_header: Optional[str] = None,
    ) -> str:
        """Generate prompt for op_kernel.cpp (kernel implementation).

        Args:
            plan: Joint plan dict
            context: Contains 'signature', 'op_name', 'python_ref', etc.
            knowledge: Retrieved knowledge context
            tiling_header: Generated tiling.h content (None for default tiling)

        Returns:
            Prompt string for generating op_kernel.cpp
        """
        formatted = _format_joint_plan(plan)
        op_name = context.get("op_name", "CustomOp")
        formatted_sig = _format_signature(context.get("signature"))
        python_ref = context.get("python_ref", "")
        tiling_strategy = formatted["tiling_strategy"]

        # Tiling header section
        if tiling_header:
            tiling_section = f"""### Custom Tiling Header (Stage 1 output)
```cpp
{tiling_header}
```"""
        else:
            tiling_section = """### Tiling: Default
Standard tiling parameters available via `GET_TILING_DATA`:
- `totalLength`: total number of elements
- `tileNum`: number of tiles
- `tileLength`: elements per tile
- `lasttileLength`: elements in last tile (may differ)"""

        return f"""## Role
You are an Ascend C kernel development expert generating the **op_kernel.cpp** file.

## Task
Implement the kernel based on the joint plan and pseudocode.

## Input

### Operator
- Name: `{op_name}`
- Signature:
{formatted_sig}

### Python Reference
```python
{python_ref}
```

### Tiling Strategy: `{tiling_strategy}`

{tiling_section}

### Kernel Design (from Joint Plan)
{formatted["kernel_design"]}

### Kernel Pseudocode (from Joint Plan)
```
{formatted["kernel_pseudocode"]}
```

### Tiling Execution (from Joint Plan)
```
{formatted["tiling_execution"]}
```

### Available Knowledge
{knowledge}

## Output Format

Output a single code block containing the complete op_kernel.cpp file.

### Fixed Structure (DO NOT modify skeleton)
```cpp
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class Kernel{op_name} {{
public:
    __aicore__ inline Kernel{op_name}() {{}}
    __aicore__ inline void Init(/* <VARIABLE: GM_ADDR params, tiling params> */) {{
        // <VARIABLE: initialization>
    }}
    __aicore__ inline void Process() {{
        // <VARIABLE: main loop>
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress) {{
        // <VARIABLE: GM -> local>
    }}
    __aicore__ inline void Compute(int32_t progress) {{
        // <VARIABLE: compute>
    }}
    __aicore__ inline void CopyOut(int32_t progress) {{
        // <VARIABLE: local -> GM>
    }}

    // <VARIABLE: member variables>
}};

extern "C" __global__ __aicore__ void {op_name.lower()}_custom(
    /* <VARIABLE: GM_ADDR inputs, outputs> */ GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
    Kernel{op_name} op;
    op.Init(/* <VARIABLE: pass params> */);
    op.Process();
}}
```

### Variable Parts

| Part | Pattern | Example |
|------|---------|---------|
| GlobalTensor init | `xGm.SetGlobalBuffer((__gm__ T*)addr)` | `xGm.SetGlobalBuffer((__gm__ half*)x);` |
| Pipe init | `pipe.InitBuffer(queue, BUFFER_NUM, size)` | `pipe.InitBuffer(inQueue, BUFFER_NUM, tileLength * sizeof(half));` |
| Alloc tensor | `queue.AllocTensor<T>()` | `LocalTensor<half> xLocal = inQueue.AllocTensor<half>();` |
| DataCopy in | `DataCopy(local, gm[offset], count)` | `DataCopy(xLocal, xGm[progress * tileLength], tileLength);` |
| EnQue | `queue.EnQue(tensor)` | `inQueue.EnQue(xLocal);` |
| DeQue | `queue.DeQue<T>()` | `LocalTensor<half> xLocal = inQueue.DeQue<half>();` |
| Compute | `Op(dst, src, ...)` | `Add(zLocal, xLocal, yLocal, tileLength);` |
| DataCopy out | `DataCopy(gm[offset], local, count)` | `DataCopy(zGm[progress * tileLength], zLocal, tileLength);` |
| FreeTensor | `queue.FreeTensor(tensor)` | `inQueue.FreeTensor(xLocal);` |
| Get tiling | `tilingData.<field>` | `tilingData.totalLength` |

### Member Variable Patterns

| Type | Declaration |
|------|-------------|
| Pipe | `TPipe pipe;` |
| Input queue | `TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;` |
| Output queue | `TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;` |
| Global tensor | `GlobalTensor<half> xGm;` |
| Tiling param | `uint32_t tileNum;` |

## Example (element-wise Add)

```cpp
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {{
public:
    __aicore__ inline KernelAdd() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum) {{
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);
        zGm.SetGlobalBuffer((__gm__ half*)z);
        this->tileNum = tileNum;
        this->tileLength = totalLength / tileNum;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength * sizeof(half));
    }}

    __aicore__ inline void Process() {{
        for (int32_t i = 0; i < tileNum; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress) {{
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        DataCopy(yLocal, yGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }}

    __aicore__ inline void Compute(int32_t progress) {{
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueue.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, tileLength);
        outQueue.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress) {{
        LocalTensor<half> zLocal = outQueue.DeQue<half>();
        DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueue.FreeTensor(zLocal);
    }}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<half> xGm, yGm, zGm;
    uint32_t tileNum;
    uint32_t tileLength;
}};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}}
```

Now generate the op_kernel.cpp for `{op_name}`:
"""

    # =========================================================================
    # Legacy interface (backward compatibility)
    # =========================================================================

    def get_tiling_impl_prompt(self, plan: dict, knowledge: str) -> str:
        """Legacy interface - deprecated.

        Use get_tiling_header_prompt + get_tiling_host_prompt instead.
        """
        context = {"op_name": "CustomOp", "signature": {}}
        return self.get_tiling_host_prompt(plan, context, knowledge, "")
