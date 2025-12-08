# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Phase 0: Compute pattern analysis prompt for Ascend C operator generation."""

from typing import Any, Dict, List


def _format_params(params: List[Dict], label: str) -> str:
    """Format parameter list for display."""
    if not params:
        return f"{label}: (none)"
    lines = [f"{label}:"]
    for p in params:
        tensor_mark = " [tensor]" if p.get("is_tensor") else ""
        lines.append(f"  - {p['name']}: {p['dtype']}{tensor_mark}")
    return "\n".join(lines)


def _format_signature(signature: Any) -> str:
    """Format operator signature for clear display in prompt."""
    if isinstance(signature, dict):
        sig = signature
    elif hasattr(signature, "to_dict"):
        sig = signature.to_dict()
    else:
        sig = {"op_name": "Unknown", "inputs": [], "outputs": [], "init_params": []}

    parts = [
        f"Operator: {sig.get('op_name', 'Unknown')}",
        _format_params(sig.get("inputs", []), "Inputs"),
        _format_params(sig.get("outputs", []), "Outputs"),
    ]

    init_params = sig.get("init_params", [])
    if init_params:
        parts.append(_format_params(init_params, "Init Params"))

    return "\n".join(parts)


class Phase0PromptMixin:
    """Phase 0: Compute pattern analysis for Ascend C operator generation."""

    def get_pattern_analysis_prompt(self, python_ref: str, signature: Any) -> str:
        """
        Generate prompt for compute pattern analysis.

        Returns a prompt that asks LLM to analyze the Python reference code
        and determine the compute pattern and generation strategies for
        Ascend C operator implementation.
        """
        formatted_sig = _format_signature(signature)

        return f"""## Your Role

You are the **analysis agent** in a multi-agent Ascend C code generation pipeline.

Your job is to:
1. Analyze the Python reference and classify its compute pattern
2. Analyze input/output shape relationship and provide shape inference formula
3. Decide which components need custom generation vs default templates
4. Provide a clear functionality description for downstream agents

**Important:** You do NOT generate any code. Downstream agents will read your output and generate the actual Ascend C code based on your analysis.

## Input

### Python Reference
```python
{python_ref}
```

### Operator Signature
{formatted_sig}

## Rules

### Compute Pattern Categories
| Pattern | Criteria | Examples |
|---------|----------|----------|
| element-wise | output shape = input shape, per-element ops | relu, sigmoid, add, mul, exp |
| reduction | output shape <= input shape, aggregation | sum, mean, max, softmax |
| matmul | matrix multiplication | mm, bmm, linear |
| broadcast | shape broadcasting involved | add with mismatched shapes |
| other | complex multi-step patterns | attention, convolution |

### Strategy Decision
| Compute Pattern | tiling | pybind |
|-----------------|--------|--------|
| element-wise | default | default |
| all others | generate | generate |

**Strategy meanings:**
- `default`: Downstream agent will use pre-defined template (no generation needed)
- `generate`: Downstream agent must generate custom code for this component

## Response Format

Respond inside `<response>` tags using the exact section headers:

<response>
## Compute Pattern
<one of: element-wise | reduction | matmul | broadcast | other>

## Output Equals Input Shape
<true | false>

## Shape Inference
input: <describe input shape, e.g., "[B, M, K]" or "[N]">
output: <describe output shape, e.g., "[B, M, N]" or "same as input">
formula: <C++ code to compute output_shape from input tensors>

## Strategies
kernel: generate
tiling: <default | generate>
pybind: <default | generate>

## Functionality
<1-2 sentences describing what this operator does mathematically>

## Reasoning
<1-2 sentences explaining your pattern classification>
</response>

## Examples

### Element-wise (ReLU)
<response>
## Compute Pattern
element-wise

## Output Equals Input Shape
true

## Shape Inference
input: [*] (any shape)
output: same as input
formula: auto output_shape = x.sizes();

## Strategies
kernel: generate
tiling: default
pybind: default

## Functionality
Applies ReLU activation max(0, x) element-wise to the input tensor.

## Reasoning
ReLU is a per-element operation. Output shape equals input shape.
</response>

### Reduction (Softmax)
<response>
## Compute Pattern
reduction

## Output Equals Input Shape
true

## Shape Inference
input: [B, D] where B=batch, D=features
output: same as input (softmax preserves shape)
formula: auto output_shape = x.sizes();

## Strategies
kernel: generate
tiling: generate
pybind: generate

## Functionality
Applies softmax along dimension 1: exp(x_i) / sum(exp(x_j)), normalizing to probability distribution.

## Reasoning
Softmax involves reduction (sum) along a dimension, but output shape equals input shape.
</response>

### MatMul
<response>
## Compute Pattern
matmul

## Output Equals Input Shape
false

## Shape Inference
input: a=[M, K], b=[K, N]
output: [M, N]
formula: auto output_shape = {{a.size(0), b.size(1)}};

## Strategies
kernel: generate
tiling: generate
pybind: generate

## Functionality
Performs matrix multiplication C = A @ B where A is [M,K] and B is [K,N].

## Reasoning
MatMul changes output shape based on input dimensions. Requires custom shape inference.
</response>

Now analyze the given operator. Output ONLY the `<response>` block, nothing else:
"""
