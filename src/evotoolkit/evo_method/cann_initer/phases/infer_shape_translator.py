# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""InferShape Translator: PyTorch C++ → Ascend C++ InferShape

Translates shape inference code from pybind format to Ascend InferShape format.
Uses minimal context for high accuracy syntax translation.
"""

import re
from typing import Callable, Optional


# Template for same-shape case (most common)
SAME_SHAPE_TEMPLATE = """    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;"""


def _is_same_shape(shape_code: str) -> bool:
    """Check if shape code is simple same-shape pattern."""
    # Patterns like: auto output_shape = x.sizes();
    # or: auto output_shape = q.sizes();
    pattern = r"auto\s+output_shape\s*=\s*\w+\.sizes\(\)\s*;"
    return bool(re.match(pattern, shape_code.strip()))


def _get_translation_prompt(shape_code: str, signature: dict) -> str:
    """Generate minimal prompt for shape translation."""
    # Extract input names for context
    inputs = signature.get("inputs", [])
    input_names = [inp.get("name", f"input{i}") for i, inp in enumerate(inputs)]

    return f"""## Task
Translate PyTorch C++ shape code to Ascend InferShape format.

## Input Code (PyTorch C++)
```cpp
{shape_code}
```

## Input Tensors
{', '.join(input_names)}

## API Mapping
| PyTorch | Ascend |
|---------|--------|
| `tensor.size(i)` | `shape->GetDim(i)` |
| `tensor.dim()` | `shape->GetDimNum()` |
| `{{a, b, c}}` initializer | `y_shape->AppendDim(a); y_shape->AppendDim(b); ...` |

## Template
```cpp
// Get input shapes (use index 0, 1, 2... for each input)
const gert::Shape* x_shape = context->GetInputShape(0);
// For second input: const gert::Shape* y_shape_in = context->GetInputShape(1);

// Get output shape pointer
gert::Shape* y_shape = context->GetOutputShape(0);

// Set output dimensions
y_shape->AppendDim(x_shape->GetDim(0));  // or direct value
```

## Response Format
<response>
    // your translated code here
</response>

Now translate the input code. Output ONLY the <response> block:
"""


class InferShapeTranslator:
    """Translate PyTorch shape code to Ascend InferShape code.

    Design:
    - Same-shape case: Use template directly (no LLM needed)
    - Other cases: Small context LLM translation

    Usage:
        translator = InferShapeTranslator(llm_call)
        infer_shape_body = translator.translate(shape_code, signature)
    """

    def __init__(self, llm_client: Optional[Callable[[str], str]] = None):
        """
        Args:
            llm_client: Callable that takes prompt and returns response.
                        If None, only template-based translation is available.
        """
        self.llm_client = llm_client

    def translate(
        self,
        shape_inference_code: str,
        signature: dict,
        output_equals_input_shape: bool = False,
    ) -> str:
        """Translate PyTorch shape code to Ascend InferShape body.

        Args:
            shape_inference_code: PyTorch C++ shape code from pybind
                e.g., "auto output_shape = x.sizes();"
                e.g., "auto output_shape = {a.size(0), b.size(1)};"
            signature: Operator signature dict
            output_equals_input_shape: If True, use same-shape template

        Returns:
            InferShape body code for Ascend C++
        """
        # Fast path: same-shape case
        if output_equals_input_shape or _is_same_shape(shape_inference_code):
            return SAME_SHAPE_TEMPLATE

        # Need translation
        if not self.llm_client:
            # No LLM available, fallback to same-shape template
            return SAME_SHAPE_TEMPLATE

        # Use LLM for translation
        prompt = _get_translation_prompt(shape_inference_code, signature)
        response = self.llm_client(prompt)

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> str:
        """Parse LLM response to extract InferShape code."""
        # Extract from <response> tag
        match = re.search(r"<response>(.*?)</response>", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: try code block
        code_match = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Last resort: return as-is with indentation
        lines = response.strip().split('\n')
        return '\n'.join('    ' + line for line in lines)
