# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Operator signature parser for Python reference code.

This module extracts operator signature (inputs, outputs) from
Python reference implementations using AST parsing.

Supports MultiKernelBench reference format:
- Model class with __init__ and forward methods
- get_inputs() and get_init_inputs() functions
"""

import ast
import re
from typing import Any, Dict, List, Optional


class OperatorSignatureParser:
    """
    Parse Python reference code to extract operator signature.

    Extracts:
    - Model.__init__ parameters (init_params)
    - Model.forward parameters (inputs)
    - Return values (outputs)
    - Type hints (if available)
    """

    def parse(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """
        Parse Python code and extract operator signature.

        Args:
            python_code: Python reference implementation (MultiKernelBench format)
            op_name: Operator name (typically from filename, e.g., "elu", "add")

        Returns:
            Signature dict containing:
                - op_name: Operator name (passed in, not parsed)
                - inputs: List of forward() input info [{name, dtype, is_tensor}]
                - outputs: List of output info [{name, dtype, is_tensor}]
                - init_params: List of __init__() param info [{name, dtype, is_tensor, default}]

        Note:
            In MultiKernelBench, op_name comes from the dataset/filename, not from
            parsing the code. The code only needs Model class with __init__/forward,
            plus get_inputs() and get_init_inputs() functions.
        """
        try:
            tree = ast.parse(python_code)
        except SyntaxError:
            # Fallback to regex parsing
            return self._parse_with_regex(python_code, op_name)

        # Find Model class and extract both __init__ and forward
        model_info = self._find_model_class(tree)

        if model_info is None:
            return self._parse_with_regex(python_code, op_name)

        return {
            "op_name": op_name,
            "inputs": model_info["inputs"],
            "outputs": model_info["outputs"],
            "init_params": model_info.get("init_params", []),
        }

    def _find_model_class(self, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Find Model class and extract __init__ and forward info.

        Returns:
            Dict with inputs, outputs, init_params
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                init_method = None
                forward_method = None

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__init__":
                            init_method = item
                        elif item.name == "forward":
                            forward_method = item

                if forward_method is None:
                    return None

                # Extract forward info (inputs/outputs)
                forward_info = self._extract_forward_info(forward_method)

                # Extract __init__ params
                init_params = []
                if init_method is not None:
                    init_params = self._extract_init_params(init_method)

                return {
                    "inputs": forward_info["inputs"],
                    "outputs": forward_info["outputs"],
                    "init_params": init_params,
                }

        return None

    def _extract_init_params(self, init_method: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Extract __init__ parameters (excluding self).

        Returns list of {name, dtype, is_tensor, default} dicts.
        """
        params = []
        args = init_method.args

        # Get defaults - they align to the end of args
        num_defaults = len(args.defaults)
        num_args = len(args.args)

        for i, arg in enumerate(args.args):
            if arg.arg == "self":
                continue

            type_info = self._extract_type_hint(arg.annotation)
            param_info = {
                "name": arg.arg,
                "dtype": type_info["dtype"],
                "is_tensor": type_info["is_tensor"],
            }

            # Check if this arg has a default value
            default_idx = i - (num_args - num_defaults)
            if default_idx >= 0 and default_idx < len(args.defaults):
                default_node = args.defaults[default_idx]
                param_info["default"] = self._extract_default_value(default_node)

            params.append(param_info)

        return params

    def _extract_default_value(self, node: ast.AST) -> Any:
        """Extract default value from AST node."""
        # ast.Constant covers all literal values in Python 3.8+
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id  # Return variable name as string
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers like -1.0
            inner = self._extract_default_value(node.operand)
            if isinstance(inner, (int, float)):
                return -inner
        return None

    def _extract_forward_info(self, func: ast.FunctionDef) -> Dict[str, Any]:
        """Extract forward method info (inputs and outputs)."""
        inputs = []

        # Extract parameters (skip 'self')
        for arg in func.args.args:
            if arg.arg == "self":
                continue
            type_info = self._extract_type_hint(arg.annotation)
            inputs.append({
                "name": arg.arg,
                "dtype": type_info["dtype"],
                "is_tensor": type_info["is_tensor"],
            })

        # Extract return type if available
        outputs = []
        if func.returns:
            type_info = self._extract_type_hint(func.returns)
            outputs.append({
                "name": "z",
                "dtype": type_info["dtype"],
                "is_tensor": type_info["is_tensor"],
            })
        else:
            outputs.append({"name": "z", "dtype": "float", "is_tensor": True})

        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    def _extract_type_hint(self, annotation: Optional[ast.AST]) -> Dict[str, Any]:
        """
        Extract type hint info from AST annotation.

        Returns:
            Dict with:
                - is_tensor: True for torch.Tensor, np.ndarray etc.
                - dtype: Data precision (float, float16, int32, list_int, etc.)
        """
        if annotation is None:
            return {"is_tensor": False, "dtype": "float"}

        if isinstance(annotation, ast.Name):
            name = annotation.id.lower()
            if "tensor" in name or "ndarray" in name:
                return {"is_tensor": True, "dtype": "float"}
            # Scalar types
            return {"is_tensor": False, "dtype": name}

        if isinstance(annotation, ast.Subscript):
            # Get base type name (e.g., "List", "Tensor", "Optional")
            base_name = ""
            if isinstance(annotation.value, ast.Name):
                base_name = annotation.value.id.lower()
            elif isinstance(annotation.value, ast.Attribute):
                base_name = annotation.value.attr.lower()

            # Handle List types -> scalar with list_* dtype
            if base_name == "list":
                inner_type = "int"  # default
                if isinstance(annotation.slice, ast.Name):
                    inner_type = annotation.slice.id.lower()
                return {"is_tensor": False, "dtype": f"list_{inner_type}"}

            # Handle Optional[X] -> recurse into X
            if base_name == "optional":
                if isinstance(annotation.slice, ast.Name):
                    return self._extract_type_hint(annotation.slice)
                elif isinstance(annotation.slice, ast.Subscript):
                    return self._extract_type_hint(annotation.slice)

            # Handle Tensor[dtype] types
            if "tensor" in base_name:
                dtype = "float"
                if isinstance(annotation.slice, ast.Name):
                    dtype = annotation.slice.id.lower()
                return {"is_tensor": True, "dtype": dtype}

            # Default: assume tensor for unknown subscript types
            dtype = "float"
            if isinstance(annotation.slice, ast.Name):
                dtype = annotation.slice.id.lower()
            return {"is_tensor": True, "dtype": dtype}

        if isinstance(annotation, ast.Attribute):
            # Handle torch.Tensor, torch.FloatTensor, np.ndarray, etc.
            attr = annotation.attr.lower()
            if "tensor" in attr or "ndarray" in attr:
                # Check for specific tensor types like FloatTensor, HalfTensor
                if "half" in attr or "float16" in attr:
                    return {"is_tensor": True, "dtype": "float16"}
                elif "int" in attr:
                    return {"is_tensor": True, "dtype": "int32"}
                return {"is_tensor": True, "dtype": "float"}
            return {"is_tensor": False, "dtype": attr}

        return {"is_tensor": False, "dtype": "float"}

    def _parse_with_regex(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """
        Fallback regex-based parsing for malformed Python code.

        Args:
            python_code: Python code string
            op_name: Operator name

        Returns:
            Basic signature dict
        """
        inputs = []

        # Try to find forward method or any function definition
        func_match = re.search(
            r"def\s+forward\s*\((.*?)\)", python_code, re.DOTALL
        )
        if not func_match:
            func_match = re.search(
                r"def\s+\w+\s*\((.*?)\)", python_code, re.DOTALL
            )

        if func_match:
            params_str = func_match.group(1)

            # Parse parameters
            params = [p.strip() for p in params_str.split(",") if p.strip()]
            for param in params:
                # Handle "self" parameter
                param_name = param.split(":")[0].strip().split("=")[0].strip()
                if param_name and param_name != "self":
                    # Check if type hint contains "tensor"
                    is_tensor = "tensor" in param.lower()
                    inputs.append({"name": param_name, "dtype": "float", "is_tensor": is_tensor})
        else:
            # Default inputs if no function found
            inputs = [
                {"name": "x", "dtype": "float", "is_tensor": True},
                {"name": "y", "dtype": "float", "is_tensor": True},
            ]

        return {
            "op_name": op_name,
            "inputs": inputs,
            "outputs": [{"name": "z", "dtype": "float", "is_tensor": True}],
            "init_params": [],
        }
