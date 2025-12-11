# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Operator signature parser for Python reference code.

This module extracts operator signature (inputs, outputs) from
Python reference implementations using AST parsing.

Supports MultiKernelBench reference format:
- Model class with __init__ and forward methods
- get_inputs() and get_init_inputs() functions

The actual invocation pattern is:
    model = Model(*get_init_inputs())
    result = model(*get_inputs())

So we extract dtype info from get_inputs() and get_init_inputs(),
not from type hints (which are often missing).
"""

import ast
import re
from typing import Any, Dict, List, Optional


class OperatorSignatureParser:
    """
    Parse Python reference code to extract operator signature.

    Extracts:
    - Inputs from get_inputs() function
    - Init params from get_init_inputs() function
    - Parameter names from Model.forward() and Model.__init__()
    - dtype inferred from torch.randn(), torch.zeros(), etc.
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
        """
        try:
            tree = ast.parse(python_code)
        except SyntaxError:
            return self._parse_with_regex(python_code, op_name)

        # 1. Parse get_inputs() to get forward inputs
        inputs = self._parse_get_inputs(tree)

        # 2. Parse get_init_inputs() to get __init__ params
        init_params = self._parse_get_init_inputs(tree)

        # 3. Get parameter names from Model class (for better naming)
        model_info = self._find_model_class(tree)
        if model_info:
            # Use parameter names from forward() if available
            inputs = self._merge_names(inputs, model_info.get("forward_params", []))
            init_params = self._merge_names(init_params, model_info.get("init_params", []))

        # 4. Infer outputs (default: same dtype as first input)
        outputs = self._infer_outputs(inputs)

        return {
            "op_name": op_name,
            "inputs": inputs,
            "outputs": outputs,
            "init_params": init_params,
        }

    def _parse_get_inputs(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Parse get_inputs() function to extract forward inputs.

        Analyzes:
            def get_inputs():
                Q = torch.randn(batch_size, seq_len, d_model)
                K = torch.randn(batch_size, seq_len, d_model, dtype=torch.float16)
                return [Q, K, V]

        Returns list of {name, dtype, is_tensor} dicts.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
                return self._parse_input_function(node)
        return []

    def _parse_get_init_inputs(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Parse get_init_inputs() function to extract __init__ params.

        Handles both list and dict returns:
            return []
            return [alpha, beta]
            return {"alpha": 1.0, "beta": 2.0}

        Returns list of {name, dtype, is_tensor, default?} dicts.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
                return self._parse_init_input_function(node)
        return []

    def _parse_input_function(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Parse a get_inputs() function body.

        Extracts variable assignments and return statement to build input list.
        """
        # Build variable -> dtype mapping from assignments
        var_dtypes: Dict[str, str] = {}
        var_is_tensor: Dict[str, bool] = {}

        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        dtype, is_tensor = self._infer_dtype_from_expr(stmt.value)
                        var_dtypes[target.id] = dtype
                        var_is_tensor[target.id] = is_tensor

        # Find return statement and extract variable order
        return_stmt = self._find_return_stmt(func_node)
        if return_stmt is None:
            return []

        inputs = []
        if isinstance(return_stmt.value, ast.List):
            # return [Q, K, V]
            for elt in return_stmt.value.elts:
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    inputs.append({
                        "name": var_name.lower(),
                        "dtype": var_dtypes.get(var_name, "float"),
                        "is_tensor": var_is_tensor.get(var_name, True),
                    })
                else:
                    # Inline expression, assume tensor
                    dtype, is_tensor = self._infer_dtype_from_expr(elt)
                    inputs.append({
                        "name": f"x{len(inputs)}",
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    })

        return inputs

    def _parse_init_input_function(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Parse get_init_inputs() function body.

        Handles:
            return []
            return [alpha, beta]
            return {"alpha": 1.0}
        """
        # Build variable -> (dtype, value) mapping
        var_info: Dict[str, Dict[str, Any]] = {}

        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        dtype, is_tensor = self._infer_dtype_from_expr(stmt.value)
                        default = self._extract_literal_value(stmt.value)
                        var_info[target.id] = {
                            "dtype": dtype,
                            "is_tensor": is_tensor,
                            "default": default,
                        }

        return_stmt = self._find_return_stmt(func_node)
        if return_stmt is None:
            return []

        params = []

        if isinstance(return_stmt.value, ast.List):
            # return [alpha, beta] or return []
            for elt in return_stmt.value.elts:
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    info = var_info.get(var_name, {"dtype": "float", "is_tensor": False})
                    param = {
                        "name": var_name.lower(),
                        "dtype": info["dtype"],
                        "is_tensor": info["is_tensor"],
                    }
                    if info.get("default") is not None:
                        param["default"] = info["default"]
                    params.append(param)
                else:
                    # Inline literal
                    dtype, is_tensor = self._infer_dtype_from_expr(elt)
                    default = self._extract_literal_value(elt)
                    param = {
                        "name": f"param{len(params)}",
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    }
                    if default is not None:
                        param["default"] = default
                    params.append(param)

        elif isinstance(return_stmt.value, ast.Dict):
            # return {"alpha": 1.0, "beta": 2.0}
            for key, value in zip(return_stmt.value.keys, return_stmt.value.values):
                if isinstance(key, ast.Constant):
                    name = str(key.value)
                    dtype, is_tensor = self._infer_dtype_from_expr(value)
                    default = self._extract_literal_value(value)
                    param = {
                        "name": name,
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    }
                    if default is not None:
                        param["default"] = default
                    params.append(param)

        return params

    def _find_return_stmt(self, func_node: ast.FunctionDef) -> Optional[ast.Return]:
        """Find the first return statement in a function."""
        for stmt in func_node.body:
            if isinstance(stmt, ast.Return):
                return stmt
        return None

    def _infer_dtype_from_expr(self, expr: ast.AST) -> tuple:
        """
        Infer dtype and is_tensor from an expression.

        Supports:
            - torch.randn(...) → (float, True)
            - torch.randn(..., dtype=torch.float16) → (float16, True)
            - torch.zeros(..., dtype=torch.int32) → (int32, True)
            - torch.randint(...) → (int64, True)
            - 1.0 → (float, False)
            - 1 → (int, False)
            - True → (bool, False)

        Returns:
            (dtype: str, is_tensor: bool)
        """
        if isinstance(expr, ast.Call):
            # Check for dtype keyword argument
            for kw in expr.keywords:
                if kw.arg == "dtype":
                    dtype = self._extract_torch_dtype(kw.value)
                    return (dtype, True)

            # Infer from function name
            func_name = self._get_call_name(expr)

            # Tensor creation functions
            tensor_funcs_float = {
                "torch.randn", "torch.rand", "torch.zeros", "torch.ones",
                "torch.empty", "torch.full", "torch.randn_like", "torch.zeros_like",
            }
            tensor_funcs_int = {"torch.randint", "torch.arange"}

            if func_name in tensor_funcs_float:
                return ("float", True)
            elif func_name in tensor_funcs_int:
                return ("int64", True)
            elif "torch" in func_name or "tensor" in func_name.lower():
                return ("float", True)

            # Non-tensor function call
            return ("float", False)

        elif isinstance(expr, ast.Constant):
            # Literal values
            value = expr.value
            if isinstance(value, bool):
                return ("bool", False)
            elif isinstance(value, int):
                return ("int", False)
            elif isinstance(value, float):
                return ("float", False)
            elif isinstance(value, str):
                return ("str", False)
            return ("float", False)

        elif isinstance(expr, ast.List):
            # List literal, check first element
            if expr.elts:
                inner_dtype, _ = self._infer_dtype_from_expr(expr.elts[0])
                return (f"list_{inner_dtype}", False)
            return ("list_int", False)

        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            # Negative number: -1.0
            inner_dtype, is_tensor = self._infer_dtype_from_expr(expr.operand)
            return (inner_dtype, is_tensor)

        return ("float", False)

    def _get_call_name(self, call_node: ast.Call) -> str:
        """Get the full name of a function call (e.g., 'torch.randn')."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            parts = []
            node = func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))
        return ""

    def _extract_torch_dtype(self, node: ast.AST) -> str:
        """
        Extract dtype from torch.float16, torch.int32, etc.
        """
        dtype_map = {
            "float16": "float16",
            "float32": "float",
            "float64": "double",
            "half": "float16",
            "float": "float",
            "double": "double",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "int": "int64",
            "long": "int64",
            "bool": "bool",
            "bfloat16": "bfloat16",
        }

        if isinstance(node, ast.Attribute):
            return dtype_map.get(node.attr, "float")
        elif isinstance(node, ast.Name):
            return dtype_map.get(node.id, "float")

        return "float"

    def _extract_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._extract_literal_value(node.operand)
            if isinstance(inner, (int, float)):
                return -inner
        elif isinstance(node, ast.List):
            return [self._extract_literal_value(elt) for elt in node.elts]
        return None

    def _find_model_class(self, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Find Model class and extract parameter names from __init__ and forward.

        Returns dict with forward_params and init_params (just names).
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                forward_params = []
                init_params = []

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "forward":
                            for arg in item.args.args:
                                if arg.arg != "self":
                                    forward_params.append(arg.arg.lower())
                        elif item.name == "__init__":
                            for arg in item.args.args:
                                if arg.arg != "self":
                                    init_params.append(arg.arg.lower())

                return {
                    "forward_params": forward_params,
                    "init_params": init_params,
                }

        return None

    def _merge_names(
        self, parsed_inputs: List[Dict[str, Any]], param_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Merge parameter names from Model class into parsed inputs.

        If param_names are available, use them instead of auto-generated names.
        """
        if not param_names:
            return parsed_inputs

        result = []
        for i, inp in enumerate(parsed_inputs):
            new_inp = inp.copy()
            if i < len(param_names):
                new_inp["name"] = param_names[i]
            result.append(new_inp)

        return result

    def _infer_outputs(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Infer output info.

        Default: single tensor output with same dtype as first input.
        """
        dtype = "float"
        if inputs and inputs[0].get("is_tensor", True):
            dtype = inputs[0].get("dtype", "float")

        return [{"name": "output", "dtype": dtype, "is_tensor": True}]

    def _parse_with_regex(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """
        Fallback regex-based parsing for malformed Python code.
        """
        inputs = []

        # Try to find get_inputs function
        get_inputs_match = re.search(
            r"def\s+get_inputs\s*\(\s*\):\s*\n(.*?)(?=\ndef|\Z)",
            python_code,
            re.DOTALL,
        )

        if get_inputs_match:
            body = get_inputs_match.group(1)
            # Find torch.randn calls
            randn_matches = re.findall(r"(\w+)\s*=\s*torch\.\w+\(", body)
            for var_name in randn_matches:
                inputs.append({
                    "name": var_name.lower(),
                    "dtype": "float",
                    "is_tensor": True,
                })

        if not inputs:
            # Default inputs
            inputs = [
                {"name": "x", "dtype": "float", "is_tensor": True},
            ]

        return {
            "op_name": op_name,
            "inputs": inputs,
            "outputs": [{"name": "output", "dtype": "float", "is_tensor": True}],
            "init_params": [],
        }
