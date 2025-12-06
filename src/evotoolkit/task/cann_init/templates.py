# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Ascend C code templates for operator generation.

This module provides template generation for the 6 components of an Ascend C operator:
1. project_json_src - Operator project configuration
2. host_tiling_src - Tiling data structure definition
3. host_operator_src - Host-side operator implementation
4. kernel_src - Device kernel (provided by LLM)
5. python_bind_src - Python binding via pybind11
6. model_src - Test model for verification

Only kernel_src needs to be provided by LLM, others are auto-generated.
"""

import json
from typing import Any, Dict, List, Optional


class AscendCTemplateGenerator:
    """
    Generate Ascend C operator code from templates.

    Given an operator signature and kernel code, generates all 6 components
    needed for a complete Ascend C operator.
    """

    def __init__(self, signature: Dict[str, Any]):
        """
        Initialize with operator signature.

        Args:
            signature: Operator signature containing:
                - op_name: Operator name (e.g., "add")
                - inputs: List of input info [{name, dtype, is_tensor}]
                - outputs: List of output info [{name, dtype, is_tensor}]
                - init_params: List of __init__ param info [{name, dtype, is_tensor, default}]
        """
        self.signature = signature
        self.op_name = signature["op_name"]
        self.op_name_lower = self.op_name.lower()
        self.op_name_capital = self._to_pascal_case(self.op_name)
        self.op_custom = f"{self.op_name_lower}_custom"
        self.op_custom_capital = self._to_pascal_case(self.op_custom)

    def _to_pascal_case(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))

    def generate(
        self,
        kernel_src: str,
        block_dim: int = 8,
        tiling_fields: Optional[List[Dict[str, str]]] = None,
        tiling_func_body: Optional[str] = None,
        host_tiling_src: Optional[str] = None,
        host_operator_src: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate all 6 code components.

        Tiling Modes:
        1. Direct mode: If host_tiling_src or host_operator_src provided, use directly
        2. Template mode: Use tiling_fields and tiling_func_body to generate
        3. Default mode: Use built-in defaults (for element-wise operators)

        Args:
            kernel_src: Kernel code (from LLM)
            block_dim: Number of parallel cores
            tiling_fields: Custom tiling fields (template mode)
            tiling_func_body: Custom TilingFunc body (template mode)
            host_tiling_src: Complete tiling header (direct mode)
            host_operator_src: Complete host operator (direct mode)

        Returns:
            Dictionary with all 6 code components
        """
        # Collect all scalar params (non-tensor) for tiling
        scalar_params = self._collect_scalar_params()

        # Generate host_tiling_src
        if host_tiling_src is None:
            # Template mode or default mode
            fields = tiling_fields if tiling_fields is not None else self._default_tiling_fields()
            # Add scalar params to tiling fields
            fields = fields + self._scalar_params_to_tiling_fields(scalar_params)
            host_tiling_src = self._gen_host_tiling(fields)

        # Generate host_operator_src
        if host_operator_src is None:
            # Template mode or default mode
            base_func_body = tiling_func_body if tiling_func_body is not None else self._default_tiling_func_body()
            # Add scalar params handling to tiling func body
            func_body = self._add_scalar_params_to_tiling_func(base_func_body, scalar_params)
            host_operator_src = self._gen_host_operator(block_dim, func_body)

        return {
            "project_json_src": self._gen_project_json(),
            "host_tiling_src": host_tiling_src,
            "host_operator_src": host_operator_src,
            "kernel_src": kernel_src,
            "python_bind_src": self._gen_python_bind(),
            "model_src": self._gen_model_src(),
        }

    def _collect_scalar_params(self) -> List[Dict[str, Any]]:
        """Collect all scalar (non-tensor) parameters from inputs and init_params."""
        scalar_params = []

        for inp in self.signature.get("inputs", []):
            if not inp.get("is_tensor", True):
                scalar_params.append(inp)

        for param in self.signature.get("init_params", []):
            if not param.get("is_tensor", False):
                scalar_params.append(param)

        return scalar_params

    def _scalar_params_to_tiling_fields(self, scalar_params: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert scalar params to tiling field definitions."""
        fields = []
        for param in scalar_params:
            cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
            # Convert to camelCase for tiling field
            field_name = self._to_camel_case(param["name"])
            fields.append({"name": field_name, "type": cpp_type})
        return fields

    def _to_camel_case(self, name: str) -> str:
        """Convert snake_case to camelCase."""
        parts = name.split("_")
        return parts[0] + "".join(word.capitalize() for word in parts[1:])

    def _add_scalar_params_to_tiling_func(self, base_func_body: str, scalar_params: List[Dict[str, Any]]) -> str:
        """Add scalar params retrieval from attrs to tiling func body."""
        if not scalar_params:
            return base_func_body

        # Generate attr retrieval code
        attr_code = "\n    // Get scalar attrs\n"
        attr_code += "    const gert::RuntimeAttrs *attrs = context->GetAttrs();\n"

        for i, param in enumerate(scalar_params):
            cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
            field_name = self._to_camel_case(param["name"])
            attr_code += f"    const {cpp_type} *{field_name}Ptr = attrs->GetAttrPointer<{cpp_type}>({i});\n"
            attr_code += f"    tiling.set_{field_name}(*{field_name}Ptr);\n"

        return base_func_body + attr_code

    def _default_tiling_fields(self) -> List[Dict[str, str]]:
        """Default tiling fields for simple element-wise operators."""
        return [
            {"name": "totalLength", "type": "uint32_t"},
            {"name": "tileNum", "type": "uint32_t"},
        ]

    def _default_tiling_func_body(self) -> str:
        """Default TilingFunc body for simple element-wise operators."""
        return """
    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalLength *= shape.GetDim(i);
    }

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(BLOCK_DIM);
"""

    def _gen_project_json(self) -> str:
        """Generate project JSON configuration."""
        inputs = self.signature.get("inputs", [])
        outputs = self.signature.get("outputs", [])
        init_params = self.signature.get("init_params", [])

        input_desc = []
        attr_desc = []

        # Process all inputs
        for inp in inputs:
            if inp.get("is_tensor", True):
                cann_type = self._dtype_to_cann_json(inp.get("dtype", "float"))
                input_desc.append({
                    "name": inp["name"],
                    "param_type": "required",
                    "format": ["ND"],
                    "type": [cann_type],
                })
            else:
                # Scalar input as attr
                attr_info = {
                    "name": inp["name"],
                    "param_type": "required",
                    "type": inp.get("dtype", "float"),
                }
                attr_desc.append(attr_info)

        # Process init_params
        for param in init_params:
            if param.get("is_tensor", False):
                cann_type = self._dtype_to_cann_json(param.get("dtype", "float"))
                input_desc.append({
                    "name": param["name"],
                    "param_type": "required",
                    "format": ["ND"],
                    "type": [cann_type],
                })
            else:
                # Scalar as attr
                attr_info = {
                    "name": param["name"],
                    "type": param.get("dtype", "float"),
                }
                # Check if optional (has default value)
                if "default" in param and param["default"] is not None:
                    attr_info["param_type"] = "optional"
                    attr_info["default_value"] = str(param["default"])
                else:
                    attr_info["param_type"] = "required"
                attr_desc.append(attr_info)

        output_desc = []
        for out in outputs:
            if out.get("is_tensor", True):
                cann_type = self._dtype_to_cann_json(out.get("dtype", "float"))
                output_desc.append({
                    "name": out["name"],
                    "param_type": "required",
                    "format": ["ND"],
                    "type": [cann_type],
                })

        config = [{
            "op": self.op_custom_capital,
            "language": "cpp",
            "input_desc": input_desc,
            "output_desc": output_desc,
        }]

        # Add attr if any scalar params exist
        if attr_desc:
            config[0]["attr"] = attr_desc

        return json.dumps(config, indent=4)

    def _gen_host_tiling(self, tiling_fields: List[Dict[str, str]]) -> str:
        """Generate host tiling header."""
        fields_code = ""
        for field in tiling_fields:
            fields_code += f"    TILING_DATA_FIELD_DEF({field['type']}, {field['name']});\n"

        return f'''#ifndef {self.op_custom.upper()}_TILING_H
#define {self.op_custom.upper()}_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF({self.op_custom_capital}TilingData)
{fields_code}END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({self.op_custom_capital}, {self.op_custom_capital}TilingData)
}}

#endif // {self.op_custom.upper()}_TILING_H
'''

    def _gen_host_operator(self, block_dim: int, tiling_func_body: str) -> str:
        """Generate host operator implementation."""
        inputs = self.signature.get("inputs", [])
        outputs = self.signature.get("outputs", [])
        init_params = self.signature.get("init_params", [])

        input_defs = ""
        attr_defs = ""

        # Process all inputs
        for inp in inputs:
            if inp.get("is_tensor", True):
                ge_dtype = self._dtype_to_ge_datatype(inp.get("dtype", "float"))
                input_defs += f'        this->Input("{inp["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'
            else:
                # Scalar as attr
                attr_defs += self._gen_attr_def(inp)

        # Process init_params
        for param in init_params:
            if param.get("is_tensor", False):
                ge_dtype = self._dtype_to_ge_datatype(param.get("dtype", "float"))
                input_defs += f'        this->Input("{param["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'
            else:
                # Scalar as attr
                attr_defs += self._gen_attr_def(param)

        output_defs = ""
        for out in outputs:
            if out.get("is_tensor", True):
                ge_dtype = self._dtype_to_ge_datatype(out.get("dtype", "float"))
                output_defs += f'        this->Output("{out["name"]}").ParamType(REQUIRED).DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}});\n'

        return f'''#include "{self.op_custom}_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {{
const uint32_t BLOCK_DIM = {block_dim};

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{{
    {self.op_custom_capital}TilingData tiling;
{tiling_func_body}
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ge {{
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{{
    const ge::DataType x1_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1_dtype);
    return GRAPH_SUCCESS;
}}
}}

namespace ops {{
class {self.op_custom_capital} : public OpDef {{
public:
    explicit {self.op_custom_capital}(const char* name) : OpDef(name)
    {{
{input_defs}{output_defs}{attr_defs}
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }}
}};

OP_ADD({self.op_custom_capital});
}}
'''

    def _gen_python_bind(self) -> str:
        """
        Generate Python binding code.

        Supports both tensor and scalar parameters uniformly.
        """
        inputs = self.signature.get("inputs", [])
        init_params = self.signature.get("init_params", [])

        param_parts = []
        first_tensor = None

        # Process all inputs
        for inp in inputs:
            if inp.get("is_tensor", True):
                param_parts.append(f"const at::Tensor& {inp['name']}")
                if first_tensor is None:
                    first_tensor = inp['name']
            else:
                cpp_type = self._dtype_to_cpp_type(inp.get("dtype", "float"))
                param_parts.append(f"{cpp_type} {inp['name']}")

        # Process init_params
        for param in init_params:
            if param.get("is_tensor", False):
                param_parts.append(f"const at::Tensor& {param['name']}")
                if first_tensor is None:
                    first_tensor = param['name']
            else:
                cpp_type = self._dtype_to_cpp_type(param.get("dtype", "float"))
                param_parts.append(f"{cpp_type} {param['name']}")

        all_params = ", ".join(param_parts)

        # Generate args for EXEC_NPU_CMD (all params + result)
        all_args = [inp["name"] for inp in inputs] + [param["name"] for param in init_params]
        exec_args = ", ".join(all_args + ["result"])

        # Use first tensor for result allocation
        if first_tensor is None:
            first_tensor = "x"

        return f'''#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor {self.op_custom}_impl_npu({all_params}) {{
    at::Tensor result = at::empty_like({first_tensor});
    EXEC_NPU_CMD(aclnn{self.op_custom_capital}, {exec_args});
    return result;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{self.op_custom}", &{self.op_custom}_impl_npu, "{self.op_name} operator");
}}
'''

    def _dtype_to_cpp_type(self, dtype: str) -> str:
        """Convert Python dtype to C++ type (for scalar parameters in pybind)."""
        dtype_map = {
            "float": "float",
            "float32": "float",
            "float16": "float",  # Use float for API, cast internally
            "int": "int64_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "bool": "bool",
        }
        return dtype_map.get(dtype.lower(), "float")

    def _gen_attr_def(self, param: Dict[str, Any]) -> str:
        """Generate CANN attr definition for scalar parameter."""
        name = param["name"]
        dtype = param.get("dtype", "float")
        dtype_lower = dtype.lower()

        # Determine if optional (has default value)
        has_default = "default" in param and param["default"] is not None
        attr_type = "OPTIONAL" if has_default else "REQUIRED"
        default_val = param.get("default")

        if dtype_lower in ("float", "float32", "float16"):
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Float({default_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Float();\n'
        elif dtype_lower in ("int", "int32", "int64"):
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Int({default_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Int();\n'
        elif dtype_lower == "bool":
            bool_val = "true" if default_val else "false"
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Bool({bool_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Bool();\n'
        else:
            if has_default:
                return f'        this->Attr("{name}").AttrType({attr_type}).Float({default_val});\n'
            return f'        this->Attr("{name}").AttrType({attr_type}).Float();\n'

    def _dtype_to_cann_json(self, dtype: str) -> str:
        """Convert Python dtype to CANN JSON type string (for tensor types)."""
        dtype_map = {
            "float": "float",
            "float32": "float",  # CANN uses "float" not "float32"
            "float16": "float16",
            "bfloat16": "bfloat16",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "bool": "bool",
        }
        return dtype_map.get(dtype.lower(), "float")

    def _dtype_to_ge_datatype(self, dtype: str) -> str:
        """Convert Python dtype to ge::DataType enum (for tensor types)."""
        dtype_map = {
            "float": "ge::DT_FLOAT",
            "float32": "ge::DT_FLOAT",
            "float16": "ge::DT_FLOAT16",
            "bfloat16": "ge::DT_BF16",
            "int8": "ge::DT_INT8",
            "int16": "ge::DT_INT16",
            "int32": "ge::DT_INT32",
            "int64": "ge::DT_INT64",
            "uint8": "ge::DT_UINT8",
            "bool": "ge::DT_BOOL",
        }
        return dtype_map.get(dtype.lower(), "ge::DT_FLOAT")

    def _gen_model_src(self) -> str:
        """
        Generate test model code (ModelNew class).

        ModelNew must have the same interface as Model:
        - Same __init__ parameters
        - Same forward parameters
        """
        inputs = self.signature.get("inputs", [])
        init_params = self.signature.get("init_params", [])

        # Generate forward parameters
        forward_params = ", ".join([inp["name"] for inp in inputs])

        # Generate __init__ signature and body
        if init_params:
            init_param_strs = []
            init_body_lines = []
            for param in init_params:
                # Build parameter string with optional default
                param_str = param["name"]
                if "default" in param and param["default"] is not None:
                    default_val = param["default"]
                    if isinstance(default_val, str):
                        param_str += f' = "{default_val}"'
                    else:
                        param_str += f" = {default_val}"
                init_param_strs.append(param_str)
                init_body_lines.append(f"        self.{param['name']} = {param['name']}")

            init_signature = ", ".join(init_param_strs)
            init_body = "\n".join(init_body_lines)
        else:
            init_signature = ""
            init_body = "        pass"

        # Generate custom op call args (all inputs + init_params as self.xxx)
        op_args = [inp["name"] for inp in inputs]
        for param in init_params:
            op_args.append(f"self.{param['name']}")
        op_args_str = ", ".join(op_args)

        return f'''import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, {init_signature}) -> None:
        super().__init__()
{init_body}

    def forward(self, {forward_params}):
        return custom_ops_lib.{self.op_custom}({op_args_str})
'''
