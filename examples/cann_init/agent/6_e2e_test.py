#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANNIniter Agent E2E 完整测试

测试完整的 Agent 流程:
- Phase 0: Signature Analysis
- Pybind Branch: pybind code generation
- Joint Branch: kernel + tiling code generation
  - Phase 1: Multi-turn dialogue
  - Phase 2: Knowledge retrieval
  - Phase 3: Code implementation

测试用例:
- easy: ReLU (element-wise, 默认 tiling)
- medium: Softmax (reduce + element-wise)
- hard: Scaled Dot-Product Attention (matmul + softmax + matmul)

用法:
    python 6_e2e_test.py [easy|medium|hard]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm, get_knowledge_base, get_test_config, load_python_ref,
    ensure_output_dir
)

from evotoolkit.task.cann_init import CANNIniterInterface, CANNInitTask
from evotoolkit.evo_method.cann_initer import CANNIniter, CANNIniterConfig


def main(test_case: str = "easy"):
    config_info = get_test_config(test_case)
    python_ref = load_python_ref(test_case)
    op_name = config_info["op_name"]

    print(f"\n{'=' * 60}")
    print(f"CANNIniter E2E Test: {config_info['name']}")
    print(f"Description: {config_info['description']}")
    print(f"{'=' * 60}\n")

    # Initialize components
    task = CANNInitTask(data={
        "op_name": op_name,
        "npu_type": "Ascend910B2",
        "python_reference": python_ref,
    })
    llm = get_llm()
    interface = CANNIniterInterface()
    kb = get_knowledge_base()
    print(f"Knowledge Base: {kb.get_api_count()} APIs, {kb.get_operator_count()} operators")

    # Create config
    config = CANNIniterConfig(
        task=task,
        interface=interface,
        output_path=str(ensure_output_dir(f"e2e_{test_case}_{op_name.lower()}")),
        running_llm=llm,
        knowledge_base=kb,
        verbose=True,
        max_debug_iterations=config_info["max_debug_iterations"],
        max_joint_turns=config_info["max_joint_turns"],
    )

    # Run
    initer = CANNIniter(config)
    result = initer.run(op_name=op_name, python_ref=python_ref)

    # Print result
    print(f"\n{'=' * 60}")
    print(f"Result: {config_info['name']}")
    print(f"{'=' * 60}")
    print(f"  Success: {result['success']}")
    print(f"  Code keys: {list(result['code'].keys())}")

    # Show generated code sizes
    for key, code in result['code'].items():
        if code:
            print(f"  - {key}: {len(code)} chars")

    # Save outputs
    output_dir = ensure_output_dir(f"e2e_{test_case}_{op_name.lower()}")
    for key, code in result['code'].items():
        if code:
            suffix = ".cpp" if key != "host_tiling_src" else ".h"
            filename = f"{key}{suffix}"
            (output_dir / filename).write_text(code)
    print(f"\n[Saved to {output_dir}/]")


if __name__ == "__main__":
    test_case = sys.argv[1] if len(sys.argv) > 1 else "easy"
    main(test_case)
