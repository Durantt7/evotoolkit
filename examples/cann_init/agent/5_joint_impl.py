#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Joint Branch Phase 3 独立测试 - 代码实现

单独测试三阶段代码生成 (跳过多轮对话):
1. tiling.h (tiling data structure)
2. op_host.cpp (tiling calculation + InferShape)
3. op_kernel.cpp (kernel implementation)

输入: joint_plan, knowledge_context (来自 4_joint_planning.py)
输出: tiling_src, operator_src, kernel_src

用法:
    python 5_joint_impl.py [easy|medium|hard]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from _config import (
    get_llm, get_test_config, load_python_ref, ensure_output_dir,
    get_phase0_context, get_joint_plan_context, get_knowledge_context
)

from evotoolkit.task.cann_init import CANNIniterInterface, CANNInitTask
from evotoolkit.evo_method.cann_initer import CANNIniterConfig
from evotoolkit.evo_method.cann_initer.parsers import parse_code


def main(test_case: str = "hard"):
    config_info = get_test_config(test_case)
    python_ref = load_python_ref(test_case)

    print("=" * 70)
    print(f"Joint Branch Phase 3 Test - {config_info['name']} (Implementation Only)")
    print("=" * 70)

    # Get contexts
    try:
        phase0_ctx = get_phase0_context(test_case)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    try:
        joint_plan = get_joint_plan_context(test_case)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    knowledge = get_knowledge_context(test_case)
    op_name = phase0_ctx["op_name"]
    tiling_strategy = joint_plan.get("tiling_strategy", "default")

    # Initialize
    print("\n[1] Initializing components...")
    print(f"    Tiling strategy: {tiling_strategy}")
    print(f"    Tiling fields: {joint_plan.get('tiling_fields', [])}")

    task = CANNInitTask(data={
        "op_name": op_name,
        "npu_type": "Ascend910B2",
        "python_reference": python_ref,
    })
    llm = get_llm()
    interface = CANNIniterInterface()

    # Prepare context for prompts
    context = {
        "op_name": op_name,
        "signature": phase0_ctx["signature"],
        "python_ref": python_ref,
    }

    tiling_header = None
    operator_src = None

    # Stage 1 & 2: tiling.h + op_host.cpp (if custom tiling)
    if tiling_strategy != "default":
        print("\n[2] Stage 1: Generating tiling.h...")
        print("-" * 70)
        tiling_header_prompt = interface.get_tiling_header_prompt(joint_plan, context)
        print(f"    Prompt length: {len(tiling_header_prompt)} chars")

        response, _ = llm.get_response(tiling_header_prompt)
        tiling_header = parse_code(response)
        print("\n--- tiling.h ---")
        print(tiling_header)

        print("\n[3] Stage 2: Generating op_host.cpp...")
        print("-" * 70)
        tiling_host_prompt = interface.get_tiling_host_prompt(
            joint_plan, context, knowledge, tiling_header
        )
        print(f"    Prompt length: {len(tiling_host_prompt)} chars")

        response, _ = llm.get_response(tiling_host_prompt)
        operator_src = parse_code(response)
        print("\n--- op_host.cpp (first 1000 chars) ---")
        print(operator_src[:1000] + "..." if len(operator_src) > 1000 else operator_src)
    else:
        print("\n[2-3] Using default tiling, skipping tiling.h and op_host.cpp")

    # Stage 3: op_kernel.cpp
    print("\n[4] Stage 3: Generating op_kernel.cpp...")
    print("-" * 70)
    kernel_prompt = interface.get_kernel_impl_prompt(
        joint_plan, context, knowledge, tiling_header
    )
    print(f"    Prompt length: {len(kernel_prompt)} chars")

    response, _ = llm.get_response(kernel_prompt)
    kernel_src = parse_code(response)
    print("\n--- op_kernel.cpp (first 1500 chars) ---")
    print(kernel_src[:1500] + "..." if len(kernel_src) > 1500 else kernel_src)

    # Save outputs
    output_dir = ensure_output_dir(f"impl_{test_case}")
    if tiling_header:
        (output_dir / "tiling.h").write_text(tiling_header)
    if operator_src:
        (output_dir / "op_host.cpp").write_text(operator_src)
    (output_dir / "op_kernel.cpp").write_text(kernel_src)

    print("\n" + "=" * 70)
    print(f"Results saved to {output_dir}/")
    print("=" * 70)
    print(f"  - tiling.h: {'Yes' if tiling_header else 'No (default tiling)'}")
    print(f"  - op_host.cpp: {'Yes' if operator_src else 'No (default tiling)'}")
    print(f"  - op_kernel.cpp: Yes ({len(kernel_src)} chars)")


if __name__ == "__main__":
    test_case = sys.argv[1] if len(sys.argv) > 1 else "hard"
    main(test_case)
