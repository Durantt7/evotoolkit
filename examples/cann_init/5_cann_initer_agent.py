#!/usr/bin/env python3
# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANNIniter Agent E2E 测试

测试用例:
- easy: ReLU (element-wise, 默认 tiling)
- medium: Softmax (reduce + element-wise)
- hard: Scaled Dot-Product Attention (matmul + softmax + matmul)

用法:
    python 5_cann_initer_agent.py [easy|medium|hard]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from _config import get_task, get_llm

from evotoolkit.task.cann_init import CANNIniterInterface
from evotoolkit.evo_method.cann_initer import CANNIniter, CANNIniterConfig

# Test cases directory
TEST_CASES_DIR = Path(__file__).parent / "test_cases"

# Test case configurations
TEST_CASES = {
    "easy": {
        "name": "ReLU",
        "op_name": "Relu",
        "file": "easy_relu.py",
        "description": "Element-wise operation, uses default tiling",
        "max_debug_iterations": 3,
        "max_joint_turns": 2,
    },
    "medium": {
        "name": "Softmax",
        "op_name": "Softmax",
        "file": "medium_softmax.py",
        "description": "Reduce + element-wise, needs shape inference",
        "max_debug_iterations": 5,
        "max_joint_turns": 3,
    },
    "hard": {
        "name": "ScaledDotProductAttention",
        "op_name": "SDPA",
        "file": "hard_sdpa.py",
        "description": "MatMul + Softmax + MatMul, complex tiling",
        "max_debug_iterations": 8,
        "max_joint_turns": 5,
    },
}


def load_python_ref(test_case: str) -> str:
    """Load Python reference code from test case file."""
    config = TEST_CASES[test_case]
    file_path = TEST_CASES_DIR / config["file"]
    return file_path.read_text()


def main(test_case: str = "easy"):
    """Run CANNIniter Agent test."""
    if test_case not in TEST_CASES:
        print(f"Unknown test case: {test_case}")
        print(f"Available: {list(TEST_CASES.keys())}")
        sys.exit(1)

    config_info = TEST_CASES[test_case]
    print(f"\n{'=' * 60}")
    print(f"CANNIniter E2E Test: {config_info['name']}")
    print(f"Description: {config_info['description']}")
    print(f"{'=' * 60}\n")

    # Load Python reference
    python_ref = load_python_ref(test_case)

    # Initialize components
    task = get_task(op_name=config_info["op_name"])
    llm = get_llm()
    interface = CANNIniterInterface()

    # Create config
    config = CANNIniterConfig(
        task=task,
        interface=interface,
        output_path=f"output/e2e_{test_case}_{config_info['op_name'].lower()}/",
        running_llm=llm,
        knowledge_base=None,
        verbose=True,
        max_debug_iterations=config_info["max_debug_iterations"],
        max_joint_turns=config_info["max_joint_turns"],
    )

    # Run
    initer = CANNIniter(config)
    result = initer.run(op_name=config_info["op_name"], python_ref=python_ref)

    # Print result
    print(f"\n{'=' * 60}")
    print(f"Result: {config_info['name']}")
    print(f"{'=' * 60}")
    print(f"  Success: {result['success']}")
    print(f"  Code keys: {list(result['code'].keys())}")
    if result.get("error"):
        print(f"  Error: {result['error']}")

    return result


if __name__ == "__main__":
    # Parse command line argument
    test_case = sys.argv[1] if len(sys.argv) > 1 else "easy"
    main(test_case)
