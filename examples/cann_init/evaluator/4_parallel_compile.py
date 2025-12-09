# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Parallel Compilation

Demonstrates parallel compilation of multiple kernel variants,
then sequential testing on NPU.

Strategy:
- msopgen: Sequential (has global resource conflicts)
- build.sh: Parallel with staggered start (CMake race condition workaround)
- test: Sequential (NPU access)

Usage:
    python 4_parallel_compile.py
    python 4_parallel_compile.py --npu Ascend910B
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution
from _config import KERNEL_SRC, get_task_data, ensure_output_dir


def build_with_delay(task, sol, delay_seconds):
    """Run build after a delay to avoid CMake race conditions."""
    time.sleep(delay_seconds)
    return task.evaluate_solution(sol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", default="Ascend910B")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between builds (seconds)")
    args = parser.parse_args()

    print("=" * 50)
    print("Parallel Compilation Test")
    print(f"Build stagger delay: {args.delay}s")
    print("=" * 50)

    output_dir = ensure_output_dir("4_parallel")

    task = CANNInitTask(
        data=get_task_data(npu_type=args.npu),
        fake_mode=False,
    )

    # Create solutions with different block_dim
    block_dims = [4, 8, 16, 32]
    solutions = []
    for i, block_dim in enumerate(block_dims):
        config = CANNSolutionConfig(
            project_path=str(output_dir / f"sol_{i:03d}"),
            block_dim=block_dim,
            compile_only=True,
            save_compile_to=str(output_dir / f"sol_{i:03d}"),
        )
        solutions.append((i, block_dim, Solution(KERNEL_SRC, config.to_dict())))

    # Phase 1: Sequential msopgen + source writing
    print(f"\nPhase 1: Sequential project setup ({len(solutions)} solutions)...")
    setup_results = []
    for idx, bd, sol in solutions:
        # Use setup_only mode to just create project and write files
        config = CANNSolutionConfig.from_dict(sol.other_info)
        config_dict = sol.other_info.copy()
        config_dict["setup_only"] = True  # New flag: only msopgen + write files
        setup_sol = Solution(sol.sol_string, config_dict)

        result = task.evaluate_solution(setup_sol)
        status = "ready" if result.valid else "failed"
        print(f"  sol_{idx} (block_dim={bd}): {status}")
        setup_results.append((idx, bd, result))

    # Filter successful setups
    ready_solutions = [(idx, bd, r) for idx, bd, r in setup_results if r.valid]
    if not ready_solutions:
        print("\nAll setups failed. Exiting.")
        return

    # Phase 2: Parallel build with staggered start
    print(f"\nPhase 2: Parallel build with {args.delay}s stagger ({len(ready_solutions)} solutions)...")
    build_results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, (idx, bd, _) in enumerate(ready_solutions):
            config_dict = {
                "project_path": str(output_dir / f"sol_{idx:03d}"),
                "build_only": True,  # New flag: only run build.sh
                "save_compile_to": str(output_dir / f"sol_{idx:03d}"),
            }
            build_sol = Solution(KERNEL_SRC, config_dict)
            delay = i * args.delay  # Stagger: 0s, 2s, 4s, 6s
            future = executor.submit(build_with_delay, task, build_sol, delay)
            futures[future] = (idx, bd)

        for future in as_completed(futures):
            idx, bd = futures[future]
            result = future.result()
            status = "built" if result.valid else "failed"
            err_info = ""
            if not result.valid:
                err = result.additional_info.get('error', '')
                # Show more of the error
                err_info = f"\n    Error: {err[:500]}"
            print(f"  sol_{idx} (block_dim={bd}): {status}{err_info}")
            build_results.append((idx, bd, result))

    # Summary
    built_count = sum(1 for _, _, r in build_results if r.valid)
    print(f"\nBuild: {built_count}/{len(ready_solutions)} succeeded")

    # Phase 3: Sequential testing
    print("\nPhase 3: Sequential testing...")
    for idx, bd, build_result in sorted(build_results):
        if not build_result.valid:
            print(f"  sol_{idx} (block_dim={bd}): skipped (build failed)")
            continue

        test_result = task.test_compiled(
            load_from=str(output_dir / f"sol_{idx:03d}")
        )
        if test_result.valid:
            rt = test_result.additional_info.get("runtime")
            print(f"  sol_{idx} (block_dim={bd}): {rt:.4f} ms")
        else:
            err = test_result.additional_info.get("error", "unknown")
            print(f"  sol_{idx} (block_dim={bd}): failed - {err[:50]}")


if __name__ == "__main__":
    main()
