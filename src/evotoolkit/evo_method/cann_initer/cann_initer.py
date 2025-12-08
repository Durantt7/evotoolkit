# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANNIniter: Ascend C operator auto-generation Agent.

Design principles: Tool-based Retrieval + Specialist collaboration
- Phase 0: Signature parsing (deterministic) + Compute pattern analysis (LLM)
- Parallel branches: Pybind independent || Kernel+Tiling joint (multi-turn)
- Debug Loop: Iterative debugging until correct
"""

# import concurrent.futures  # TODO: Enable for production parallel execution
import os
from pathlib import Path

from .phases import DebugLoop, JointBranch, Phase0Analyzer, PybindBranch
from .run_config import CANNIniterConfig
from .run_state_dict import CANNIniterRunStateDict


class CANNIniter:
    """CANNIniter main workflow."""

    def __init__(self, config: CANNIniterConfig):
        self.config = config
        self.run_state_dict = self._load_or_create_state()

    def _load_or_create_state(self) -> CANNIniterRunStateDict:
        """Load or create run state."""
        state_file = os.path.join(self.config.output_path, "run_state.pkl")
        if os.path.exists(state_file):
            self._verbose("Loading state from pickle...")
            return CANNIniterRunStateDict.from_pickle(state_file)
        return CANNIniterRunStateDict()

    def _save_state(self):
        """Save run state."""
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        state_file = os.path.join(self.config.output_path, "run_state.pkl")
        self.run_state_dict.to_pickle(state_file)

    def _verbose(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(msg)

    def run(self, op_name: str, python_ref: str) -> dict:
        """
        Execute the complete operator generation workflow.

        Args:
            op_name: Operator name
            python_ref: Python reference implementation code

        Returns:
            {"success": bool, "code": dict}
        """
        self._verbose(f"\n{'='*60}")
        self._verbose(f"CANNIniter: {op_name}".center(60))
        self._verbose("=" * 60)

        # Phase 0: Signature parsing + Compute pattern analysis
        self._verbose("\n--- Phase 0: Signature Analysis ---")
        phase0 = Phase0Analyzer(self.config, self.run_state_dict)
        phase0.analyze(op_name, python_ref)
        self._save_state()

        # Unpack strategies for flow control
        strategies = self.run_state_dict.strategies
        tiling_strategy = strategies.get("tiling", "generate")
        pybind_strategy = strategies.get("pybind", "generate")

        self._verbose(f"\nStrategy decisions:")
        self._verbose(f"  - kernel: generate (always)")
        self._verbose(f"  - tiling: {tiling_strategy}")
        self._verbose(f"  - pybind: {pybind_strategy}")

        # Branch processing (sequential for debugging, parallel for production)
        # TODO: Enable parallel execution for production
        # self._verbose("\n--- Parallel Branches: Pybind || Joint ---")
        # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        #     pybind_future = executor.submit(pybind_branch.run)
        #     joint_future = executor.submit(joint_branch.run, python_ref)
        #     pybind_future.result()
        #     joint_future.result()

        # Pybind branch: default template or generate
        self._verbose(f"\n--- Branch: Pybind ({pybind_strategy}) ---")
        pybind_branch = PybindBranch(self.config, self.run_state_dict)
        pybind_branch.run()
        self._save_state()

        # Joint branch: kernel (always generate) + tiling (default or generate)
        self._verbose(f"\n--- Branch: Joint (kernel=generate, tiling={tiling_strategy}) ---")
        joint_branch = JointBranch(self.config, self.run_state_dict)
        joint_branch.run(python_ref)
        self._save_state()

        # Evaluate + Debug Loop
        self._verbose("\n--- Debug Loop ---")
        debug_loop = DebugLoop(self.config, self.run_state_dict)
        result = debug_loop.run(python_ref)

        self.run_state_dict.is_done = True
        self._save_state()

        return result
