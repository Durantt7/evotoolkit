# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANNIniter: Ascend C 算子自动生成 Agent

基于 Tool-based Retrieval + 专员分工协作的设计理念：
- Phase 0: 签名解析（确定性）+ 计算模式识别（LLM）
- 并行分支: Pybind 独立 || Kernel+Tiling 联合（多轮对话）
- Debug Loop: 迭代调试直到正确
"""

import concurrent.futures
import json
import os
import re
from pathlib import Path

from evotoolkit.tools.llm import HttpsApi

from .run_config import CANNIniterConfig
from .run_state_dict import CANNIniterRunStateDict


class CANNIniter:
    """CANNIniter 主流程"""

    def __init__(self, config: CANNIniterConfig):
        self.config = config
        self.run_state_dict = self._load_or_create_state()

    def _load_or_create_state(self) -> CANNIniterRunStateDict:
        """加载或创建状态"""
        state_file = os.path.join(self.config.output_path, "run_state.pkl")
        if os.path.exists(state_file):
            self._verbose("Loading state from pickle...")
            return CANNIniterRunStateDict.from_pickle(state_file)
        return CANNIniterRunStateDict()

    def _save_state(self):
        """保存状态"""
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        state_file = os.path.join(self.config.output_path, "run_state.pkl")
        self.run_state_dict.to_pickle(state_file)

    def _verbose(self, msg: str):
        """输出信息"""
        if self.config.verbose:
            print(msg)

    def run(self, op_name: str, python_ref: str) -> dict:
        """
        执行完整的算子生成流程

        Args:
            op_name: 算子名称
            python_ref: Python 参考实现代码

        Returns:
            {"success": bool, "code": dict}
        """
        self._verbose(f"\n{'='*60}")
        self._verbose(f"CANNIniter: {op_name}".center(60))
        self._verbose("=" * 60)

        # Phase 0: 签名解析 + 计算模式识别
        self._verbose("\n--- Phase 0: Signature Analysis ---")
        self._phase0_analyze(op_name, python_ref)
        self._save_state()

        # 并行分支处理
        self._verbose("\n--- Parallel Branches: Pybind || Joint ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pybind_future = executor.submit(self._pybind_branch)
            joint_future = executor.submit(self._joint_branch, python_ref)
            pybind_future.result()
            joint_future.result()
        self._save_state()

        # Evaluate + Debug Loop
        self._verbose("\n--- Debug Loop ---")
        result = self._debug_loop(python_ref)

        self.run_state_dict.is_done = True
        self._save_state()

        return result

    # ==================== Phase 0 ====================

    def _phase0_analyze(self, op_name: str, python_ref: str):
        """Phase 0: 签名解析（确定性）+ 计算模式识别（LLM）"""
        # 1. 签名解析（复用 Evaluator 的 parser）
        self._verbose("Parsing signature...")
        self.run_state_dict.signature = self.config.task.parser.parse(python_ref, op_name)

        # 2. 计算模式识别（LLM）
        self._verbose("Analyzing compute pattern with LLM...")
        prompt = self.config.interface.get_pattern_analysis_prompt(
            python_ref, self.run_state_dict.signature
        )
        response, _ = self.config.running_llm.get_response(prompt)
        result = self._parse_json(response)

        self.run_state_dict.compute_pattern = result.get("compute_pattern", "other")
        self.run_state_dict.strategies = result.get("strategies", {
            "kernel": "generate",
            "tiling": "generate",
            "pybind": "generate"
        })

        self._verbose(f"Compute pattern: {self.run_state_dict.compute_pattern}")
        self._verbose(f"Strategies: {self.run_state_dict.strategies}")

    # ==================== Pybind 独立分支 ====================

    def _pybind_branch(self):
        """Pybind 独立分支（简单上下文，可独立并行）"""
        self._verbose("[Pybind] Starting...")

        if self.run_state_dict.strategies.get("pybind") == "default":
            self._verbose("[Pybind] Using default template")
            self.run_state_dict.pybind_src = None
        else:
            self._verbose("[Pybind] Generating with LLM...")
            prompt = self.config.interface.get_pybind_prompt(
                self.run_state_dict.signature
            )
            response, _ = self.config.running_llm.get_response(prompt)
            self.run_state_dict.pybind_src = self._parse_code(response)

        self._verbose("[Pybind] Done")

    # ==================== Kernel + Tiling 联合分支 ====================

    def _joint_branch(self, python_ref: str):
        """Kernel + Tiling 联合分支（多轮对话达成共识）"""
        self._verbose("[Joint] Starting...")

        # Phase 1: 联合规划讨论
        self._verbose("[Joint] Phase 1: Joint Planning...")
        self._joint_planning(python_ref)

        # Phase 2: 知识检索
        self._verbose("[Joint] Phase 2: Knowledge Retrieval...")
        self._retrieve_knowledge()

        # Phase 3: 代码实现
        self._verbose("[Joint] Phase 3: Code Implementation...")
        self._implement_code(python_ref)

        self._verbose("[Joint] Done")

    def _joint_planning(self, python_ref: str):
        """多轮对话达成共识"""
        context = {
            "signature": self.run_state_dict.signature,
            "compute_pattern": self.run_state_dict.compute_pattern,
            "python_ref": python_ref
        }
        conversation = []

        for turn in range(self.config.max_joint_turns):
            self._verbose(f"[Joint] Turn {turn + 1}")

            # Tiling 专员提出策略
            tiling_prompt = self.config.interface.get_tiling_propose_prompt(context, conversation)
            tiling_msg, _ = self.config.running_llm.get_response(tiling_prompt)
            conversation.append({"role": "tiling", "content": tiling_msg})

            # Kernel 专员评审
            kernel_prompt = self.config.interface.get_kernel_review_prompt(context, conversation)
            kernel_msg, _ = self.config.running_llm.get_response(kernel_prompt)
            conversation.append({"role": "kernel", "content": kernel_msg})

            # 检查是否达成共识
            if self._check_consensus(kernel_msg):
                self._verbose("[Joint] Consensus reached")
                break

        self.run_state_dict.joint_conversation = conversation
        self.run_state_dict.joint_plan = self._extract_joint_plan(conversation)

    def _check_consensus(self, kernel_msg: str) -> bool:
        """检查 Kernel 专员是否接受方案"""
        return "accepted" in kernel_msg.lower() or "agree" in kernel_msg.lower()

    def _extract_joint_plan(self, conversation: list) -> dict:
        """从对话中提取联合规划"""
        # TODO: 实现从对话中提取结构化规划
        return {
            "conversation": conversation,
            "retrieval_requests": []
        }

    def _retrieve_knowledge(self):
        """根据规划检索知识"""
        if not self.config.knowledge_base:
            self.run_state_dict.knowledge = {}
            return

        knowledge = {}
        retrieval_requests = self.run_state_dict.joint_plan.get("retrieval_requests", [])

        for req in retrieval_requests:
            if req.get("type") == "api":
                api_name = req.get("name")
                knowledge[f"api_{api_name}"] = self.config.knowledge_base.search_api(api_name)
            elif req.get("type") == "example":
                op_name = req.get("name")
                knowledge[f"example_{op_name}"] = self.config.knowledge_base.search_operator(op_name)

        self.run_state_dict.knowledge = knowledge

    def _implement_code(self, python_ref: str):
        """代码实现"""
        # Kernel 必须生成
        kernel_prompt = self.config.interface.get_kernel_impl_prompt(
            self.run_state_dict.joint_plan,
            self.run_state_dict.knowledge,
            python_ref
        )
        response, _ = self.config.running_llm.get_response(kernel_prompt)
        self.run_state_dict.kernel_src = self._parse_code(response)

        # Tiling 根据策略决定
        if self.run_state_dict.strategies.get("tiling") == "default":
            self.run_state_dict.tiling_src = None
            self.run_state_dict.operator_src = None
        else:
            tiling_prompt = self.config.interface.get_tiling_impl_prompt(
                self.run_state_dict.joint_plan,
                self.run_state_dict.knowledge
            )
            response, _ = self.config.running_llm.get_response(tiling_prompt)
            result = self._parse_json(response)
            self.run_state_dict.tiling_src = result.get("host_tiling_src")
            self.run_state_dict.operator_src = result.get("host_operator_src")

    # ==================== Debug Loop ====================

    def _debug_loop(self, python_ref: str) -> dict:
        """迭代调试循环"""
        for iteration in range(self.config.max_debug_iterations):
            self.run_state_dict.current_iteration = iteration
            self._verbose(f"[Debug] Iteration {iteration + 1}/{self.config.max_debug_iterations}")

            # 组装代码
            full_code = self._assemble_code()

            # 评估
            result = self.config.task.evaluate_code(full_code)

            if result.valid:
                self._verbose("[Debug] SUCCESS!")
                self.run_state_dict.success = True
                return {"success": True, "code": full_code}

            # 记录错误
            error_info = result.additional_info or {}
            self._verbose(f"[Debug] Error: {error_info.get('stage', 'unknown')}")

            # 错误分类 + 分派修复
            error_type = self._classify_error(error_info)
            self._dispatch_fix(error_type, error_info)

            self.run_state_dict.debug_history.append({
                "iteration": iteration,
                "error_type": error_type,
                "error": error_info
            })
            self._save_state()

        self._verbose("[Debug] Max iterations reached")
        return {"success": False, "code": self._assemble_code()}

    def _assemble_code(self) -> dict:
        """组装完整代码"""
        return {
            "kernel_src": self.run_state_dict.kernel_src,
            "host_tiling_src": self.run_state_dict.tiling_src,
            "host_operator_src": self.run_state_dict.operator_src,
            "python_bind_src": self.run_state_dict.pybind_src,
        }

    def _classify_error(self, error_info: dict) -> str:
        """错误分类"""
        stage = error_info.get("stage", "")
        error_msg = str(error_info.get("error", ""))

        if stage == "compile":
            if "kernel" in error_msg.lower() or ".cpp" in error_msg:
                return "kernel"
            elif "tiling" in error_msg.lower() or "host" in error_msg.lower():
                return "tiling"
            elif "pybind" in error_msg.lower() or "bind" in error_msg.lower():
                return "pybind"
        elif stage == "correctness":
            return "kernel"
        elif stage == "deploy":
            return "pybind"

        return "unknown"

    def _dispatch_fix(self, error_type: str, error_info: dict):
        """分派给对应专员修复"""
        self._verbose(f"[Debug] Dispatching to {error_type} agent...")

        if error_type == "kernel":
            prompt = self.config.interface.get_debug_prompt(
                "kernel", self.run_state_dict.kernel_src, error_info
            )
            response, _ = self.config.running_llm.get_response(prompt)
            self.run_state_dict.kernel_src = self._parse_code(response)

        elif error_type == "tiling":
            prompt = self.config.interface.get_debug_prompt(
                "tiling",
                {"tiling": self.run_state_dict.tiling_src, "operator": self.run_state_dict.operator_src},
                error_info
            )
            response, _ = self.config.running_llm.get_response(prompt)
            result = self._parse_json(response)
            self.run_state_dict.tiling_src = result.get("host_tiling_src")
            self.run_state_dict.operator_src = result.get("host_operator_src")

        elif error_type == "pybind":
            prompt = self.config.interface.get_debug_prompt(
                "pybind", self.run_state_dict.pybind_src, error_info
            )
            response, _ = self.config.running_llm.get_response(prompt)
            self.run_state_dict.pybind_src = self._parse_code(response)

        else:
            # unknown: 尝试修复 kernel
            self._verbose("[Debug] Unknown error, attempting kernel fix...")
            prompt = self.config.interface.get_debug_prompt(
                "kernel", self.run_state_dict.kernel_src, error_info
            )
            response, _ = self.config.running_llm.get_response(prompt)
            self.run_state_dict.kernel_src = self._parse_code(response)

    # ==================== 辅助方法 ====================

    def _parse_json(self, response: str) -> dict:
        """从 LLM 响应中解析 JSON"""
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    def _parse_code(self, response: str) -> str:
        """从 LLM 响应中解析代码"""
        code_match = re.search(r"```(?:cpp|c\+\+|python)?\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response.strip()
