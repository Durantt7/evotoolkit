# CANNInit 测试

## Evaluator 测试

```bash
cd /root/Huawei_CANN/evotoolkit/examples/cann_init/evaluator
python 1_signature_parser.py
python 2_template_generator.py
python 3_basic_evaluation.py
python 4_parallel_compile.py
```

## Agent 测试

### 测试用例说明

| 难度 | Op | 计算模式 | 输入形状 | 说明 |
|------|-----|---------|---------|------|
| easy | ReLU | element-wise | `[16, 16384]` | 最简单，用 default tiling |
| medium | Softmax | reduction | TBD | 需要 custom tiling |
| hard | SDPA | matmul+softmax | `[B, S, D]` | 复杂，需要 Gemm 等高级 API |

### Easy 测试 (ReLU)

```bash
cd /root/Huawei_CANN/evotoolkit/examples/cann_init/agent

# 分步执行
python 2_phase0.py easy           # Phase 0: 签名分析
python 3_pybind.py easy           # Pybind: 绑定代码
python 4_joint_planning.py easy   # Planning: 多轮对话
python 5_joint_impl.py easy       # Impl: 代码生成
python 7_evaluate.py easy         # Eval: 编译验证

# 或 E2E 一步到位
python 6_e2e_test.py easy
```

### Hard 测试 (SDPA)

```bash
cd /root/Huawei_CANN/evotoolkit/examples/cann_init/agent

# 分步执行
python 2_phase0.py hard           # Phase 0: 签名分析
# → 输出填入 _config.py PHASE0_CONTEXT

python 3_pybind.py hard           # Pybind: 绑定代码

python 4_joint_planning.py hard   # Planning: 多轮对话 + 知识检索
# → 输出填入 _config.py JOINT_PLAN_CONTEXT

python 5_joint_impl.py hard       # Impl: 代码生成
# → 输出: impl_hard/tiling.h, op_host.cpp, op_kernel.cpp

python 7_evaluate.py hard         # Eval: 编译验证

# 或 E2E 一步到位
python 6_e2e_test.py hard
```

## 单独测试

```bash
# 知识检索单独测试
python 1_knowledge_retrieval.py
```

