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

```bash
cd /root/Huawei_CANN/evotoolkit/examples/cann_init/agent

# 1. Phase 0 分析器
python 2_phase0.py easy
python 2_phase0.py medium
python 2_phase0.py hard
# → 输出填入 _config.py PHASE0_CONTEXT

# 2. Pybind Branch
python 3_pybind.py hard

# 3. Joint Branch 完整流程
python 4_joint_planning.py hard
# → 输出填入 _config.py JOINT_PLAN_CONTEXT

# 4. 三阶段代码实现 (单独测试)
python 5_joint_impl.py hard

# 5. 知识检索 (独立测试)
python 1_knowledge_retrieval.py

# 6. E2E 完整测试
python 6_e2e_test.py easy
python 6_e2e_test.py medium
python 6_e2e_test.py hard
```
