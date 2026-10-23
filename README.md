# Onco-RadioMap

**V1.3.2：肿瘤亚型 × 放射性诊疗药物的人体证据评估代码。**

本仓库对应 V1.3.2 归档实验：3 个数据集、4 种方法、5 轮运行，共 35,240 条预测。包内包含规则、提示词、检索、来源归属审核、成熟度处理和评分代码；数据、上下文和模型回执通过显式路径传入。

## 安装与离线检查

使用 Python 3.11 或更高版本。在仓库根目录运行：

```bash
python -m pip install -e ".[test]"
python -m onco_radiomap_agent --version
python -m pytest -q
python examples/offline_demo.py
```

需要模型调用时安装 `.[live]`；需要构建或查询 LightRAG 索引时安装 `.[retrieval]`。依赖版本对应本次打包验证环境。运行时保留 Python 断言检查，使用普通模式，不加 `-O`。

## 目录与保留范围

| 位置 | 内容 |
| --- | --- |
| `src/onco_radiomap_agent/engine.py` | 四方法推理、回执规范化和最终处理的统一入口 |
| `preparation.py`、`task_contract.py` | 冻结分类请求、规则提案和提示词 |
| `arbiter.py` | Full System 的允许改判字段和来源审核 |
| `prediction_facts.py`、`prediction_rules.py` | 无参考答案的事实提取与确定性规则 |
| `naive_rag.py`、`lightrag_*.py` | 检索与两实体 LightRAG 接口 |
| `claim_scope.py`、`claim_context.py`、`claim_renderer.py` | 来源资格、配对归属和证据上下文 |
| `claims.py`、`conclusion_alignment.py` | 有来源约束的结果核验与最终结论汇总 |
| `ownership*.py`、`maturity*.py` | 最终来源归属、研究设计和成熟度处理 |
| `evaluation.py`、`expert_metrics.py`、`repeated_metrics.py` | 自动参考评分、专家一致性和五轮整病例 bootstrap |
| `resources/` | 必需的实体表、开放世界实体补表、预测 schema 和构图提示词 |
| `tests/` | 人工合成输入的离线回归测试 |
| `code_provenance.json` | 源码来源、源文件 SHA-256 和保留范围 |

完整结果复现需要配套的冻结数据、检索上下文和模型回执；这些文件独立于代码包保存。

`prediction_rules` 生成初始规则提案，`arbiter` 使用该提案核验允许的改判。有效配对的最终结论由 `conclusion_alignment.aggregate_claims` 汇总：标记 `conflicting` 需要独立研究在可比较的终点、方案和临床场景下出现相反结果。`claim_scope` 仅筛选来源资格与配对归属。提示词版本标识保留归档值，以便核对请求身份。

## 运行自己的输入

复制并填写 `examples/bundle.json`。该文件是路径模板，其 `inputs/` 文件需要由使用者提供。每次调用处理一个数据集；病例池应包含该数据集的完整原子证据记录。

| 字段 | 文件内容 |
| --- | --- |
| `cases` | 无答案字段的预测输入 JSONL：`atomic_source_record` 或 `pair_snapshot` |
| `rule_facts` | 分类规则使用的事实 JSONL，含 `origin_dataset_id` 和 `origin_case_id` |
| `naive_contexts`、`lightrag_contexts` | JSONL，每行包含 `case_id` 和 `context` |
| `claim_facts` | 最终结论审核使用的事实 JSONL，额外保留 `raw_source_text` |
| `claim_contexts` | 最终结论阶段的来源排序上下文，格式同上 |

`baseline_data.load_prediction_inputs`、`prediction_facts.build_prediction_facts`、`BM25EvidenceIndex` 和 `FrozenLightRAGRetriever` 提供输入准备与检索接口。检索输入仍须遵守数据集、日期、用途、人群和研究家族隔离。

准备分类请求，不调用 API：

```bash
python -m onco_radiomap_agent prepare --bundle examples/bundle.json --output outputs/requests.json
```

使用已有回执离线运行：

```bash
python -m onco_radiomap_agent predict --bundle examples/bundle.json --receipts receipts/run1 --run-id 1 --output outputs/predictions.jsonl
```

回执按 `call_id.json` 命名，必须包含相同的 `identity`、对应 `run_id`、`finish_reason: "stop"`、`response` 和 `provider_model`。旧版单轮回执可省略 `run_id`，此时仅按第 1 轮处理。`call_id` 同时包含重复轮号与请求身份哈希；缺失、跨轮或不匹配的回执会中止。Full System 的结论审核请求在分类完成后，按实际有效配对生成。

显式开启新模型调用：

```bash
python -m pip install -e ".[live]"
# 在当前终端设置 DEEPSEEK_API_KEY；参见 .env.example。
python -m onco_radiomap_agent predict --bundle examples/bundle.json --live --max-calls 20 --save-receipts receipts/new-run --output outputs/new-predictions.jsonl
```

`--max-calls` 是调用次数上限，不是费用上限。每个请求只尝试一次，回执写入新目录。历史模型别名为 `deepseek-flash`；今天的同名 API 不保证等同于历史模型，因此新调用不构成历史结果的逐值复现。

评分单独读取参考答案：

```bash
python -m onco_radiomap_agent evaluate --cases data/reference_cases.jsonl --predictions outputs/predictions.jsonl --roles data/dataset_roles.json --output outputs/evaluation.json
```

推理入口拒绝 `reference_*`、`gold_*`、`expert_*`、`final_*` 答案字段，研究设计字段 `gold_standard_comparison` 除外。关系任务只评原子记录，成熟度只评配对快照；有效、弃答和无效输出分开报告。

## 验证

`release_verification.json` 记录离线测试、冻结回执回放和本地打包检查的结果与范围。源码来源及整理时的修改记录见 `code_provenance.json`。

仓库文本统一使用 LF 换行。解压后可在仓库根目录运行 `sha256sum -c SHA256SUMS` 校验文件完整性。

代码使用 Apache-2.0 许可证；外部文献与数据按各自来源许可使用。
