# 第五步处理消融：预注册与离线 Test

脚本入口为 `python -m scripts.agent_ablation`。在隔离的后端 storage/队列和同版本 Web、worker 上执行；认证仅通过运行环境注入。先检查 10 个上传数据集的内容 SHA-256、模型参数、LLM 模型及 tokenizer，然后生成预注册 JSON。`register-processing` 会在任何决策前冻结 40 行及其摘要，重复注册同一 ID 会拒绝。

预注册 JSON 顶层包含 `tasks`（10 个 `{task_id,dataset_id,dataset_sha256}`）、`model_pools`（`ML` 与 `DL`，合起来恰好覆盖 13 个可执行模型）、`fixed_processing`（每模型一组合法 normalization/class_balance）、`seed: 42`、`repeat` 和 `conditions`。`conditions` 必须是 `backend_binding`、`scope_binding`、`llm_binding`、`model_configs_digest` 四个 SHA-256 摘要。摘要分别使用脚本的 `digest`、`LLMConfig.fingerprint()` 与冻结配置计算，不能填明文 token。数据集 ID 需属于实际运行的隔离后端。

```text
python -m scripts.agent_ablation register-processing --config work/step5/registration.json --storage work/step5/records --experiment-id step5-processing
```

按 `plan.json` 中的 `rows` 顺序逐行调用原有单行入口，传入该行的 `--experiment-id`、`--dataset-id`、`--allowed-models`、`--processing-mode`，并统一传入 `--kind agent --decision-mode recipe_id --knowledge off --seed 42 --defer-test --plan <plan.json> --fixed-processing <fixed.json>`。其余后端、scope、LLM、模型参数及预算参数须与注册摘要一致。脚本会拒绝不匹配的行，保留 attempt、源码绑定、checkpoint 与不确定提交记录；失败行不复写成新样本。

全部 40 行决策终止后，才运行：

```text
python -m scripts.agent_ablation collect-processing --plan work/step5/records/step5-processing/plan.json --storage work/step5/records --backend-url http://127.0.0.1:8000
```

收集命令会先核对后端 URL 与预注册摘要，再逐行检查记录和持久 checkpoint。只有全部行处于已关闭的终态，才会发出读取 Test 的请求；等待、初始化、恢复、待人工处理中的 checkpoint 均会阻止收集。

离线汇总写入计划旁的 `summary.json`。每对动态/固定 Run 必须使用注册数据集、不同 Run、相同分割及相同模型配置，否则记录排除原因，不计入 `paired_count`、配对均值或对应组的 `complete_mean`。缺失/失败 Test 保持空值；成功子集均值明确带分母。`paired_complete_mean_difference` 仅在全部配对有效时出现。单 seed 的配对差值仅作描述，不作统计显著性结论。

本地受控组件验收可执行 `python -m scripts.verify_processing_matrix`，逐一拟合有限目录中的 188 个合法处理组合，结果保存在 `work/step5_local_acceptance/processing-matrix.json`。该组件验证不产生正式消融 Run 或 Test 指标。
