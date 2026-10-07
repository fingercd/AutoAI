# 接口与命令

依赖 Python 3.10+、httpx、openpyxl、swanlab。后端必须具有 `/api/training/preflight`、strict_config 和 Idempotency-Key 支持。服务地址默认 `http://127.0.0.1:8000`，可由 `AUTOAI_SERVICE_URL` 或 `--service-url` 指定；鉴权只读取环境变量 `AUTOAI_SERVICE_TOKEN`。不要把令牌传到命令行。

```text
python <skill>/scripts/autoai_client.py health
python <skill>/scripts/autoai_client.py capabilities
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment upload prepared/dataset.csv
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment upload external/dataset.csv --role external_test
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment preflight --plan plan.json
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment submit
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment watch --seconds 60
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment status
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment result --download
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment stop
python <skill>/scripts/autoai_client.py --task-dir outputs/my-experiment sync-cloud
```

全局参数放在子命令前。preflight 从任务记录填入已上传的主/测试 Dataset ID，也允许计划显式引用已有 Dataset。

单模型计划示例（模型必须根据数据建议或用户选择替换）：

```json
{
  "task_type": "run",
  "config": {
    "model_type": "logistic_regression",
    "experiment_version": "word-0904",
    "training_profile": "quick",
    "feature_scheme": "full",
    "normalization": "zscore",
    "split_mode": "stratified_holdout",
    "split_train": 8,
    "split_valid": 1,
    "split_test": 1,
    "seed": 42
  }
}
```

Batch 使用 `task_type=batch`、顶层 `model_types` 和 `base_seed`，config 删除 model_type/seed/split_seed/model_seed；后端固定每模型一个 Run。

## 原始曲线

```text
python <skill>/scripts/autoai_client.py inspect-hplc curve-A.txt curve-B.txt
python <skill>/scripts/autoai_client.py --task-dir outputs/raw-experiment preprocess --kind hplc --options preprocessing.json curve-A.txt curve-B.txt
```

options 是已确认参数的 JSON，例如 `{"range_mode":"row","start_row":1,"hplc_interpolate":true}`。HPLC 不固定点数，范围依检测结果；不消负或归一化。拉曼使用 `--kind raman`，先截取后基线，baseline_method 默认 arPLS。处理完成后将 preprocessed.csv 通过 prepare_dataset 的 Name 映射补齐元数据，再上传。

## 接口行为与恢复

- `POST /api/training/preflight` 返回 training-preflight-v1、runnable、errors、warnings、normalized_configs、data、worker、run_count、feature_scheme_counts 和 submit_payload。失败不入队，也不拟合。
- Run/Batch 请求增加 `strict_config=true`，未知/不生效参数在创建前拒绝；旧接口默认兼容。
- 客户端把 preflight 的提交请求与随机键先写 state.json，再通过 Idempotency-Key 提交。服务端按 Principal 和任务类型去重，同键不同请求或已删除目标返回 409。
- 首次 submit 会刷新预检并检查 Worker；已有未知响应的提交只重放已保存请求，不重做规划。已有 ID 直接查状态。
- GET 和带幂等键的训练创建可有限重试；上传、预处理、停止不盲目自动重试。上传/处理响应未知时保留状态，核实后才使用 `--retry-uncertain`。
- 同一任务目录绑定服务地址。新配置、新训练使用新目录；同目录再次 submit 用于恢复，不用于重新训练。
- 结果保存 run-result-v1、比较投影、summary.json 和公开 artifact。下载检查大小与 SHA-256。经典链接为 `#/results?run_id=...` 或 `#/comparison?batch_id=...`。

SwanLab 在线项目为 AutoAI-Skill。云端 ID 与 events.jsonl 保存在任务目录，跨命令续写同一个实验。client 结束不会停止后端 Worker，但中断期间未轮询的实时进度不会被补造；最终结果恢复后补齐。`cloud_status=verified` 表示配置和最近指标都与本地记录核对通过，其他状态都需说明并必要时 sync-cloud。指标按 test、external_test 或 pooled_oof 及模型名分组，不混淆评估口径。

Batch 的 result --download 在结果可比较时导出 predictions.xlsx；自动归档已就绪时同时下载 comparison.zip。归档未就绪时保存 archive-status.json，稍后再次获取结果，不为了下载自动重训或启动新实验。
