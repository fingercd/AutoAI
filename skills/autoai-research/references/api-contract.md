# 内部执行手册（供 Agent 使用）

本文件是操作参考，不是用户回复模板。工具返回的内部编号、连接状态和接口诊断只用于执行与恢复，不在普通对话中转述。用户看见的是数据情况、推荐模型、拟用参数、必要问题和最终结果。

## 准备运行环境

首次使用或环境异常时先按 [环境配置与修复手册](environment-setup.md) 执行检查。客户端依赖由技能根目录 requirements.txt 管理，本地计算另需项目 backend/requirements.txt。使用仅依赖标准库的 check_environment.py 检查，避免客户端缺少 httpx 时连诊断命令都不能运行。发现问题先告知用户，再按手册修复并重检；不把维护命令交给研究用户。

使用本技能目录中的 scripts。若技能根目录有 runtime.json，读取其中的 project_dir 和 python，采用该解释器和项目路径；它是机器本地配置，不含凭据。否则从工作目录或输入文件附近定位 AutoAI 项目。健康环境复用，不根据本机示例强制使用其他用户的路径。

服务地址可从 AUTOAI_SERVICE_URL 读取，默认 http://127.0.0.1:8000；凭据只从 AUTOAI_SERVICE_TOKEN 读取，不能进入命令行或日志。

```text
python <skill>/scripts/autoai_client.py --task-dir <experiment>/task ready --source <input-file> --project-dir <AutoAI-project>
python <skill>/scripts/autoai_client.py --task-dir <experiment>/task capabilities
```

ready 先检查已有服务的实际能力与执行可用性。已有环境满足要求就复用；本地服务未启动、过旧或执行不可用时，用当前项目代码在空闲端口后台启动独立的 Web 和 Worker，存储限制在本次实验目录，随后继续操作。不会终止其他服务、抢占原队列或改写用户数据。新地址与来源关系自动保存，后续命令使用同一任务目录即可恢复。

如果已经提交过任务，不能因故障换到另一队列创建重复训练。来自非本机的连接故障不擅自切换为本地实验，保留状态并诊断。环境问题均先通知用户，再尝试手册中的修复；确实超出 Agent 可处理的权限、文件或环境条件时，说明具体阻断和所需信息。

## 一次执行训练

先用 prepare_dataset.py inspect 只读检查文件，按 SKILL.md 汇报数据与推荐方案，并集中提出一轮可选问题。普通建模请求须等待回复后再 convert、preprocess 或 train；仅明确要求“直接运行”等跳过询问的请求可当轮执行。确定设置后按已明确语义建立映射并 convert，保存原始文件与转换记录。原始曲线需处理时先执行后文流程。

将最终设置保存为 plan.json，采用下面的格式。模型需按本次数据推荐或用户选择填写，示例不代表固定默认模型：

```json
{
  "task_type": "run",
  "config": {
    "model_type": "pls_da",
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

```text
python <skill>/scripts/autoai_client.py --task-dir <experiment>/task train <prepared>/dataset.csv --plan plan.json --project-dir <AutoAI-project> --seconds 60
```

train 会自行准备环境、上传已确认数据、完成严格检查、提交一次训练并等待结果。独立测试文件用 --external-file <prepared-external>/dataset.csv，并在计划中使用相应评估模式及主数据 8:2 的比例；内部 split_test=0。

多模型计划使用 task_type=batch、顶层 model_types 和 base_seed；config 不含 model_type/seed/split_seed/model_seed。每模型只运行一次。

返回 phase=training 时继续 watch，直到结束，再读取结果；不要在这一步结束用户任务或提出新确认：

```text
python <skill>/scripts/autoai_client.py --task-dir <experiment>/task watch --seconds 60
python <skill>/scripts/autoai_client.py --task-dir <experiment>/task result --download
```

返回的结构化数据只用来写用户能理解的摘要，不原样粘贴。训练过程中主动告知所选模型、特征处理、划分方式和拟用参数；无需解释实际调用了哪些接口。完整参数与审计记录保存在目录中。

## 原始曲线

准备环境后执行：

```text
python <skill>/scripts/autoai_client.py --task-dir <experiment>/task inspect-hplc curve-A.txt curve-B.txt
python <skill>/scripts/autoai_client.py --task-dir <experiment>/task preprocess --kind hplc --options preprocessing.json curve-A.txt curve-B.txt
```

options 示例为 {"range_mode":"row","start_row":1,"hplc_interpolate":true}。按动态检测点数处理范围，HPLC 不消负或归一化。拉曼使用 kind=raman，先截取后基线。参数有研究歧义时与模型和数据问题一起问清；已处理数据不重复处理。

预处理后的 preprocessed.csv 标签与样品编号待补齐，按 Name 应用用户提供或已确认的映射，再 convert 和 train。重复文件名、缺失标签或不明真实分组不能猜测。

## 内部检查、恢复与记录

底层仍调用现有公开接口：能力目录 /api/models、数据上传 /api/datasets/upload、训练预检 /api/training/preflight、Run/Batch 创建及结果接口。保持严格检查与服务端幂等，不能为了流程顺畅绕过数据校验或直接调用训练算法。

- 提交前把请求与幂等键写入 state.json。未知响应用同目录 submit 恢复，不创建新键；新实验用新目录。
- 已注册数据和首次环境切换的文件指纹自动恢复，不能将先前连接中的编号当成新环境里的数据。
- GET 和有幂等保障的创建请求可有限重试。上传、预处理和停止不盲目重试；核实后才使用 --retry-uncertain。
- 用户要求停止时调用同目录 stop，随后停止轮询。
- 下载校验大小和 SHA-256。单模型保存公开结果文件；Batch 可比较时导出 predictions.xlsx，归档就绪时下载 comparison.zip。
- 配置、seed、代码与数据版本、运行命令及提交信息保存在 state.json，进度和结果指标写入本地 events.jsonl。指标按 test/external_test/pooled_oof 和模型名组织，不调用外部实验记录服务。
- last_error 和 last_runtime_diagnosis 用于 Agent 排障；必要时读取本次 runtime 日志并解决问题。用户默认摘要不显示这些字段。
