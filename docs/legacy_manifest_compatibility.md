# 第五至八步迁移后的历史 Manifest 读取

> 本页仅定义旧 Manifest 的只读识别与下载边界。修复起点是已部署的 39743e4；当前主线 019f1cc 已包含第九步诊断，但不把新写入格式扩大为旧格式。本次文档核对不改变 Test 权限、写入格式或下载功能。维护范围见根目录 [AGENTS.md](../AGENTS.md)、[CONTEXT.md](../CONTEXT.md)。

历史格式来自 2026-09-13 保全的 `uncommitted-code.tar.gz`：
`backend/app/records/writer.py` 写入六位序号的 `details/<collection>/*.json`
及 `weights/fold-NNNN.pt|pkl`；旧 `runs/artifacts.py` 将其登记进 v2 Manifest。
保全的 writer SHA-256 为 `f49ea76ea6fc1faa329142d3edba48711e251eb4472c303ed21021e895a84a80`，
artifact 模块为 `85754ad5f63e98931ecf68fa3eb468ddf2614bb4eeea3d38832f5067990987b1`。

完整盘点的 385 个成功 Run 中，381 个含内部嵌套条目；实际集合为 epochs、
search_trials、candidate_samples、samples。读侧仅识别这四种已证实的集合，
不采用旧实现任意 `[a-z_]+` 目录放行。未知目录或版本仍可解释地失败。
新 writer 与 Guard 的扁平发布约束保持不变。

解析拒绝绝对路径、Windows 路径、反斜线、控制字符、百分号编码、空段、点段、
父目录及越出 Run 根目录的符号链接，包括 Manifest 本身的逃逸链接。
不会规范化可疑名字为 basename。原 Manifest、指标、权重、seed 和划分不改写。

公开下载必须同时符合 Manifest 标记和 catalog。历史全局重要性仍仅保留既有
直接下载兼容；v1 未登记 catalog 的扁平文本只允许 txt/csv/json。
内部 details、weights、模型对象及 joblib 不开放；含路径字段的 config 保持私有。
这些内部内容不会加入 Agent Observation。

公开结果文件在描述阶段核对大小与 SHA，指标投影校验实际读取的同一份字节。
v2 缺少摘要或大小会标为不完整。摘要不符、缺文件、Manifest 损坏均不会回退到
可变 `status.json` 的科学指标。内部私有文件的普通页面检查仍为大小检查；
完整迁移验收另外逐项核对全部登记文件摘要，不能将页面 ready 当作全盘审计。

第九步合并时保留 `_read_path` 的窄历史识别和私有下载限制；不要复制第二个
artifact 核心，也不要将历史识别规则加入新 Run Guard 发布白名单。
运行 `test_closeout_legacy_manifest.py`、`test_run_artifact_manifest.py`、
`test_run_result_contract_v1.py` 和第九步自身 Guard/诊断测试后，再做完整回归。
