# 拉曼预处理 CSV 数据完整性与 Excel 兼容修复计划

> **历史计划归档（原计划日期：2026-07-14；归档标识：2026-09-29）：** 本文保留当时的目标、方案和验收记录，不能据此判断功能已交付或按文中的分支、推送、部署命令操作。当前事实与工作流程以 [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[README.md](../../README.md) 和[文档导航](../README.md)为准。当前主线为 `019f1cc`；服务器唯一开发目录为 `/users/fotile/AutoAI/Pan`，固定 `pan/agent`。不得创建新分支、worktree、fork 或可开发复制；开发用 Git 命令仅在该服务器 Pan 根目录执行。本次整理不提交、推送或部署，GitHub 其他分支保留。

## 目标与成功标准

修复预处理结果把超长 `XXX`、`Intensity` 数组交给 Excel 后拆成多行，导致续段数值落入 `Index`、`Name` 等列的问题。继续保持统一建模 CSV 的六列契约 `Index, Name, XXX, Intensity, Label, Sample_ID`，不静默删点或降采样；修复后，当前 50 个拉曼文件在截图参数（按行号 100–2000、`range_then_baseline`、`arPLS`）下必须能被 Excel 正确打开、填写标签、保存并由建模加载器完整回读。

根因已确认：原始 X 轴和强度被转成 `float32` 后又按 Python 完整浮点文本输出，并使用带空格的默认 JSON；1901 点使 `XXX` 达 34,569 字符、`Intensity` 达 37,414–37,793 字符，超过 Excel 单元格 32,767 字符上限。Excel 正好从该边界把 `3164.840087...` 和 `25.649410...` 等续段拆到后续行，形成截图中的列错位。

## 实现修改

### 1. 统一数值输出与长度保护

**涉及模块：** `backend/app/parsers.py`、`backend/app/hplc.py`

- 在解析层新增唯一的数值数组输出辅助函数：拒绝 NaN/Inf，将元素规范成 9 位有效数字，并使用 `json.dumps(..., separators=(",", ":"), allow_nan=False)` 生成无多余空格的紧凑 JSON。
- X 轴读取保留为 `float64`，避免原始 `450.82` 先变成 `float32` 再暴露为 `450.82000732421875`；强度和现有算法仍保持当前 `float32` 计算契约。9 位有效数字必须保证序列化后的强度重新转换为 `float32` 时逐点完全一致。
- `preprocess_raw_files`、`preprocess_raw_files_with_preview` 和 HPLC 统一使用该辅助函数，CSV 中的数组与 API 预览数组来自同一份规范化数据，避免页面曲线与下载文件不一致。
- 每个 `XXX`、`Intensity` 字段写入 DataFrame 前检查字符数不得超过 32,767。超过时返回明确的 400 错误，指出文件名、字段名、点数和建议缩小行号/X 轴范围；禁止静默截断、拆行或降采样。
- 保持成功响应字段、下载接口和六列 CSV 模式不变；新增的失败行为只覆盖无法安全交给 Excel 的超长数组。

### 2. 回归测试与接口契约

**涉及模块：** `backend/tests/test_smoke.py`、`docs/frontend_backend_handoff.md`

- 添加精度测试：输入含 `450.82` 等小数的原始 CSV，输出 `XXX` 保持紧凑原值，不出现 `450.82000732421875`；JSON 回读后的 X/Y 长度和对应关系不变。
- 添加无损测试：对包含正数、负数、小数和科学计数法的 `float32` 强度，规范化文本回读为 `float32` 后必须 `array_equal`；X 轴相对输入误差不得超过 9 位有效数字的界限。
- 添加 Excel 长度测试：1901/2048 点的代表性数据生成后，两个数组字段均小于等于 32,767 字符；构造确实无法容纳的更长数组时，函数与 HTTP 接口返回可读的 400 错误且不生成结果 CSV。
- 添加结构测试：用 Python 标准 `csv.reader` 读取结果，必须只有六列、每个样本一行；`Index` 连续唯一，`Name` 与上传文件顺序一致，`XXX`/`Intensity` 都是等长列表。
- 文档补充 Excel 兼容约束、9 位有效数字策略以及超限时缩小范围的错误说明，不改变建模 CSV 字段定义。

## 验证与验收

### 自动化验证

在项目根目录运行：

```powershell
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

所有既有测试和新增测试必须通过。此次不修改前端内联脚本，因此无需 Node 语法检查；若实施时增加前端提示，则补跑项目规定的脚本抽取和 `node --check --input-type=commonjs`。

### 当前拉曼数据真实闭环

使用 `D:\PythonProject\AutoAI\拉曼` 下 50 个 CSV，以截图参数重新调用 `/api/preprocess/raman`：

- HTTP 返回 200，结果严格为 50 个样本、每条 1901 个 X/Y 点，无 NaN/Inf。
- CSV 物理行数为 51（表头 1 行 + 样本 50 行），每行用标准解析器得到恰好 6 个字段。
- `Index` 必须为 1–50；`Name` 必须逐项对应上传文件；每行 `XXX` 与该原始文件第 100–2000 行 X 轴一一对应，`Intensity` 与同一条曲线的 arPLS 输出一一对应。
- 所有 `XXX`、`Intensity` 单元格均不超过 32,767 字符；预期当前数据最大约为 `XXX <= 14,819`、`Intensity <= 21,671`。
- 规范化前后的强度转成 `float32` 后逐点完全一致；不得以通过 Excel 验收为由删除、重排或插值数据点。

### Excel 与建模回读验收

- 用本机 Excel 以只读方式打开新 CSV，`UsedRange` 必须为 51 行 × 6 列；不得再出现以 `3164.84...`、`25.64941...` 开头的续段行，`Name` 列只能是文件名，`XXX`/`Intensity` 列只能是完整列表。
- 在副本中填写标签：按文件名前缀设置 `Label=big/small`，按 `big11..big15`、`small11..small15` 的首位组号设置 10 个 `Sample_ID` 组，每组 5 条；由 Excel 另存为 UTF-8 CSV。
- 使用 `load_modeling_csv` 回读该副本，必须得到 50 样本、2 个类别、10 个 `Sample_ID` 组、每组 5 条、曲线长度 1901；X/Y 与保存前逐点一致，随后至少用轻量传统模型完成一次训练/评估闭环。

只有自动化测试、真实拉曼数据、Excel 打开/保存和建模回读四项全部通过，才判定“数据没问题”并完成交付。

## 假设与边界

- 统一建模 CSV 六列及“每个样本一行、数组存于 JSON 字段”的现有契约保持不变。
- 当前修复覆盖拉曼、简单色谱和 HPLC 的共同导出风险；不引入 XLSX、不改变基线算法、不改变模型输入精度、不对光谱自动降采样。
- 对超过 Excel 上限且在 9 位有效数字紧凑输出后仍无法容纳的数据，明确拒绝并要求缩小范围，这是保证数据完整性优先于强行导出的默认策略。
