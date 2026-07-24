# HPLC 动态点数检测、前端预检与原文件名宽表实施计划

## 目标与成功标准

将 HPLC 点数从固定 7500 改为由当前上传批次决定：选择文件后，后端按与正式预处理相同的解析规则返回逐文件原名、有效点数和时间范围；前端立即展示检测表，并用批次统一点数设置起止行范围。批次点数不一致时，以唯一出现次数最多的点数为预期值，明确列出所有偏离文件及其实际点数并阻止生成；若最高频次并列，则列出全部点数组且不选择基准。整批一致时支持任意不少于 2 的点数。

新预处理 CSV 在现有前三列后增加 `Name`，列顺序为 `Index,Label,Sample_ID,Name,<真实坐标…>`，其中 `Name` 保存上传文件的完整基础文件名（含扩展名、不含客户端路径）。训练加载器同时接受新带名宽表和现有无 `Name` 宽表，旧六列数组格式继续拒绝。

## 接口与数据契约

- 新增 `POST /api/preprocess/hplc/inspect`，multipart 参数仍为多个 `files`。成功解析请求本身时返回 200；即使批次不可处理，也通过 `processable=false` 和逐文件状态返回完整诊断，避免只暴露第一条错误。
- 检测响应包含 `files[]`（`name`、`point_count`、`x_start`、`x_stop`、`status`、可空 `message`）、`point_count_groups[]`、`expected_point_count`、`common_point_count`、`processable` 和批次级 `message`。检测临时文件在响应前删除。
- 正式 `POST /api/preprocess/hplc` 不信任前端检测结果，重新执行同一批次校验；不一致时 HTTP 400 文案使用原文件名，列出预期点数和所有偏离项。整批一致时用实际点数构造 `HplcGridConfig`，行号上限、目标轴点数和响应元数据均来自该配置。
- 新导出格式标识为 `wide-feature-v2`：`Index,Label,Sample_ID,Name,<features>`。`wide-feature-v1`（无 `Name`）继续只读兼容；新拉曼、HPLC 和兼容色谱导出都保留各自原文件名。

## 实施任务

### 1. 后端批次检测与动态 HPLC 网格

- 在 HPLC 模块提取单文件检查和批次统计逻辑，统一验证可解析的二维数值、有限值、至少 2 点及严格递增时间轴。
- 使用点数频次的唯一众数作为预期值；有偏离或解析错误时 `processable=false`，并稳定排序输出点数组及异常文件。众数并列时不选预期值。
- 删除生产路径对 7500 默认值的依赖；保留 0–50 分钟、容差和单位为轴业务配置，点数必须由批次检查结果或测试显式注入。
- 正式预处理复用检查结果和原始数组，错误、插值、关闭插值及曲线预览全程使用上传原文件名，而不是 UUID 落盘名。

### 2. 新宽表 Name 列与训练兼容

- 宽表构造器在前三列后写入 `Name`，特征从第 5 列开始；Excel 特征上限相应按 4 个元数据列计算。
- 建模读取器按表头识别 v2（第四列为 `Name`）或 v1（第四列起为数值坐标），校验 v2 的 Name 非空；返回元数据时保留 Name，但标签、Sample_ID 分组和训练矩阵逻辑不变。
- 摘要、预览和格式元数据报告实际格式；旧 v1 文件继续加载，旧六列 `XXX/Intensity` 数组文件继续给出迁移错误。

### 3. 两套前端选择即检测

- 经典前端与 v2 工作台在 HPLC 文件选择变化时调用 inspect 接口，显示“原文件名、有效点数、时间范围、状态”的逐文件表格。
- 检测成功且整批一致时，将起始/终止行输入的 `max` 和默认终止值设置为 `common_point_count`；提交校验显式传入该动态值，不再使用前端 7500 常量。
- 检测中、检测失败或批次不可处理时禁用“开始/生成”按钮；更换文件后使旧检测结果失效，避免异步响应覆盖新选择。
- 页面文案改为“范围上限由所选批次检测结果决定”，成功结果同时显示完整批次点数和本次截取后的实际点数。

### 4. 测试、文档和真实数据验收

- 单元测试覆盖任意一致点数（如 5、7500、8000）、唯一多数含偏离者、众数并列、解析失败、原文件名错误信息、动态行范围和插值/关闭插值行为。
- API 测试覆盖 inspect 的结构化响应、临时文件清理、正式接口二次校验，以及 v2 CSV 下载后 Name 与上传文件名逐行对应。
- 宽表契约测试覆盖 v2 读写、v1 兼容、Name 空值拒绝、主/独立测试集轴一致性和训练 smoke。
- 前端契约测试覆盖无 7500 生产常量、动态 `pointCount` 传入校验器、逐文件表格与按钮门禁；经典内联脚本执行 Node 语法检查。
- 更新 `AGENTS.md`、`CONTEXT.md`、README 和前后端交接文档，删除“固定 7500/前三列后立即是特征”的旧描述。
- 使用 `24个样品色谱数据` 的 72 个文件进行真实预检：当前应全部报告 7500、可处理；复制一份测试夹具增加/删除一点，确认异常文件名和实际点数准确。运行全量后端测试、compileall，并在本地 `data.csv` 可用时完成轻量训练闭环。

## 验证命令

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

另外抽取 `static/index.html` 的内联脚本运行 `node --check --input-type=commonjs`，并执行现有 v2 Node 合约测试。验证完成后重启 Web 与 Worker，重新调用线上 inspect 和 HPLC 预处理接口确认运行进程已加载新契约。

## 明确边界

- 不自动截断、补点或吞掉少数点数异常文件；批次不一致必须由用户修正后重新选择。
- 动态支持的是点数，不改变 HPLC 0–50 分钟业务范围、线性映射、边界相位容差、强度精度或面积/消负规则。
- 不重写历史 CSV、Run 或原始数据；旧 v1 建模文件保持可训练，新输出统一使用 v2。
