# HPLC 单一无损 CSV 修改计划

> **历史计划归档（原计划日期：2026-07-20；归档标识：2026-09-29）：** 本文保留当时的目标、方案和验收记录，不能据此判断功能已交付或按文中的分支、推送、部署命令操作。当前事实与工作流程以 [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[README.md](../../README.md) 和[文档导航](../README.md)为准。当前主线为 `019f1cc`；服务器唯一开发目录为 `/users/fotile/AutoAI/Pan`，固定 `pan/agent`。不得创建新分支、worktree、fork 或可开发复制；开发用 Git 命令仅在该服务器 Pan 根目录执行。本次整理不提交、推送或部署，GitHub 其他分支保留。

## 1. 目标与成功标准

HPLC 预处理只生成一个 `hplc_<token>.csv`。该文件继续使用
`Index, Name, XXX, Intensity, Label, Sample_ID` 六列和“每条曲线一行”的现有建模契约，
但新文件中的 `XXX` 改为实际展开的 X 坐标数组，不再写 `linspace-v1` 描述；
`Intensity` 保存 float64 线性映射后的往返无损 JSON 数值，不再为了 Excel 单元格上限自动降到 1 位小数。

成功标准：

- 每次 HPLC 预处理只返回一个下载 URL，也只落盘一个结果 CSV。
- 开启插值时，每行 `XXX` 都是 7500 个实际坐标组成的 JSON 数组，范围为 0–50 分钟且严格递增。
- 每行 `Intensity` 都是 7500 个实际映射结果，JSON 回读后的 float64 与序列化前数值逐点一致。
- 不再生成 `hplc_<token>_xxx.csv`，响应中不再出现 `xxx_download_url` 或 `xxx_rows`。
- 历史 `linspace-v1` 文件仍可上传建模，避免破坏已有数据。
- 拉曼和旧 `/api/preprocess/chromatography` 的 Excel 自适应精度策略不变。

## 2. 已核实的问题原因

检查的最近真实批次为 `storage/preprocessed/hplc_891bf57a83.csv`，包含 6 条真实命名曲线、
每条 7500 点。对应 6 个原始上传文件的强度文本最多有 17–18 位小数，最小有效变化约为
`0.000476837158`，因此原始数据并非只有 1 位小数。

当前代码对整个批次按 `5 → 4 → 3 → 2 → 1 → 0` 尝试统一精度，并要求每个数组序列化后的
单元格不超过 Excel 的 32,767 字符。真实批次的强度长度如下：

| 保留小数位 | 批次最长 Intensity 字符数 | Excel 单元格可容纳 |
| ---: | ---: | :---: |
| 5 | 63,581 | 否 |
| 4 | 56,093 | 否 |
| 3 | 48,479 | 否 |
| 2 | 40,761 | 否 |
| 1 | 32,378 | 是 |
| 0 | 19,339 | 是 |

因此落盘 1 位小数是现有 Excel 兼容策略的预期行为，不是插值算法只算出 1 位，也不是原始仪器精度不足。
六条曲线按 1 位小数量化后与当前文件逐点完全一致；相对未量化插值值的最大绝对误差小于 `0.05`，
平均绝对误差约为 `0.0234–0.0261`。

实际 X 坐标也无法同时满足“7500 点严格递增”和“单个 Excel 单元格不超过 32,767 字符”：

| X 小数位 | JSON 字符数 | 唯一坐标数 | 严格递增 |
| ---: | ---: | ---: | :---: |
| 5 | 65,161 | 7,500 | 是 |
| 4 | 57,663 | 7,500 | 是 |
| 3 | 50,159 | 7,500 | 是 |
| 2 | 42,599 | 5,001 | 否 |
| 1 | 34,508 | 501 | 否 |
| 0 | 21,076 | 51 | 否 |

完整 float64 X 数组采用 JSON 最短往返表示时约 138,112 字符。因此，本计划按用户当前目标选择
“一个 CSV、实际坐标、数值准确”优先，并明确取消新 HPLC 文件的 Excel 单元格兼容保证。
CSV 文件本身没有 32,767 字符限制，项目读取器和 pandas 可以正常读取；但 Excel 打开或再次保存时可能截断长单元格。

## 3. 关键实现决策

### 3.1 新 HPLC 输出使用 float64 往返无损 JSON

新增一个仅供 HPLC 使用的数组序列化函数：

- 输入必须是一维、非空且全部有限的 float64 数组。
- 使用 Python JSON 浮点最短往返表示和紧凑分隔符，不执行 `np.round`。
- 序列化后立即 JSON 回读，并验证长度一致、数值逐点往返一致。
- X 轴额外验证严格递增、首尾点和目标点数。
- 不应用 `EXCEL_CELL_CHARACTER_LIMIT`，但返回实际字符数和 `excel_cell_safe=false` 元数据。

这项变更只作用于 HPLC。通用/拉曼序列化继续保留现有 5–0 位自适应策略。

### 3.2 新文件写实际数组，历史文件继续兼容

`preprocess_hplc_files_with_preview()` 在插值开启时：

- `XXX` 写入 `target_x` 的完整实际 JSON 数组。
- `Intensity` 写入未量化映射结果的完整实际 JSON 数组。
- API 曲线预览与最终 CSV 从同一回读数组构造，避免页面和下载文件不一致。
- `output_precision` 改为准确描述序列化方式，例如：

```json
{
  "serialization": "json-float64-roundtrip",
  "lossless_roundtrip": true,
  "xxx_encoding": "expanded-array",
  "xxx_character_count": 138112,
  "intensity_max_characters": 150000,
  "excel_cell_safe": false,
  "point_count": 7500
}
```

`_parse_modeling_axis()` 仍保留 `linspace-v1` 分支，同时继续读取普通数组。历史文件不迁移、不重写。

### 3.3 路由只生成一个文件

删除 HPLC 专用的 `_hplc_visible_axis_frame()` 和 `_xxx.csv` 写出逻辑。
响应只保留主文件的 `download_url`；去掉 `xxx_download_url`、`xxx_rows`。
发生错误时仍不得落盘半成品。

### 3.4 前端明确数据准确性与 Excel 限制

经典前端和 v2 前端都删除“下载可见 XXX 时间轴”按钮和相关下载函数，只保留“下载整理 CSV”。
结果提示改为：X 与强度均保存实际 float64 数值、未降采样、未按 Excel 上限舍入；
同时明确提示不要用 Excel 打开后再次保存该文件，否则长单元格可能被截断。

## 4. 文件与任务顺序

### 任务 1：增加 HPLC 无损数组序列化并切换主文件内容

**修改：**

- `backend/app/parsers.py`
- `backend/app/hplc.py`
- `backend/tests/test_smoke.py`

**实施内容：**

1. 在 `parsers.py` 增加 HPLC 专用 float64 往返序列化函数，不改通用 Excel-safe 函数。
2. 在 `hplc.py` 中让插值开启分支对 `XXX` 和 `Intensity` 都使用新函数。
3. X 轴序列化后验证 7500 点、0–50 分钟、严格递增；强度验证长度与 X 一致。
4. 从最终回读值构造 CSV 和 API 预览，杜绝隐式二次转换。
5. 更新精度元数据，不再返回误导性的 `intensity_decimal_places=1`。

**验收：**

- 新 HPLC CSV 的 `XXX` 是数组而不是字典描述。
- 原始强度中小于 0.1 的有效变化在输出中仍存在。
- JSON 回读结果与序列化前 float64 数组逐点一致。
- 历史 `linspace-v1` CSV 仍能被 `load_modeling_csv()` 恢复为 7500 点。

### 任务 2：收敛为单一下载文件

**修改：**

- `backend/app/routers/preprocess.py`
- `backend/tests/test_smoke.py`

**实施内容：**

1. 删除 `_hplc_visible_axis_frame()`。
2. 删除 `_xxx.csv` 的生成与写盘。
3. 从 HPLC 响应删除 `xxx_download_url` 和 `xxx_rows`。
4. 增加文件级断言，确保一次请求只新增一个 `hplc_<token>.csv`。

**验收：**

- HPLC 成功响应只有一个结果下载 URL。
- 预处理目录没有对应 token 的 `_xxx.csv`。
- 下载主文件可直接读到每条曲线的完整 X 和强度数组。

### 任务 3：更新两套前端

**修改：**

- `static/index.html`
- `static/v2/views/workbench.js`
- `backend/tests/test_task13_ui_contract.py`
- `backend/tests/test_frontend_v2_contract.py`

**实施内容：**

1. 删除第二下载按钮和 `xxx_download_url` 分支。
2. 删除 v2 的 `downloadVisibleAxis()`。
3. 将结果说明改为“实际坐标数组 + float64 往返无损强度”。
4. 显示后端返回的实际字符数和 Excel 不兼容警告，不在前端自行估算。

**验收：**

- 页面只显示一个下载按钮。
- 页面不再出现 `linspace-v1` 或“采用 1 位小数”的新结果提示。
- 页面明确提示 Excel 重新保存可能截断长单元格。

### 任务 4：同步契约和项目文档

**修改：**

- `AGENTS.md`
- `CONTEXT.md`
- `README.md`
- `docs/frontend_backend_handoff.md`

**实施内容：**

1. 将新 HPLC 导出契约改为单一 CSV、展开 X 数组、无损 float64 强度。
2. 记录新 HPLC 文件不再保证 Excel 单元格兼容。
3. 明确旧 `linspace-v1` 只读兼容，不再用于新导出。
4. 保留拉曼和兼容色谱接口的现有精度规则。

## 5. 验证方案

### 自动化验证

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

经典前端内联脚本抽取后执行 `node --check --input-type=commonjs`，并运行现有 v2 Node 测试。

### 真实批次验证

使用本次已核对的 6 条、每条 7500 点 HPLC 原始文件重新执行预处理：

1. 确认只生成一个 CSV。
2. 每行回读 `XXX` 和 `Intensity`，均为 7500 点。
3. 确认 X 为 0–50 分钟严格递增数组。
4. 用 `map_hplc_intensity()` 重新计算未量化结果，与 CSV 回读值逐点比较。
5. 确认原先被量化为 `0.0` 的约 `0.00858`、`-0.02909` 等小值在新文件中保留。
6. 把新 CSV 补齐 Label/Sample_ID 后完成一次轻量训练闭环。
7. 不把原始数据、新 CSV、Run 目录或模型产物加入 Git。

## 6. 风险、兼容与备选方案

- **Excel 风险：** 新 HPLC 每行的 `XXX` 和 `Intensity` 都会显著超过 32,767 字符。CSV 本身有效，
  但 Excel 不是安全编辑器；用 Excel 保存后可能不可恢复地截断数组。
- **文件体积：** 同一固定 X 轴会在每条曲线重复保存，文件体积增大。这是满足“每行 XXX 都是实际坐标”的直接代价。
- **历史兼容：** 只保留旧描述符读取，不再输出旧描述符；不要批量改写历史文件。
- **回滚：** 后端、前端、测试和文档作为一个改动簇提交；回滚代码时不删除已生成文件。

如果业务仍要求用户必须在 Excel 中填写 Label/Sample_ID，则不能采用本计划的宽表长单元格方案。
届时应另立计划，把 HPLC 改为“每个点一行”的长表或“X 坐标作为特征列名”的矩阵表，
并同步修改建模加载器；这属于数据契约迁移，改动和验证范围明显更大。
