# HPLC 真实时间轴、7500 点边界、单一 CSV 与界面术语统一实施计划

> **历史计划归档（原计划日期：2026-07-21；归档标识：2026-09-29）：** 本文保留当时的目标、方案和验收记录，不能据此判断功能已交付或按文中的分支、推送、部署命令操作。当前事实与工作流程以 [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[README.md](../../README.md) 和[文档导航](../README.md)为准。当前主线为 `019f1cc`；服务器唯一开发目录为 `/users/fotile/AutoAI/Pan`，固定 `pan/agent`。不得创建新分支、worktree、fork 或可开发复制；开发用 Git 命令仅在该服务器 Pan 根目录执行。本次整理不提交、推送或部署，GitHub 其他分支保留。

## 1. 目标与当前结论

本计划覆盖两个最终一起回归、但可分开审查的改动簇：

1. HPLC 保留真实保留时间，完整输入固定为 7500 个有效点；默认开启插值时，用户按行号或
   保留时间选择范围后，只使用实际选中的固定时间点进行插值、导出和训练。关闭插值时继续
   使用各文件所选原始 X/Y。前端终止行最大为 7500，越界必须在前端提示，并由后端再次拒绝。
2. 落实 `0720修改.txt` 中的 AI 建模、建模结果和训练记录文案调整，并在训练记录增加
   “测试集 Macro F1”。

本文件是当前推荐执行稿，并明确取代以下两个未确认草稿中的 HPLC 方案：

- 不采用 `2026-07-21-hplc-fraction-axis-and-ui-terminology.md` 的 `n/7500`、
  `fraction-range-v1` 和 0–1 归一化坐标，因为这些值不是真实保留时间。
- 不采用 `2026-07-20-hplc-single-lossless-csv.md` 的“把 7500 个 float64 X/Y 全部展开到单元格”
  方案，因为它会主动放弃 Excel 32,767 字符兼容。该草稿的“只保留一个主 CSV”目标继续保留。

实施时不删除或改写这两份历史草稿；以本文件作为唯一实施依据。

## 2. 已确认的业务契约

### 2.1 完整输入和真实时间

- 每个原始 HPLC 文件必须解析出恰好 `HplcGridConfig.point_count` 个有效点，默认是 7500 点。
- 原始 X/Y 必须一维、等长、有限，X 严格递增。
- 默认固定目标轴继续是包含首尾端点的 0–50 分钟、7500 点：

```text
t(n) = start_minutes + (n - 1) * (stop_minutes - start_minutes) / (point_count - 1)
```

- 默认配置下，第 `n` 个 1 基点是 `(n - 1) * 50 / 7499` 分钟；不能使用 `n/7500` 代替时间。
- 7500 是“完整原始文件和完整固定网格”的长度，不代表每次范围截取后的输出仍必须为 7500 点。

### 2.2 行号范围

- 行号使用 1 基、首尾都包含。
- 起始行和终止行都必须在 1–7500 内，且起始行不得大于终止行。
- 终止行留空时按 7500 处理。
- 不能像当前 Python slice 那样把 7501、9000 等值静默截成文件末尾。

| 请求范围 | 实际点数 | 固定轴位置 | 默认配置下的真实时间范围 |
| --- | ---: | --- | --- |
| 1–7500 | 7500 | 第 1–7500 点 | 0–50 分钟 |
| 1–4000 | 4000 | 第 1–4000 点 | 0–约 26.663555 分钟 |
| 100–4000 | 3901 | 第 100–4000 点 | 约 0.660088–26.663555 分钟 |
| 7501 作为终止行 | 0 | 非法 | 前端提示，后端 HTTP 400 |

### 2.3 按保留时间范围

- HPLC 的 X 值输入明确命名为“保留时间下限/上限（分钟）”。
- 开启插值时，范围作用于完整固定目标时间轴，而不是把原始点编号或 0–1 分数当作时间。
- 使用闭区间选择：`target_time >= lower` 且 `target_time <= upper`。
- 边界留空时分别等价于不限制该侧；提供的边界必须是有限数值，且下限不得大于上限。
- 开启插值时只保留固定目标轴中命中的点；输入边界超出 0–50 不做静默钳制，但只可能命中配置
  轴内的点。关闭插值时继续在每条文件自己的真实原始 X 轴上应用相同时间范围，不强制要求
  原始轴边界等于 0 或 50。
- 用户填写的时间不一定恰好落在固定网格点上；最终运行范围以落入闭区间的实际固定时间点为准，
  API 和前端必须展示实际首点、末点和点数。
- 开启插值时，选择结果少于 2 个固定时间点继续返回明确错误。

### 2.4 强度如何运行

- 先读取并验证每条完整 7500 点源曲线。
- 开启插值时，只把用户选中的固定真实时间点传给 `map_hplc_intensity()`；每个目标点仍从完整源曲线
  中寻找左右邻点进行 float64 分段线性映射。
- 选择 100–4000 时，最终 `XXX`、`Intensity`、预览和训练特征都必须是 3901 点；不能重新铺满
  0–50 分钟，也不能补回 7500 点。
- 保留现有边界相位差不超过一个采样间隔时的线性延伸规则。
- 不新增消负、面积归一化或降采样。
- `hplc_interpolate=false` 继续导出所选原始 X/Y；新紧凑描述只用于默认开启插值的固定时间轴路径。

### 2.5 Excel 与强度精度边界

- `XXX` 使用本计划的紧凑真实时间轴 JSON，避免把数千个小数完整写入单元格。
- `Intensity` 继续使用现有的批次统一 Excel-safe 精度搜索：最多 5 位，并在 5→0 位中选择所有样本都
  不超过 32,767 字符的最高精度。
- 若 `Intensity` 即使 0 位仍超限，或降精度会使有效变化全部消失，继续拒绝导出；本任务不通过
  静默删点、降采样或放弃 Excel 兼容绕过保护。

## 3. `XXX` 的紧凑真实时间 JSON 方案

### 3.1 为什么不继续写 `1/7500`

`1/7500` 是点位置比例，不是分钟。默认固定轴的第 1 点是 0 分钟，第 7500 点是 50 分钟；
真实时间分母来自 7499 个采样间隔，而不是 7500 个点。

同时，把 7500 个分数或小数作为数组完整写出仍会超过 Excel 单元格上限。因此，CSV 的每个
`XXX` 单元格只保存“完整固定分钟网格 + 本次连续切片”的可逆规则。

### 3.2 新描述类型

完整范围：

```json
{"type":"linspace-slice-v1","grid_start":0.0,"grid_stop":50.0,"grid_count":7500,"offset":0,"length":7500,"unit":"minute"}
```

行号 100–4000：

```json
{"type":"linspace-slice-v1","grid_start":0.0,"grid_stop":50.0,"grid_count":7500,"offset":99,"length":3901,"unit":"minute"}
```

字段语义：

- `grid_start/grid_stop/grid_count` 描述完整固定真实时间轴；
- `offset` 是零基切片起点；
- `length` 是实际选中并导出的点数；
- `unit` 固定来自 `HplcGridConfig.unit`，默认是 `minute`。

读取器必须先构造完整轴，再切片：

```python
full_axis = np.linspace(grid_start, grid_stop, grid_count, dtype=np.float64)
selected_axis = full_axis[offset:offset + length]
```

这样比“只保存切片首尾小数后再次局部 `linspace`”更严格：CSV 回读结果能与管线实际使用的完整轴
切片逐位一致，同时描述文本仍远小于 Excel 单元格限制。

### 3.3 严格解析规则

`_parse_modeling_axis()` 对 `linspace-slice-v1` 执行白名单解析：

- `grid_start/grid_stop` 必须是有限数值，且起点小于终点；
- `grid_count` 必须是真整数，布尔值不能冒充整数，范围为 2 到
  `MAX_AXIS_DESCRIPTOR_POINTS`；
- `offset` 和 `length` 必须是真整数，`offset >= 0`、`length >= 1`；
- `offset + length <= grid_count`；
- `unit` 必须命中描述类型的单位白名单；`linspace-slice-v1` 初始只接受 `minute`；
- 展开后必须有限、严格递增且非空；
- `Intensity` 仍只接受非空数值数组，不能使用轴描述。

历史兼容必须保留：

- 普通 `XXX=[...]` 数值数组继续读取；
- 现有 `linspace-v1` 文件继续读取；
- 不批量迁移或重写历史 CSV。

## 4. HTTP 响应与单一文件契约

### 4.1 主 CSV

一次成功的 HPLC 预处理只生成一个：

```text
hplc_<token>.csv
```

文件继续保持六列和“一条曲线一行”：

```text
Index, Name, XXX, Intensity, Label, Sample_ID
```

默认插值路径的每一行使用相同 `linspace-slice-v1` 描述；`Intensity` 长度必须等于描述展开后的
实际选择点数。

删除以下旧输出：

- `hplc_<token>_xxx.csv`；
- `xxx_download_url`；
- `xxx_rows`；
- 两套前端的“下载可见 XXX 时间轴”按钮和处理函数。

保留主文件 `download_url`，页面只提供一个“下载统一建模 CSV”入口。

### 4.2 HPLC 轴元数据

保留现有字段含义，并补充完整网格及实际切片位置：

```json
{
  "start": 0.660088011734898,
  "stop": 26.663555140685425,
  "unit": "minute",
  "point_count": 3901,
  "step_minutes": 0.006667555674089879,
  "mapping": "piecewise_linear",
  "input_point_count_required": 7500,
  "encoding": "linspace-slice-v1",
  "grid_start": 0.0,
  "grid_stop": 50.0,
  "grid_point_count": 7500,
  "selected_start_row": 100,
  "selected_end_row": 4000
}
```

约束：

- `start/stop/point_count` 描述本次实际选中的输出时间轴；
- `grid_*` 描述完整固定网格；
- `selected_start_row/selected_end_row` 始终是完整固定网格中的 1 基位置；
- X 值范围模式也必须返回最终命中的实际行位置；
- `curves[].x` 和 `common_time` 继续返回实际选择的分钟数组，保持图表兼容；
- `output_precision.xxx_encoding` 改为 `linspace-slice-v1`，
  `xxx_max_characters` 使用最终 JSON 的真实字符数；
- 关闭插值时继续使用 `hplc_axis=null`、`common_time=[]` 和原始 X 数组契约。

## 5. 成功标准

### HPLC 数据与范围

- 完整输入 7499、7501 或其他非 7500 点文件被拒绝。
- 行号 1–7500 成功，输出和训练读取均为 7500 点、0–50 分钟。
- 行号 100–4000 成功，输出和训练读取均为 3901 点，首末时间等于完整固定轴第 100、4000 点。
- 行号 0、终止行 7501、9000、起点大于终点均被前端阻止，并被后端 HTTP 400 拒绝。
- 按时间范围时，输出点数等于实际落入闭区间的固定时间点数量；不重标、不拉伸。
- CSV 经 `load_modeling_csv()` 展开后的 X 与 API `common_time` 使用 `np.array_equal` 逐位一致。
- 同一请求切换新旧 X 描述编码前后，映射强度和最终 `Intensity` 序列化值不发生额外变化。
- 一次成功请求只生成一个主 CSV，响应和页面均没有第二个 XXX 下载入口。

### 界面和训练记录

- 两套前端对相同业务概念使用同一中文名称。
- AI 建模摘要按固定顺序显示：数据量、类别数、样本数、每样本测量数、特征数。
- 分组标题为“按样本分组”，表头为：样本编号、类别、每样本测量数。
- “标签分布”改为“类别分布”，表头为：类别、数据量。
- 模型目录成功加载后不再显示“模型目录已加载；模型按文档分类分组显示。”。
- 建模结果把“曲线数”改为“数据量”，“Sample_ID 数”改为“样本数”。
- 创建时间和开始时间合并为一个“训练时间”：优先显示开始时间；尚未开始时显示创建时间并标注
  “任务创建”。
- 用户可见的 OOF 文案改成“交叉验证”相关中文；内部字段、枚举和文件名不改。
- 各类别指标中的 `Support` 和“样本量”统一显示为“数据量”。
- 两套训练记录都增加“测试集 Macro F1”；成功 Run 显示测试主口径值，其余显示 `—`。

## 6. 术语映射和替换边界

| 当前可见文案 | 目标文案 | 边界 |
| --- | --- | --- |
| 总数据 / 曲线数（行） | 数据量 | 对应 CSV 曲线行数 |
| 类别数 | 类别数 | 保持 |
| Sample_ID 组数 | 样本数 | API 字段名不改 |
| 每组测量数 | 每样本测量数 | 对应重复测量数 |
| 曲线长度 | 特征数 | 对应实际选择后的点数 |
| Sample_ID 分组 | 按样本分组 | 标题 |
| Sample_ID | 样本编号 | 只改业务表格；CSV 字段不改 |
| Label | 类别 | 只改业务表格；CSV 字段不改 |
| 数量（分组表） | 每样本测量数 | 分组内重复曲线数 |
| 标签分布 | 类别分布 | 标题 |
| Count / 数量（类别表） | 数据量 | 类别曲线数 |
| 曲线数（结果页） | 数据量 | `dataset.curve_count` 不改 |
| Sample_ID 数（结果页） | 样本数 | `dataset.sample_id_count` 不改 |
| 创建时间 + 开始时间 | 训练时间 | 开始优先，创建回退 |
| 合并 OOF 预测 | 合并交叉验证预测 | `pooled_oof` 不改 |
| Support / 样本量 | 数据量 | 指标数值不改 |

`Run ID`、CSV 中的 `Sample_ID`/`Label`、模型 ID、API key、artifact 文件名，以及 Precision、Recall、
F1、Macro F1 等正式指标名称不做机械替换。技术文档首次出现时可以写“交叉验证折外预测（OOF）”，
但用户界面不再直接显示 OOF。

## 7. 分任务实施

### 任务 A1：建立 HPLC 专用严格范围选择

**涉及文件或模块：**

- 修改：`backend/app/hplc.py`
- 测试：`backend/tests/test_smoke.py`

**接口与依赖：**

- 消费：`HplcGridConfig`、`build_hplc_target_axis()` 和现有范围请求参数。
- 产出：一个包含完整目标轴、连续零基索引、offset、length、1 基首尾行及实际选择分钟数组的
  HPLC 选择结果。

**实施内容：**

1. 增加 HPLC 专用纯函数或小型不可变数据结构，例如 `_select_hplc_target_axis()` /
   `HplcAxisSelection`；不要修改通用拉曼 `_range_indexer()` 的兼容行为。
2. 行号模式严格校验 `1 <= start_row <= end_row <= config.point_count`；`end_row=None` 映射为
   `config.point_count`。错误信息同时包含允许上限和实际值。
3. 时间模式校验有限值和上下限顺序；开启插值时在完整固定轴上执行闭区间选择，不把超出配置
   轴的输入边界静默改写成 0 或 50。
4. 选择时同时处理 `np.arange(config.point_count)`，直接得到整数位置；禁止从浮点时间反推行号。
5. 验证选择结果非空、连续；开启插值时至少 2 点。
6. `preprocess_hplc_files_with_preview()` 使用选择结果的真实 `target_x` 进行映射；源曲线仍先完整
   验证，再由完整 X/Y 提供插值邻点。
7. 关闭插值分支保留原始 X/Y，复用同一行号上限、有限值和上下限顺序校验；时间范围继续分别
   应用于每条原始 X 轴，不强制套用固定轴边界或 `linspace-slice-v1`。

**验收标准：**

- 100–4000 返回 offset=99、length=3901，实际时间与完整轴切片逐位一致。
- 7501/9000 不再被静默截到末尾。
- 自定义 `HplcGridConfig(point_count=5)` 使用 5 作为上限，证明算法未硬编码 7500。

**验证：**

- 单测覆盖完整范围、部分行范围、时间范围、空范围、单点、倒序及所有越界分支。

### 任务 A2：实现 `linspace-slice-v1` 编解码

**涉及文件或模块：**

- 修改：`backend/app/hplc.py`
- 修改：`backend/app/parsers.py`
- 测试：`backend/tests/test_smoke.py`

**接口与依赖：**

- 消费：任务 A1 的 offset/length 和注入的完整 `HplcGridConfig`。
- 产出：短 JSON `XXX` 描述，以及 `load_modeling_csv()` 可恢复的真实分钟数组。

**实施内容：**

1. 把 `_axis_descriptor()` 改为接收完整网格配置和整数切片位置，生成紧凑
   `linspace-slice-v1` JSON；不从切片首尾浮点数反推网格。
2. 在 `_parse_modeling_axis()` 增加新类型的严格白名单分支，先构造完整 `np.linspace`，再按
   offset/length 切片。
3. 保留普通数组和历史 `linspace-v1` 分支。
4. 更新 `_axis_metadata()` 和 `output_precision`，区分完整网格点数、选择点数和输入要求点数。
5. 从同一个选择数组构造 CSV、`curves[].x` 和 `common_time`，避免三处语义漂移。
6. 不修改 `mapped_intensities`、`_serialize_modeling_arrays(..., "Intensity", ...)` 或任何强度算法。

**验收标准：**

- 完整与截取描述均低于 Excel 单元格上限。
- 新描述展开结果与 `common_time` 逐位相同。
- `len(XXX) == len(Intensity)`，并且 X 严格递增。
- 畸形类型、布尔整数、越界切片、空切片、非法单位和超大网格全部拒绝。
- 历史普通数组和 `linspace-v1` CSV 继续正常加载。

### 任务 A3：HPLC 收敛为单一主 CSV

**涉及文件或模块：**

- 修改：`backend/app/routers/preprocess.py`
- 测试：`backend/tests/test_smoke.py`

**实施内容：**

1. 删除 `_hplc_visible_axis_frame()`；若 `pandas` 不再被本路由使用，删除对应 import。
2. 删除 `_xxx.csv` 创建和写盘逻辑。
3. 删除响应中的 `xxx_download_url`、`xxx_rows`。
4. 保留主 CSV 的 `download_url`、预览、`common_time`、`hplc_axis`、强度摘要、精度元数据和警告。
5. 在测试中把 `PREPROCESSED_DIR` 重定向到临时目录，按 token 断言一次请求只新增一个主 CSV。
6. 失败请求不得生成主文件或半成品；上传暂存文件的既有生命周期不在本任务中扩展。

**验收标准：**

- API 只有一个结果下载 URL。
- 主 CSV 由 `load_modeling_csv()` 独立恢复全部真实时间，不依赖第二文件。

### 任务 A4：更新两套 HPLC 前端范围和结果展示

**涉及文件或模块：**

- 修改：`static/index.html`
- 修改：`static/js/ui-utils.js`（共享、可测试的 HPLC 行号校验纯函数）
- 修改：`static/v2/views/workbench.js`
- 修改：`static/v2/views/manual.js`
- 测试：`backend/tests/test_smoke.py`
- 测试：`backend/tests/test_task13_ui_contract.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`
- 测试：`static/v2/tests/run-tests.mjs`（导入共享纯校验函数并执行行为测试）

**实施内容：**

1. 经典前端把色谱终止行默认值从 9000 改为 7500；起始/终止输入均设置 `min=1,max=7500,step=1`。
2. v2 的 HPLC 行号输入设置同样上限；拉曼输入不套用 HPLC 的 7500 点规则。
3. 在 `static/js/ui-utils.js` 增加并导出共享纯函数（例如 `validateHplcRowRange()`），统一完成
   整数、1–7500、空终点归一化和起点不大于终点校验；经典前端通过 `window.SpecAutoAIUI`
   调用，v2 直接 import，同一规则不能复制两份。
4. 两套前端都必须在调用 API 前执行该函数；不能只依赖 HTML `max`。
5. 越界提示统一为可执行信息，例如：
   `HPLC 终止行不能超过 7500；当前为 7501。每个色谱文件固定为 7500 点。`
6. 时间模式标签改成“保留时间下限/上限（分钟）”；开启插值时说明固定目标范围为 0–50 分钟，
   关闭插值时说明会按每条文件自己的原始时间轴筛选。
7. 行号摘要显示实际预计点数，例如“第 100–4000 行，共 3901 点；最大终止行 7500”。
8. 成功结果以服务端 `hplc_axis` 为准，显示：
   `XXX：真实保留时间 0.660088–26.663555 分钟，第 100–4000 点，共 3901 点；CSV 使用 linspace-slice-v1 JSON 保存。`
9. 曲线图 X 轴继续标为“保留时间（分钟）”，不得显示“归一化 X 坐标”。
10. 删除第二下载按钮、经典前端相关点击处理和 v2 `downloadVisibleAxis()`。
11. 更新 v2 使用说明：`XXX` 是可恢复真实分钟轴的 JSON 描述，不是必须展开写出的 X 数组；
    页面和 API 只保留一个主 CSV 下载入口。
12. 页面只保留“下载统一建模 CSV”。

**验收标准：**

- `validateHplcRowRange()` 的 Node 测试覆盖空终点、100–4000、0、7501、9000、小数和倒序；
  两套提交路径均调用该函数。
- 人工浏览器验收使用 Network 面板确认 7501/9000 不会发出请求。
- 绕过前端直接请求同样由后端返回 400。
- 100–4000 的页面摘要与后端实际 3901 点一致。
- 页面不出现 `fraction-range-v1`、`1/7500` 或归一化坐标文案。

### 任务 B1：统一 AI 建模摘要、分组表和类别表

**涉及文件或模块：**

- 修改：`static/index.html`
- 修改：`static/js/ui-utils.js`
- 修改：`static/v2/views/modeling.js`
- 测试：`backend/tests/test_smoke.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`

**实施内容：**

1. 两套摘要统一为五项固定顺序：数据量、类别数、样本数、每样本测量数、特征数。
2. v2 使用现有 `summary.sample_id.expected_repeats_per_group` 补齐每样本测量数。
3. 把经典前端的“Sample_ID 分组”改为“按样本分组”。
4. 共享分组表可见表头改为“样本编号、类别、每样本测量数”；内部 `sample_id/label/count` 不改。
5. 把“标签分布”改为“类别分布”，列名改为“类别、数据量”。
6. 模型目录成功加载时清空或隐藏成功提示；加载失败、刷新失败和不可用原因继续显示。
7. 不修改 CSV 的 `Label`、`Sample_ID`，也不修改后端摘要字段。

**验收标准：**

- 主数据集和独立测试集摘要名称与顺序一致。
- 大量样本的折叠、自然排序、键盘聚焦和无障碍文本不回归。

### 任务 B2：统一建模结果文案与训练时间

**涉及文件或模块：**

- 修改：`static/index.html`
- 修改：`static/js/run-results.js`
- 修改：`static/v2/views/result.js`
- 修改：`static/v2/views/modeling.js`
- 修改：`static/v2/views/manual.js`
- 修改：`static/v2/lib/format.js`
- 修改：`backend/app/runs/artifacts.py`（仅用户可见标签）
- 测试：`backend/tests/test_result_frontend_contract.py`
- 测试：`backend/tests/test_task13_ui_contract.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`
- 测试：`static/v2/tests/run-tests.mjs`

**实施内容：**

1. 两套结果页把曲线数/Sample_ID 数改为数据量/样本数；独立测试曲线数改为“独立测试数据量”。
2. 提供统一训练时间格式：`started_at` 优先；没有开始时间时使用 `created_at` 并追加“任务创建”。
3. 删除同一卡片中重复的创建时间、开始时间；结束时间和耗时保留。
4. queued/running/failed/cancelled/缺失结果等降级卡片使用同一时间规则，避免只改成功页。
5. 用户可见的 `合并 OOF 预测` 改为“合并交叉验证预测”；说明文字改为“全部交叉验证折的测试预测合并计算”。
6. 更新经典结果页、v2 建模向导、v2 使用说明、v2 结果页和 artifact 可见标签；内部
   `pooled_oof`、`cv_predictions.csv` 和计算口径不改。
7. 经典分类报告 `Support` 表头改为“数据量”，“各类别指标”说明中的“样本量”同步修改。
8. v2 的“真实/预测数量”统一为“真实/预测数据量”；不凭空生成当前不存在的分类报告表。

**验收标准：**

- CV Test 主指标仍取 pooled 主值，文案变化不改变计算。
- 用户界面搜索不到孤立的 `OOF` 和 `Support`；源码契约枚举和文件名仍存在。
- 每个结果状态区域只出现一个“训练时间”。

### 任务 B3：训练记录增加测试集 Macro F1

**涉及文件或模块：**

- 修改：`backend/app/routers/runs.py`
- 修改：`static/index.html`
- 修改：`static/v2/components/run-list.js`
- 修改：`static/v2/lib/format.js`（若复用指标格式函数）
- 测试：`backend/tests/test_run_result_contract_v1.py`
- 测试：`backend/tests/test_smoke.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`
- 测试：`static/v2/tests/run-tests.mjs`

**接口与依赖：**

- `GET /api/training/runs?projection=summary` 每项新增：

```json
{"test_macro_f1": 0.9234}
```

字段可为空，不返回完整 `metrics`。

**实施内容：**

1. 增加只解析测试主 Macro F1 的小型辅助函数；列表页不得逐 Run 调用 `project_run_result()`。
2. 仅对 `succeeded` Run 读取指标，并通过 Manifest 的大小/SHA-256 完整性验证。
3. holdout/external test 优先读取 `metrics.json -> test -> macro_f1`。
4. 留一交叉验证优先读取测试主口径；若需回退 `cv_metrics.json`，只允许
   `cv_summary.pooled_test.macro_f1`，禁止使用 `fold_mean.test.macro_f1`。
5. 旧格式只在存在明确、有限且可确认属于测试集的字段时兼容；不得把 valid F1 或不明顶层值猜成测试值。
6. 值必须有限且位于 0–1；非成功、缺失、损坏、非法值返回 `null`。
7. `_summary_projection()` 只加入这个标量，不泄露完整指标、配置或服务器路径。
8. 经典训练记录和 v2 记录增加“测试集 Macro F1”列，固定显示 4 位小数；空值显示 `—`。
9. v2 的“创建时间”列同步改成“训练时间”，使用开始优先/创建回退规则。
10. 更新空表列数、响应式宽度、键盘行导航和操作按钮断言。

**验收标准：**

- 成功 CV Run 的列表值等于结果页 pooled test Macro F1，并明确不等于 fold mean。
- queued/running/failed/cancelled、缺失或损坏产物显示 `—`。
- summary API 仍不包含完整 `metrics`、`config` 或任何服务器路径。

### 任务 C：同步权威文档和契约

**涉及文件或模块：**

- 修改：`AGENTS.md`
- 修改：`CONTEXT.md`
- 修改：`README.md`
- 修改：`docs/frontend_backend_handoff.md`
- 按需修改：`docs/run_result_contract.md`（只补与列表摘要相关的交叉引用，不改变 run-result-v1）

**实施内容：**

1. 写明完整 HPLC 输入固定 7500 点，但范围输出和训练特征数等于实际选择点数。
2. 写明真实时间公式、1 基含首尾行号、终止行最大值和 100–4000=3901 点示例。
3. 把新 HPLC `XXX` 契约更新为 `linspace-slice-v1`，并保留历史 `linspace-v1` 读取兼容说明。
4. 删除仍生成 `xxx_download_url`/第二个 `_xxx.csv` 的陈旧说明。
5. 明确 `Intensity` 仍使用 Excel-safe 自适应精度，不把本任务描述成 float64 强度无损导出。
6. 记录两套前端的 7500 点双重门禁和术语映射。
7. 记录 summary API 的可空 `test_macro_f1` 及 CV pooled test 口径。
8. 技术文档可以保留“交叉验证折外预测（OOF）”用于精确定义，UI 操作文案使用中文。

**验收标准：**

- AGENTS、CONTEXT、README、handoff 对点数、时间轴、JSON 类型、文件数和指标口径没有冲突。

## 8. 测试矩阵

### 8.1 HPLC 范围与编码

| 用例 | 请求 | 预期 |
| --- | --- | --- |
| 完整范围 | 1–7500 | 7500 点；0–50 分钟；offset=0,length=7500 |
| 前半范围 | 1–4000 | 4000 点；offset=0,length=4000 |
| 中间范围 | 100–4000 | 3901 点；offset=99,length=3901 |
| 最后一行 | 7500–7500 | 关闭插值可导出 1 点；开启插值按现有至少 2 点门禁拒绝 |
| 起点非法 | 0–100 | HTTP 400，包含允许范围 |
| 终点越界 | 1–7501 / 1–9000 | HTTP 400，不静默截取 |
| 范围倒序 | 4000–100 | HTTP 400 |
| 时间范围 | 0.66–26.67 分钟 | 选择闭区间内实际固定点，返回实际首末时间和点数 |
| 时间超出固定轴 | -1–10 / 10–51 | 开启插值时只选择固定轴内命中点，不静默改写输入；关闭时按原始轴筛选 |
| 时间空集/单点 | 不命中或仅命中 1 点 | 空集拒绝；插值开启的单点拒绝 |
| 输入长度 | 7499/7501 点文件 | 固定流程拒绝 |
| 历史文件 | 数组 / linspace-v1 | 继续读取 |
| 新文件 | linspace-slice-v1 | 与 common_time 逐位一致 |

### 8.2 单一文件与 Excel

- 一次成功 HPLC 请求只新增一个 `hplc_<token>.csv`。
- 响应不含 `xxx_download_url`、`xxx_rows`。
- 主 CSV 每行固定六列，不发生续行或列错位。
- 每个 `XXX` 描述文本小于 32,767 字符。
- 每个 `Intensity` 仍通过现有 Excel-safe 字符数和有效变化检查。
- 主 CSV 补齐 `Label`/`Sample_ID` 后可由 `load_modeling_csv()` 完整读取。

### 8.3 UI 与 Run 摘要

- 两套 HPLC 表单均有 `max=7500`，并调用同一个经过 Node 行为测试的纯校验函数。
- 经典默认终止行不再是 9000。
- 两套页面都只出现一个 HPLC 下载按钮。
- AI 建模五项摘要、样本分组表和类别表文案一致。
- 结果页只有一个训练时间；started/created 两条分支都有测试。
- 用户可见 OOF/Support 文案完成替换，内部 key 未误改。
- summary 的 `test_macro_f1` 覆盖 holdout、external test、CV pooled、空值、损坏和越权作用域。
- 两套训练记录表的列数、空状态、格式化、键盘导航和操作按钮不回归。

## 9. 验证命令

### 9.1 定向验证

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -k 'hplc or modeling_axis' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_run_result_contract_v1.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_result_frontend_contract.py' 'D:\PythonProject\AutoAI\backend\tests\test_task13_ui_contract.py' 'D:\PythonProject\AutoAI\backend\tests\test_frontend_v2_contract.py' -q
node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'
```

### 9.2 完整回归

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'
```

经典前端 `static/index.html` 的内联脚本修改后，先抽取 `<script>` 内容，再运行：

```powershell
node --check --input-type=commonjs
```

不要直接对 HTML 文件运行 `node --check`，也不要为此引入 Playwright 或新依赖。

## 10. 真实流程验收

项目根目录当前存在本地 `data.csv`。实施后按项目规则完成一次轻量传统分类训练闭环，但不得把
`data.csv`、新预处理 CSV、Run 目录、模型或缓存加入 Git。

HPLC 另使用真实或等价受控的 7500 点原始批次完成：

1. 完整范围 1–7500：确认只有一个 CSV，真实时间为 0–50 分钟，读取后 7500 点。
2. 行号 100–4000：确认描述 offset=99、length=3901；API、CSV 回读和强度长度均为 3901。
3. 终止行 7501 和 9000：确认浏览器不发请求；直接 API 请求返回清晰 400，且没有结果文件。
4. 按保留时间范围：确认实际首末时间是固定网格中落入闭区间的点，而不是用户输入值的强行重标。
5. 关闭插值：确认所选原始 X/Y 与轴不一致 warning 保持原行为。
6. 在主 CSV 中补齐 Label/Sample_ID 后重新上传，确认数据摘要的特征数等于实际选择点数。
7. 使用轻量模型完成训练，确认训练记录的测试集 Macro F1 与结果页测试主指标一致；CV 时与 pooled
   test 一致且不取 fold mean。
8. 用浏览器人工检查两套前端的范围提示、中文术语、单一下载按钮和不同 Run 状态下的训练时间。
9. 最后运行 `git status --short`，确认只出现本任务相关代码、测试和文档。

## 11. 风险、回滚和非目标

### 主要风险

- **点数概念混淆：**完整输入固定 7500 点，选择后输出可能少于 7500 点；任何代码或文案把二者
  混成一个值都会让范围选择失效。
- **off-by-one：**7500 个含首尾点只有 7499 个时间间隔；真实时间不能用 `n/7500` 计算。
- **静默截取：**HTML `max` 不是安全边界，必须同时有 JS 和后端配置驱动校验。
- **浮点恢复：**只保存切片首尾再局部 `linspace` 可能出现极小 ULP 差异；
  `linspace-slice-v1` 必须先恢复完整轴再切片。
- **外部脚本兼容：**绕过项目读取器、只认识 `linspace-v1` 的外部脚本需要升级；项目内部继续读取
  历史格式。
- **Intensity 限制：**紧凑 X 描述不能解决所有强度单元格问题；现有失败门禁必须保留。
- **列表性能：**Macro F1 不能通过逐 Run 构造完整结果页投影获取，只读取完整性验证后的必要标量。
- **机械替换：**术语只改用户可见文本，不能全局替换 CSV/API key、模型 ID、文件名和指标字段。

### 明确非目标

- 不改变默认 0–50 分钟、7500 点的 `HplcGridConfig`。
- 不接收 9000 个有效 HPLC 点；9000 是非法终止行或非法文件长度，不做截断或重采样。
- 不删除 `hplc_interpolate` 开关。
- 不改拉曼和旧 `/api/preprocess/chromatography` 兼容接口的数据精度契约。
- 不把 HPLC 强度改成超长 float64 展开数组，不放弃 Excel 单元格保护。
- 不新增 ROC-AUC、ROC 或 Precision-Recall 指标。
- 不修改训练指标的计算，只增加一个现有测试主指标的列表投影。

### 回滚原则

- A（HPLC）和 B（界面/列表）可分两个审查检查点实施；最终一起跑完整回归。
- 若 A 需要回滚，新 `linspace-slice-v1` 的读取能力应保留，避免已生成文件立即不可用。
- 回滚单一文件输出时不得删除已经生成的用户 CSV。
- 不主动删除或移动现有历史计划、数据和 Run 产物。

## 12. 推荐执行顺序

1. A1：先用测试锁定严格范围和实际选择点数。
2. A2：增加真实时间切片描述与历史读取兼容。
3. A3：确认主 CSV 可独立回读后，再删除第二文件。
4. A4：接入两套前端上限、提示和单一下载。
5. 完成 HPLC 定向测试和一次 100–4000 真实/受控流程检查。
6. B1/B2：统一建模和结果页文案。
7. B3：增加 summary 标量和两套训练记录列。
8. C：同步权威文档。
9. 执行全部后端测试、前端测试、语法检查和真实轻量训练闭环。

可选审查检查点：A1–A4 完成后先审查数据契约；B1–B3 完成后再审查用户界面和指标口径。
