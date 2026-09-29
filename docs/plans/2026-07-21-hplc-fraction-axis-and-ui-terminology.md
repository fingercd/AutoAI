# HPLC 分数坐标、单一 CSV 与界面术语统一实施计划

> **历史计划归档（原计划日期：2026-07-21；归档标识：2026-09-29）：** 本文保留当时的目标、方案和验收记录，不能据此判断功能已交付或按文中的分支、推送、部署命令操作。当前事实与工作流程以 [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[README.md](../../README.md) 和[文档导航](../README.md)为准。当前主线为 `019f1cc`；服务器唯一开发目录为 `/users/fotile/AutoAI/Pan`，固定 `pan/agent`。不得创建新分支、worktree、fork 或可开发复制；开发用 Git 命令仅在该服务器 Pan 根目录执行。本次整理不提交、推送或部署，GitHub 其他分支保留。

## 1. 目标与推荐结论

本计划覆盖两个可独立审查、最终一起回归的改动簇：

1. HPLC 预处理只生成一个六列整理 CSV，不再额外生成 `_xxx.csv`；默认插值模式下，
   每条曲线的 `XXX` 改为按原始点序号定义的归一化分数坐标。
2. 落实 `0720修改.txt` 中的 AI 建模、建模结果和训练记录用词调整，并在训练记录增加
   “测试集 Macro F1”列。

推荐把 HPLC 第 `n` 个点定义为 `n / point_count`。默认 `point_count=7500`，因此：

- 完整范围是 `1/7500, 2/7500, ... , 7500/7500`；
- 行号范围 100–4000 按首尾都包含计算，共 3901 点；
- 对应坐标是 `100/7500, 101/7500, ... , 4000/7500`；
- 选择范围后分母仍是完整网格点数 7500，不能改成所选点数 3901，也不能从 1 重新编号。

这里的分数坐标是“归一化点位置”，不是原仪器 X 值，也不是 0–50 分钟的保留时间。
当前 HPLC 强度仍需先在内部 0–50 分钟固定轴上完成 float64 线性映射；映射完成后，
整理 CSV 和建模读取器使用 `n/7500` 作为输出 X 坐标。这样不会改变强度插值算法，只改变
整理 CSV、预览和下游解释图使用的 X 坐标语义。

> 若业务真正需要的是物理分钟坐标，则第 `n` 点应为
> `(n-1) * 50 / 7499`，而不是 `n/7500`。本计划按用户给出的 `n/7500` 规则执行，
> 不把两套坐标混称为“分钟”。

## 2. 为什么不能逐项写字面分数

统一建模 CSV 继续保持“一条曲线一行”，`XXX` 和 `Intensity` 位于各自单元格中。
如果把每个分数写成字符串数组：

```json
["1/7500","2/7500","3/7500"]
```

会出现三个问题：

- 完整 7500 点约 88,894 个字符，超过 Excel 单元格 32,767 字符上限；
- 100–4000 的 3901 点也约 45,913 个字符，仍超过上限；
- `1/7500` 不是合法 JSON 数值，现有建模读取器不能把它直接当数值数组使用。

因此新文件在 `XXX` 中写一条短小、精确且可逆的分数区间描述：

```json
{"type":"fraction-range-v1","numerator_start":100,"numerator_stop":4000,"denominator":7500}
```

`load_modeling_csv()` 读取时将它展开为：

```text
100/7500, 101/7500, ..., 4000/7500
```

分数关系在 CSV 中无损保存；进入 NumPy、预览或训练后，按既有数值计算约定转换成
float64。`1/7500` 这类值无法被二进制浮点绝对精确表示，但每次都会用相同的
“整数分子 ÷ 整数分母”规则确定性恢复。

`0720修改.txt` 中“色谱数据从 1–9000”的表述需要区分两种情况：

- 若 1–9000 是原始 X 的数值范围，而文件仍有配置要求的 7500 个有效点，新分数描述会让
  `XXX` 不再因展开坐标过长触发 Excel 单元格上限；
- 若文件实际含 9000 个有效点，当前 HPLC 固定流程仍会按既有规则拒绝，因为它要求
  `HplcGridConfig.point_count=7500`。本计划不把 9000 点静默截断、降采样或强行映射成 7500 点。

如果业务后续确认必须接收 9000 个有效点，应另行修改固定网格配置和真实数据验收标准，
不能只靠更换 `XXX` 的文本表示绕过点数校验。

## 3. 范围、插值和兼容边界

### 3.1 行号范围

行号继续使用 1 基、首尾包含语义。后端必须从固定完整轴中保留全局位置编号，不能用
截取后数组的局部下标重新编号。

| 请求 | 分子起点 | 分子终点 | 点数 |
| --- | ---: | ---: | ---: |
| 全部 | 1 | 7500 | 7500 |
| 1–4000 | 1 | 4000 | 4000 |
| 100–4000 | 100 | 4000 | 3901 |
| 7500–7500 | 7500 | 7500 | 1 |

现有插值模式要求至少 2 点，因此单点范围在插值开启时仍应返回可理解的 400；
分数描述解析器本身可支持单点，以便兼容其他读取场景，但 HPLC 业务层继续执行至少 2 点门禁。

### 3.2 按 X 值范围

HPLC 表单中的“按 X 值范围”继续按物理保留时间筛选，因为原文件和内部映射轴仍以分钟表示。
筛选得到固定轴上的连续全局下标后，再把这些下标转换为分数坐标。例如筛选结果落在固定轴
第 100–4000 点，输出描述仍是 `100/7500` 到 `4000/7500`。

前端应把 HPLC 的输入标签改成“保留时间下限/上限（分钟）”，避免用户误以为这里输入的是
0–1 分数坐标。

### 3.3 插值开关

项目当前契约要求保留 `hplc_interpolate`：

- 开启：使用本计划的新 `fraction-range-v1` 输出坐标；这是主页面默认路径。
- 关闭：继续导出所选原始 X/Y，`XXX` 保持原始数值数组，并继续提示多文件轴不一致。

这里把“所有 `XXX`”解释为默认 HPLC 整理 CSV 中所有曲线行的 `XXX`，不擅自破坏
“关闭插值即保留原始 X/Y”的已有功能。如果后续要求关闭插值时也强制使用分数坐标，
需要同时重新定义该开关的业务意义，不能只改序列化文本。

### 3.4 强度精度

本计划不修改 `Intensity` 的现行 Excel-safe 批次统一精度规则，也不改变线性映射、边界延伸、
消负或面积归一化行为。已有 `2026-07-20-hplc-single-lossless-csv.md` 草稿提出了不同的
float64 强度输出方案；该草稿与本计划不是同一个已确认需求，执行本计划时不得顺带引入。

拉曼和旧 `/api/preprocess/chromatography` 兼容接口也不改坐标或精度契约。

## 4. 成功标准

### HPLC 输出

- 一次 HPLC 预处理只落盘一个 `hplc_<token>.csv`。
- API 只返回主文件 `download_url`，不再返回 `xxx_download_url`、`xxx_rows`。
- 默认完整范围每行 `XXX` 都是同一条 `fraction-range-v1` 描述，展开后恰好 7500 点。
- 100–4000 行展开后恰好 3901 点，首值为 `100/7500`，末值为 `4000/7500`。
- 展开后的 `XXX` 与同一行 `Intensity` 长度严格一致，并严格递增。
- API 的曲线预览 X 值与 CSV 经读取器展开后的 X 值一致。
- 内部强度映射仍使用物理分钟轴，不因输出改成 0–1 分数坐标而改变逐点强度。
- 旧数值数组和旧 `linspace-v1` HPLC 文件仍可读取，无需迁移历史 CSV。
- `hplc_interpolate=false` 行为保持原样。

### 界面和训练记录

- 两套前端对相同业务概念使用同一中文名称。
- AI 建模摘要显示：数据量、类别数、样本数、每样本测量数、特征数。
- 分组标题为“按样本分组”，表头为：样本编号、类别、每样本测量数。
- “标签分布”改为“类别分布”，表头为：类别、数据量。
- 成功加载模型目录后不再显示“模型目录已加载；模型按文档分类分组显示。”。
- 建模结果把“曲线数”改为“数据量”，“Sample_ID 数”改为“样本数”。
- 创建时间和开始时间合并成一个“训练时间”：优先显示 `started_at`；尚未开始时显示
  `created_at` 并明确标注“任务创建”。
- 所有用户可见的 OOF 文案改成“交叉验证”相关中文；内部字段
  `pooled_oof`、文件名和计算口径不改。
- 各类别指标中的“样本量”和 `Support` 统一显示为“数据量”。
- 训练记录新增“测试集 Macro F1”，成功任务显示测试主口径值，其余状态或缺失旧产物显示 `—`。

## 5. 术语映射表

| 当前可见文案 | 目标文案 | 备注 |
| --- | --- | --- |
| 总数据 / 曲线数（行） | 数据量 | 对应 CSV 曲线行数 |
| 类别数 | 类别数 | 保持 |
| Sample_ID 组数 | 样本数 | 技术字段名仍为 `Sample_ID` |
| 每组测量数 | 每样本测量数 | 对应 `expected_repeats_per_group` |
| 曲线长度 | 特征数 | 对应 `curve_length` |
| Sample_ID 分组 | 按样本分组 | 标题 |
| Sample_ID | 样本编号 | 只改业务表格；CSV 字段名不改 |
| Label | 类别 | 只改业务表格；CSV 字段名不改 |
| 数量（分组表） | 每样本测量数 | 分组内重复曲线数 |
| 标签分布 | 类别分布 | 标题 |
| Count | 数据量 | 类别曲线数 |
| 曲线数（结果页） | 数据量 | 对应 `dataset.curve_count` |
| Sample_ID 数（结果页） | 样本数 | 对应 `dataset.sample_id_count` |
| 创建时间 + 开始时间 | 训练时间 | 只显示一个值，开始优先 |
| 合并 OOF 预测 | 合并交叉验证预测 | API 枚举仍为 `pooled_oof` |
| Support / 样本量 | 数据量 | 分类报告数值不变 |

`Run ID`、`Sample_ID`/`Label` 的 CSV 字段名、模型 ID、Precision、Recall、F1 等正式标识或
标准指标名不做机械替换；只在用户明确要求的业务标题和说明中使用统一中文，避免破坏接口和文件契约。

## 6. 接口和数据契约

### 6.1 `XXX` 新描述类型

在 `backend/app/parsers.py` 的 `_parse_modeling_axis()` 中新增严格白名单分支：

```json
{
  "type": "fraction-range-v1",
  "numerator_start": 100,
  "numerator_stop": 4000,
  "denominator": 7500
}
```

校验要求：

- 三个参数都必须是真整数，布尔值不得冒充整数；
- `1 <= numerator_start <= numerator_stop <= denominator`；
- 展开点数为 `numerator_stop - numerator_start + 1`，不得超过
  `MAX_AXIS_DESCRIPTOR_POINTS`；
- 分母必须为正，并设置合理上限，防止恶意描述触发异常内存或计算；
- 使用 `np.arange(start, stop + 1, dtype=np.float64) / denominator` 展开，
  不使用会改变逐点除法语义的二次 `linspace`；
- 展开后检查有限值、长度和严格递增。

旧 `linspace-v1` 和普通数组分支完整保留。`Intensity` 仍只接受非空数值数组，禁止把
分数描述误用到强度字段。

### 6.2 HPLC 响应元数据

`hplc_axis` 应把“输出坐标”和“强度映射轴”明确分开，避免继续把 0–1 坐标标成分钟：

```json
{
  "point_count": 3901,
  "mapping": "piecewise_linear",
  "output_axis": {
    "encoding": "fraction-range-v1",
    "numerator_start": 100,
    "numerator_stop": 4000,
    "denominator": 7500,
    "unit": "normalized_position"
  },
  "interpolation_axis": {
    "start": 0.6600880117,
    "stop": 26.6635551407,
    "step_minutes": 0.0066675557,
    "unit": "minute"
  },
  "input_point_count_required": 7500
}
```

示例小数只用于说明结构，实际值必须直接来自本次选中的物理目标轴，不能手写常量。
`curves[].x` 使用展开后的分数数值，供预览与最终 CSV 回读保持一致；现有
`common_time` 可在兼容期继续表示内部物理分钟轴，但文档必须明确它不是 CSV 的 `XXX`。

`output_precision` 对默认插值路径使用：

- `xxx_encoding="fraction-range-v1"`；
- `xxx_max_characters` 为最终描述文本实际长度；
- `intensity_decimal_places`、`intensity_max_characters` 和 Excel 上限保持现行含义；
- 不再把新输出描述为 `linspace-v1` 或“固定分钟轴写入 XXX”。

### 6.3 Run 摘要接口

`GET /api/training/runs?projection=summary` 为每项增加一个可空标量：

```json
{
  "test_macro_f1": 0.9234
}
```

取值规则必须与结果页主口径一致：

1. 成功任务优先读取通过 Manifest 完整性检查的 `metrics.json`；
2. 优先取 `metrics.test.macro_f1`，旧格式才回退到顶层 `metrics.macro_f1`；
3. 留一交叉验证取 pooled test 主值，禁止取 fold mean；
4. 非成功状态、缺失/损坏产物、非有限数值返回 `null`；
5. 摘要接口只增加该标量，不返回完整 `metrics`、配置或服务器路径。

历史 `status.json` 的兼容回退只在其结构明确且数值有限时使用，不能猜测或把 valid F1
当成 test F1。

## 7. 分任务实施

### 任务 A1：增加分数轴构造、描述和读取能力

**涉及文件：**

- 修改：`backend/app/hplc.py`
- 修改：`backend/app/parsers.py`
- 测试：`backend/tests/test_smoke.py`

**实施内容：**

1. 在 `hplc.py` 增加从选中固定轴位置计算全局 1 基分子的纯函数。
2. 分母一律来自注入的 `HplcGridConfig.point_count`，默认才是 7500；禁止在算法函数体内散落硬编码。
3. 行号和 X 值布尔掩码都先转换为完整目标轴上的全局下标，验证结果连续。
4. 增加 `_fraction_axis_descriptor()` 和输出数值轴构造函数。
5. 在 `_parse_modeling_axis()` 中增加 `fraction-range-v1` 严格解析；保留旧数组和 `linspace-v1`。
6. 对畸形类型、布尔参数、倒序、0 分母、越界和超大展开分别返回带行号的清晰错误。

**验收：**

- 完整范围展开为 7500 点，首末值分别等于 `1/7500` 和 `1.0`。
- 100–4000 展开为 3901 点，且与逐个整数除以 7500 的预期数组逐点一致。
- 旧 HPLC 文件和普通数组建模文件仍能读取。

### 任务 A2：让默认 HPLC 输出使用分数轴，但保持分钟轴插值

**涉及文件：**

- 修改：`backend/app/hplc.py`
- 测试：`backend/tests/test_smoke.py`

**实施内容：**

1. 保留 `target_time_minutes` 作为 `map_hplc_intensity()` 的唯一目标轴。
2. 从同一选中下标构造 `output_fraction_x`，只用于 CSV、API 曲线预览和建模 X 轴。
3. `XXX` 写入同一条可逆分数描述；`Intensity` 继续使用当前批次统一精度序列化结果。
4. 用最终读取器回读的分数数组构造 `curves[].x`，防止预览与下载文件不一致。
5. 更新 `hplc_axis` 与 `output_precision`，同时保留内部分钟轴审计信息。
6. 明确隔离 `interpolate=false` 分支，避免意外改变原始 X/Y 导出。

**验收：**

- 同一请求改动前后的映射强度逐点一致；变化只发生在输出 X 坐标。
- `len(curves[i].x) == len(curves[i].processed_y)`。
- 100–4000 的 API 预览和 CSV 回读首末 X 坐标一致。

### 任务 A3：删除第二个 HPLC CSV 和响应字段

**涉及文件：**

- 修改：`backend/app/routers/preprocess.py`
- 测试：`backend/tests/test_smoke.py`

**实施内容：**

1. 删除 `_hplc_visible_axis_frame()`、相关 pandas 依赖和 `_xxx.csv` 写盘逻辑。
2. 删除 `xxx_download_url`、`xxx_rows` 响应字段。
3. 保留主文件 `download_url`、预览、强度摘要、警告和 HPLC 坐标元数据。
4. 测试中把预处理目录重定向到临时目录，断言一次成功请求只新增一个对应 token 的 CSV。
5. 失败请求继续保证不生成主文件或半成品。

**验收：**

- 页面和 API 都只有一个 HPLC 下载入口。
- 预处理目录不存在同 token 的 `_xxx.csv`。
- 下载主 CSV 后即可由 `load_modeling_csv()` 恢复全部 X 坐标。

### 任务 A4：更新两套预处理前端

**涉及文件：**

- 修改：`static/index.html`
- 修改：`static/v2/views/workbench.js`
- 修改：`static/v2/views/manual.js`
- 测试：`backend/tests/test_task13_ui_contract.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`

**实施内容：**

1. 删除“下载可见 XXX 时间轴”按钮、事件处理函数及 `xxx_download_url` 判断。
2. HPLC 默认插值结果只显示“下载整理 CSV/统一建模 CSV”。
3. 结果说明改为“输出 X 坐标：第 n 点/7500”；范围截取时显示实际分子起止和点数。
4. 曲线图 X 轴标签改为“归一化 X 坐标”，不要继续写“保留时间”。
5. HPLC 按值筛选输入明确写“保留时间（分钟）”，与输出坐标区分。
6. `hplc_interpolate=false` 时仍显示“原始 X 轴”，不套用分数轴文案。

**验收：**

- 两个前端都只出现一个下载按钮。
- 默认完整范围显示 `1/7500–7500/7500，共 7500 点`；100–4000 显示对应 3901 点。
- 页面不再把新 `XXX` 说成 `linspace-v1` 或分钟轴。

### 任务 B1：统一 AI 建模摘要与分组/类别表文案

**涉及文件：**

- 修改：`static/index.html`
- 修改：`static/js/ui-utils.js`
- 修改：`static/v2/views/modeling.js`
- 测试：`backend/tests/test_smoke.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`

**实施内容：**

1. 两套建模摘要统一为五项固定顺序：数据量、类别数、样本数、每样本测量数、特征数。
2. v2 使用现有 `summary.sample_id.expected_repeats_per_group` 补齐“每样本测量数”。
3. 经典前端分组标题和共享表头按术语映射表修改。
4. 类别分布标题和列名改为“类别 / 数据量”。
5. 模型目录加载成功时清空或隐藏提示区域；目录失败、刷新失败和不可用原因仍保留。
6. 不改后端摘要字段名，不改 CSV 的 `Label`、`Sample_ID` 列名。

**验收：**

- 训练集和独立测试集摘要使用相同五项名称与顺序。
- 大量样本折叠/展开、自然排序和无障碍标签仍正常。
- 成功状态没有冗余模型目录提示，错误状态仍可诊断。

### 任务 B2：统一建模结果文案和时间字段

**涉及文件：**

- 修改：`static/js/run-results.js`
- 修改：`static/v2/views/result.js`
- 修改：`static/v2/lib/format.js`
- 修改：`backend/app/runs/artifacts.py`（仅用户可见 artifact 标签）
- 测试：`backend/tests/test_result_frontend_contract.py`
- 测试：`backend/tests/test_task13_ui_contract.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`

**实施内容：**

1. 结果概览和 v2 数据集卡把曲线数/Sample_ID 数改为数据量/样本数。
2. 抽出统一的训练时间格式函数：`started_at` 优先；没有开始时间时显示创建时间并加“任务创建”后缀。
3. 删除同一卡片中重复的创建/开始时间项；结束时间和耗时保留。
4. 只替换用户可见 OOF 文案，内部枚举 `pooled_oof`、计算和 JSON 字段不改。
5. 经典分类报告表头 `Support` 改为“数据量”，相邻说明的“样本量”同步修改。
6. v2 中同义的“真实/预测样本数量”改为“真实/预测数据量”；不凭空增加不存在的分类报告表。

**验收：**

- CV Test 仍读取 pooled 主指标，文案变化不改变口径。
- 页面搜索不到用户可见的 `OOF` 和 `Support`，但源码中的契约枚举仍存在。
- 每个结果状态区域只显示一个训练时间字段。

### 任务 B3：为训练记录提供测试集 Macro F1

**涉及文件：**

- 修改：`backend/app/routers/runs.py`
- 修改：`static/index.html`
- 修改：`static/v2/components/run-list.js`
- 测试：`backend/tests/test_run_result_contract_v1.py`
- 测试：`backend/tests/test_smoke.py`
- 测试：`backend/tests/test_frontend_v2_contract.py`

**实施内容：**

1. 增加只解析测试主 Macro F1 的小型后端辅助函数，复用 Manifest 完整性结论，避免为列表页构造完整结果投影。
2. `_summary_projection()` 增加可空 `test_macro_f1`，不增加完整指标对象。
3. 覆盖 holdout、external test、CV pooled test、旧格式回退、损坏/缺失和活跃任务。
4. 经典训练记录在“状态”前增加“测试集 Macro F1”；空值显示 `—`，有效值固定显示 4 位小数。
5. v2 `run-list.js` 同步增加该列，并复用 `formatMetric()`。
6. 更新空表 `colSpan`、响应式布局和键盘行导航断言。

**验收：**

- 成功 CV Run 列表值等于结果页 pooled test Macro F1，不等于 fold mean。
- queued/running/failed/cancelled 和无指标旧 Run 显示 `—`。
- summary API 仍不含完整 `metrics`、`config` 或服务器路径。

### 任务 C：同步权威文档

**涉及文件：**

- 修改：`AGENTS.md`
- 修改：`CONTEXT.md`
- 修改：`README.md`
- 修改：`docs/frontend_backend_handoff.md`

**实施内容：**

1. 记录默认 HPLC CSV 的 `fraction-range-v1` 规则、1 基含首尾范围和单文件输出。
2. 明确内部分钟插值轴与输出分数坐标是两套用途不同但逐点对齐的轴。
3. 删除新导出仍使用 `linspace-v1`、仍生成 `xxx_download_url` 的陈旧说明；保留历史读取兼容说明。
4. 记录训练记录 summary 的 `test_macro_f1` 可空字段及 CV pooled 口径。
5. 文档中的技术术语 OOF 可以在首次出现时保留为“交叉验证折外预测（OOF）”，
   但 UI 截图、操作说明和用户可见文案使用“交叉验证”。

## 8. 测试矩阵

### 8.1 分数描述解析

- 完整 `1..7500 / 7500`。
- 范围 `100..4000 / 7500`。
- 自定义 `HplcGridConfig(point_count=5)` 生成 `1/5..5/5`，证明没有硬编码 7500。
- 单点描述可解析，但插值业务层拒绝单点范围。
- 非整数、布尔、负数、0 分母、起点大于终点、终点大于分母、超大展开全部拒绝。
- 普通数组与 `linspace-v1` 兼容回归。

### 8.2 HPLC 管线

- 完整范围：CSV 描述、预览分数轴、物理分钟映射强度逐点对应。
- 1–4000 与 100–4000 行号边界和点数。
- X 值范围先在分钟轴选择，再输出正确全局分子。
- 多文件共享同一描述，Intensity 长度一致。
- 边界一个采样间隔内延伸及超限拒绝保持原行为。
- `hplc_interpolate=false` 原始轴与不一致警告保持原行为。
- API 只生成一个文件，且响应不含第二下载字段。

### 8.3 UI 与 Run 摘要

- 两套建模摘要五项字段、顺序和空值行为一致。
- 经典样本分组表、类别分布表的标题和列名准确。
- 结果页只有一个训练时间，且 started/created 两个分支都有覆盖。
- 所有可见 OOF/Support 文案完成替换，内部契约 key 未被误改。
- summary 的 `test_macro_f1` 对 holdout、CV pooled、空值和损坏产物行为正确。
- 两套训练记录表的列数、空状态 colspan、格式化和操作按钮不回归。

## 9. 验证命令

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'
```

经典前端 `static/index.html` 的内联脚本修改后，抽取 `<script>` 内容，再运行：

```powershell
node --check --input-type=commonjs
```

不要直接对 HTML 文件运行 `node --check`，也不要为此引入 Playwright 或新的前端依赖。

### 真实流程验证

项目根目录当前存在本地 `data.csv`，实施后必须完成一次真实轻量分类训练闭环，但不得把该文件或
Run 产物加入 Git。HPLC 部分另使用一个实际或受控生成的 7500 点原始批次验证：

1. 请求完整范围，确认只得到一个 CSV；
2. 请求 100–4000，确认描述、3901 点、预览和强度长度；
3. 补齐 Label/Sample_ID 后上传，确认加载器展开的首末 X 和特征数；
4. 使用轻量传统模型完成训练；
5. 确认训练记录测试集 Macro F1 与结果页测试主指标一致；
6. 检查 `git status --short`，确保数据、缓存、模型和 `storage/` 产物未进入 Git。

## 10. 风险、回滚与交付边界

- **坐标语义变化：** 新 Run 的解释图 X 轴从分钟变为 0–1 归一化点位置；这是真实业务变化，
  必须在文档和前端标清，不能继续显示“分钟”。
- **历史兼容：** 只增加读取类型，不重写历史文件；旧 `linspace-v1` 仍恢复分钟轴。
- **外部脚本兼容：** 绕过项目读取器且假设 `XXX` 必须是数组的脚本需要升级。
- **插值关闭模式：** 保持原始 X/Y 是有意兼容边界；不要无意套用分数描述。
- **摘要性能：** 列表页只读取一个经过完整性确认的小型指标标量；禁止逐 Run 调用完整结果投影。
- **指标口径：** CV 列表值必须是 pooled test，不能使用逐折 Macro F1 平均值。
- **文案替换风险：** 只改用户可见文本，不全局替换 API key、CSV 列名或 artifact 文件名。
- **回滚：** A、B 两个改动簇可分别回滚；A 回滚后新 `fraction-range-v1` 文件仍需保留读取能力，
  避免已经生成的新 CSV 立刻不可用。

推荐分两个审查检查点实施：先完成 A1–A4 的数据契约与单文件输出，再完成 B1–B3 和文档；
最终必须运行完整后端测试、两套前端契约测试和真实轻量训练闭环。
