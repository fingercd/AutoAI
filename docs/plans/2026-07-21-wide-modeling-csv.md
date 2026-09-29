# 真实 X 轴宽表建模 CSV 迁移计划

> **历史计划归档（原计划日期：2026-07-21；归档标识：2026-09-29）：** 本文保留当时的目标、方案和验收记录，不能据此判断功能已交付或按文中的分支、推送、部署命令操作。当前事实与工作流程以 [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[README.md](../../README.md) 和[文档导航](../README.md)为准。当前主线为 `019f1cc`；服务器唯一开发目录为 `/users/fotile/AutoAI/Pan`，固定 `pan/agent`。不得创建新分支、worktree、fork 或可开发复制；开发用 Git 命令仅在该服务器 Pan 根目录执行。本次整理不提交、推送或部署，GitHub 其他分支保留。

## 目标与成功标准

把预处理导出与训练上传统一迁移到 `wide-feature-v1` 宽表：每条曲线占一行，前三列固定为 `Index, Label, Sample_ID`，其余每一列的列名是一个真实 `XXX` 坐标，单元格是该坐标对应的 `Intensity`。新训练只接受这种宽表，不再接受把 `XXX`、`Intensity` 数组塞进两个单元格的旧六列格式。

成功标准：

- 拉曼、简单色谱和 HPLC 预处理都只生成一个可编辑宽表 CSV。
- HPLC 默认完整范围生成 `3 + 7500` 列；100–4000 范围生成 `3 + 3901` 列，表头坐标可逐点恢复服务端返回的真实分钟轴。
- 上传、主训练集和独立测试集都按同一宽表契约校验；训练矩阵来自特征单元格，解释性分析继续使用表头中的真实 X 轴。
- 任何轴不一致、重复坐标、非数字坐标、非数字强度或主/独立测试集轴不一致都明确拒绝，不能静默取首条轴、重排、补值或插值。
- 两套前端、README、项目上下文和前后端交接文档不再宣传旧六列数组/JSON 格式。
- 快速回归、完整后端测试、前端 Node 测试、Python 编译和经典前端脚本语法检查全部通过。

## 固定契约与关键决策

### CSV 结构

```csv
Index,Label,Sample_ID,0,0.0066675556740898788,...,50
1,A,S001,0.12,0.15,...,0.08
2,A,S001,0.11,0.16,...,0.09
```

- 元数据列及顺序固定为 `Index, Label, Sample_ID`，与目标草图一致。预处理响应的 `curves[].name` 继续保留原文件名，但不写入训练 CSV；训练结果需要名称时使用 `Index` 作为稳定回退。
- 第 4 列开始全部是特征列；列名必须能解析为有限浮点数，并且按列严格递增、数值唯一。
- 真实坐标使用 Python float64 可往返的规范文本（`format(value, ".17g")`）写入表头，不使用点位序号，也不再使用 `linspace-slice-v1` 或数组 JSON。
- 每个特征单元格必须是有限数值；预处理生成的强度仍最多保留 5 位小数，但不再为适配 Excel 单元格字符上限而降低整条数组精度。
- `Label`、`Sample_ID` 必填；同一 `Sample_ID` 只能对应一个 Label，且各 Sample_ID 的重复测量数继续要求一致。
- 生成和加载都限制总列数不超过 Excel 工作表上限 16,384；三个元数据列之外最多 16,381 个特征。当前 HPLC 7500 点在此范围内。

### 公共轴约束

宽表只有一组特征列表头，无法表达“每行不同的 XXX”。因此同一预处理批次内所有输出曲线必须具有逐点一致的公共轴：

- HPLC 开启插值时天然使用同一固定目标轴，直接输出该轴或其范围切片。
- HPLC 关闭插值时，仅当所有所选原始轴完全一致时允许导出；否则返回 400，提示开启插值或先对齐数据。原先的“带警告继续导出”行为不适用于宽表。
- 拉曼和简单色谱多文件同样要求轴一致；不一致时拒绝，不能把第一条曲线的坐标套到其他强度上。
- 独立测试集不仅特征数要相同，坐标值及顺序也必须与主训练集完全一致。

### 兼容性边界

- 新数据上传和 worker 训练加载器只接受 `wide-feature-v1`。检测到旧 `Index, Name, XXX, Intensity, Label, Sample_ID` 格式时返回带迁移说明的明确错误。
- 已完成 Run 的结果和 artifact 不重新解析原 CSV，不受影响。
- 已注册但尚未训练的旧数据集会在新 worker 加载时失败并显示格式错误；本轮不自动迁移，因为旧文件可能为每行保存不同轴，自动取首行会改变数据语义。
- `Repeat_index` 不再作为新宽表字段；新文件必须使用 `Sample_ID`。历史结果读取逻辑不在本次范围内。
- 根目录现有 `data.csv` 是旧六列文件，且多条曲线轴不一致，不能无损转为单一真实轴宽表。本轮只验证它被清晰拒绝，不改写、不提交该文件，也不以伪造公共轴完成真实训练。

## 影响范围

- `backend/app/parsers.py`
  - 定义宽表元数据、表头解析、公共轴校验、宽表构造和训练数据加载。
  - 移除主路径对数组 JSON、轴描述和 Excel 单元格字符限制的依赖。
  - 摘要预览只返回元数据列，曲线预览仍返回真实 `x/y`；名称显示回退为 `Index`。
- `backend/app/hplc.py`
  - 直接以目标/原始公共轴构造宽表。
  - 关闭插值且轴不一致时拒绝导出。
  - 将轴编码元数据改为 `wide-feature-v1` / `column_headers`。
- `backend/app/routers/preprocess.py`
  - 宽表预览只选择四个元数据列，不再删除不存在的 `XXX/Intensity` 列。
- `backend/app/training.py`
  - 独立测试集增加真实坐标逐点一致校验。
- `backend/tests/modeling_data_factory.py`
  - 所有共享训练测试数据改用宽表生成器。
- `backend/tests/test_smoke.py`
  - 更新基础解析、预处理、API、训练闭环和错误分支。
- `backend/tests/test_hplc_axis_contract.py`
  - 将描述符测试替换为宽表真实表头、部分范围和不一致轴拒绝测试。
- 其他直接手写六列 CSV 的专项测试
  - 改用共享宽表工厂，或在专门的旧格式拒绝用例中保留旧文件。
- `static/index.html`
  - 更新经典前端操作说明、输出精度和 HPLC 轴存储文案。
- `static/v2/views/modeling.js`、`static/v2/views/manual.js`、`static/v2/views/workbench.js`
  - 更新上传提示、格式说明和预处理结果文案。
- `backend/tests/test_frontend_v2_contract.py`、`backend/tests/test_task13_ui_contract.py`
  - 更新前端字符串契约。
- `README.md`、`CONTEXT.md`、`docs/frontend_backend_handoff.md`
  - 将权威契约更新为宽表，并记录公共轴、Excel 列数和迁移边界。

## 实施任务

### 任务 1：建立宽表构造与加载核心

**实施内容：**

1. 在 `parsers.py` 定义 `MODELING_METADATA_COLUMNS = ("Index", "Label", "Sample_ID")`、格式版本和 Excel 最大列数。
2. 新增公共轴规范化：校验一维、非空、有限、严格递增；以 `.17g` 生成可往返且不重复的列名。
3. 新增批次公共轴校验；报告首个不一致文件及坐标位置，禁止静默使用首行。
4. 新增宽表构造器：三列元数据与二维强度矩阵一次性拼接，强度固定最多 5 位，返回格式/精度/列数元数据。
5. 为建模上传单独实现保留字符串元数据和原始表头的灵活编码/分隔符读取；在 pandas 重命名重复列前检测原始重复表头。
6. 重写 `load_modeling_csv()`：严格检查前三列、解析数值表头、转换二维强度、校验空值/非有限值和 Sample_ID 规则，继续返回现有 `ModelingDataset`，从而把模型算法改动控制为零。
7. `summarize_modeling_csv()` 使用三列元数据做预览，以 `Index` 作为曲线名称回退，并标记 `data_format=wide-feature-v1`。

**验收标准：**

- 合法宽表能无损得到二维强度矩阵和公共真实轴。
- 旧六列数组格式得到明确迁移错误。
- 重复/非数字/无穷/非递增表头、空白或非数字强度、错误元数据顺序均被拒绝。
- `Sample_ID` 的前导零在读取后保留，摘要和解释性元数据在没有 `Name` 时使用 `Index` 显示。

### 任务 2：迁移三条预处理导出路径

**实施内容：**

1. 拉曼和简单色谱改用宽表构造器；曲线预览与落盘强度来自同一量化结果。
2. HPLC 开启插值时将 `selection.target_x` 直接写入表头；范围切片点数保持现有业务规则。
3. HPLC 关闭插值时执行严格公共轴检查；多文件不一致直接报错并说明开启插值。
4. API 仍只原子生成一个主 CSV；预览只返回 `Index/Label/Sample_ID`，文件名继续通过 `curves[].name` 提供。
5. 将 `output_precision` 改为宽表语义：格式版本、轴存储位置、float64 往返、强度位数、特征数、总列数和 Excel 列数上限。

**验收标准：**

- 下载文件前三列固定，后续表头与 `curves[].x` 一致，每行特征值与 `curves[].processed_y/corrected_y/raw_y` 一致。
- HPLC 7500 点和 3901 点切片均可由下载文件直接回读训练。
- 轴不一致请求不留下部分输出文件。

### 任务 3：收紧训练集与独立测试集契约

**实施内容：**

1. 保持训练、归一化、模型与解释性算法继续消费 `ModelingDataset.intensity` 和 `x_axis`。
2. 在独立测试集校验中同时比较特征数和真实表头坐标；给出首个差异位置和两侧坐标。
3. 公共轴宽表成功加载后，训练结果中的 `x_axis_warning` 应稳定为 consistent。

**验收标准：**

- 主数据集和独立测试集只要坐标有一处不同就不能创建有效训练结果。
- 合法宽表能用轻量传统模型完成训练、评估和解释性产物闭环。

### 任务 4：更新两套前端与接口文档

**实施内容：**

1. 经典前端和 v2 将“固定六列”改为“前三列元数据 + 真实 XXX 特征列”。
2. HPLC 结果文案改为“真实时间坐标保存在特征列表头”，删除 JSON 描述/Excel 单元格字符限制说明。
3. 上传失败提示列出固定前三列，并提示第 4 列起必须是严格递增的数值坐标。
4. README、CONTEXT 和交接文档记录格式示例、公共轴规则、旧格式拒绝和 Excel 16,384 列限制。

**验收标准：**

- 用户界面与权威文档中不再把 `XXX/Intensity` 描述为两个数组单元格，也不再声称 HPLC 使用 `linspace-slice-v1`。
- 下载入口仍只有一个，不新增轴文件。

### 任务 5：重建契约测试并执行验证

**实施内容：**

1. 更新共享建模数据工厂和所有训练测试 fixture 为宽表。
2. 增加宽表解析正反例、原始重复表头、前导零 Sample_ID、公共轴不一致、HPLC 完整/部分范围以及独立测试轴不一致测试。
3. 更新 API 下载断言与前端字符串契约。
4. 运行定向测试后运行完整后端测试、Node 测试、compileall 与经典前端内联脚本语法检查。
5. 对根目录现有 `data.csv` 只运行新加载器拒绝检查，记录其不能完成新格式真实训练的原因；不得自动覆盖。

**验证命令：**

```powershell
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_hplc_axis_contract.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'
```

经典前端另抽取 `static/index.html` 的内联 `<script>` 到 `work/`，再执行：

```powershell
node --check --input-type=commonjs '< extracted-script.js'
```

## 风险与回滚说明

- 这是刻意的不兼容数据契约迁移；回滚代码即可恢复旧加载器，但新宽表与旧加载器互不兼容。
- 宽表解决单元格 32,767 字符问题，但把限制转为 Excel 16,384 列；超过 16,381 个特征必须先做业务认可的截取/降采样，不能由导出器静默处理。
- `.17g` 表头可能较长，但可确保 HPLC float64 固定轴逐点往返；界面显示仍可用短格式，不影响 CSV 真值。
- 多曲线轴不一致会从“警告继续”变成“明确失败”。这是保证每个强度值不被绑定到错误坐标的必要约束。
- 本轮不改任何模型架构、搜索空间、划分策略或结果 artifact 契约。
