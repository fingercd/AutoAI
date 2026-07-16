# SpecAutoAI 需求 1–6、8 修改方案

> 最终范围：解决原需求 1、2、3、4、5、6、8；明确不实施原需求 7“显示模型架构”。

## 一、目标与成功标准

在不改变现有 FastAPI + 静态前端 + SQLite Run/worker 主架构的前提下，完成品牌更名、`Sample_ID` 字段迁移、模型源码清理、预处理 Excel 上限与数值精度保护、数据集自然排序、交叉验证指标修正和训练记录改版。

完成后应满足：

- 正式页面、FastAPI 标题、启动输出和当前维护文档使用 `SpecAutoAI`。
- 新 CSV、新 API 响应、新 Run 和新 artifact 使用 `Sample_ID`，旧数据只通过受控兼容入口读取。
- `backend/app/models` 中每个正式模型只保留一个最终实现文件，不再同时存在旧版和 `*_v2.py`。
- 预处理 CSV 的 `XXX`、`Intensity` 最多保留小数点后 5 位；单元格实际文本超过 32,767 字符时拒绝生成并给出明确提示。
- AI 建模数据摘要按自然顺序显示 `1, 2, 3, ... 10, 11`。
- 交叉验证的 Test macro F1、macro Precision、macro Recall 来自合并后的 out-of-fold 测试预测；Train/Valid/Test 三组指标都能在当前训练结果中看到。
- 训练记录显示上传数据集名和 Train/Valid/Test macro F1，不再显示 Run ID、数据量及训练/测试 Accuracy。
- 不新增模型架构 UI、架构文本 artifact 或相关元数据。

## 二、实施决策与兼容边界

### 1. 品牌更名边界

本轮将用户可见品牌、应用标题、启动输出、当前维护文档和对应测试从 `AutoAI` 改为 `SpecAutoAI`。为避免部署中断，本轮不移动 `D:\PythonProject\AutoAI` 工作目录，不重命名现有 API 路径、数据库、storage 目录、服务器作业名及现有部署脚本文件名；这些运行标识不属于老师看到的产品品牌。

### 2. Sample_ID 迁移边界

`Sample_ID` 是新系统唯一规范字段。新导出、UI、接口、配置和 artifact 不再写 `Repeat_index`。为了让现有 CSV 和历史 Run 仍能打开，只在单独兼容/迁移逻辑中识别旧列名与旧策略名，并立即归一化为新名称；正式业务路径不继续传播旧字段。

新评估策略名使用 `leave_one_sample_id_cv`。历史值 `leave_one_repeat_index_cv` 只在读兼容映射中转换，不作为新 Run 的输出。

### 3. 5 位精度定义

默认按老师原话实现为“小数点后最多 5 位，不补尾零”，即在数组 JSON 序列化前执行等价于 `round(value, 5)` 的量化，并把 `-0.0` 规范为 `0.0`。这与“5 位有效数字”不同；本方案选择前者。

由于 HPLC 面积归一化可能产生很小的有效数值，增加量化失真保护：原数组有有效变化、量化后却全零或完全失去变化时，拒绝导出并提示 5 位小数不足，避免静默生成不能建模的数据。

### 4. 指标口径

页面中的 Precision 和 Recall 统一明确标为 `Macro Precision`、`Macro Recall`，即每个类别等权；与 `Macro F1` 使用同一类别集合和 `zero_division=0` 规则。

- `stratified_holdout`、`external_test_holdout`：直接对各自 Train/Valid/Test 预测计算指标。
- `leave_one_sample_id_cv`：
  - Test 主指标：合并所有折的 out-of-fold `y_true/y_pred` 后统一计算一次。
  - Train、Valid 主指标：各折对应指标的算术平均；标准差保存在 `cv_summary.fold_std`。
  - 每折 Test 指标仍保存在 `cv_summary.fold_mean`、`fold_std` 和 `fold_metrics.csv`，但不再冒充最终 Test 指标。
- 传统模型继续沿用既有算法：Train/Valid 指标来自只用 Train 拟合并在 Valid 选参的模型；参数锁定后以 Train+Valid 重训，Test 指标来自重训模型。本轮只修正汇总和展示，不改变选参/重训数学流程。

## 三、目标文件与职责

| 模块 | 主要职责 |
|---|---|
| `backend/app/parsers.py` | `Sample_ID` CSV 契约、旧列兼容、自然排序、5 位量化、Excel 单元格上限 |
| `backend/app/hplc.py` | HPLC 新字段输出及统一数组序列化 |
| `backend/app/training.py` | Sample_ID 分组划分、OOF 指标、三组指标、预测和 fold artifact 字段 |
| `backend/app/models/` | 最终模型文件归并、旧版和退役实现清理 |
| `backend/app/routers/catalog.py` | 模型模块能力检查指向最终文件 |
| `backend/app/routers/datasets.py` | 上传摘要与原始数据集名返回 |
| `backend/app/routers/deps.py` | 创建 Run 时解析可信数据集元数据 |
| `backend/app/routers/runs.py` | Run 投影携带数据集名和规范指标 |
| `backend/app/runs/status_projection.py` | 新旧 Run 字段兼容、数据集名及指标恢复 |
| `static/index.html` | SpecAutoAI 品牌、Sample_ID 文案/排序、九项指标、训练记录表 |
| `backend/app/main.py`、`run.py` | FastAPI 标题和启动品牌 |
| `backend/tests/` | CSV、模型目录、CV 指标、Run 投影和前端契约回归 |
| `README.md`、`CONTEXT.md`、`docs/frontend_backend_handoff.md`、`AGENTS.md` | 新正式契约和迁移说明 |

## 四、按依赖顺序实施

### 任务 1：建立回归基线并保护现有工作区

**涉及文件或模块：**

- 检查：当前全部已修改文件。
- 测试：`backend/tests/test_smoke.py` 及字段、模型、Run 相关专项测试。

**实施内容：**

1. 实施前运行 `git status --short` 并保存本轮目标文件清单。当前工作区已有大量用户修改，所有编辑必须基于现状增量完成，不覆盖、不回滚。
2. 用 `rg` 分别列出 `AutoAI`、`Repeat_index`、`repeat_index`、`leave_one_repeat_index_cv`、`*_v2` 和训练记录旧列的引用。
3. 先增加新契约的失败测试，再改实现，防止全局替换遗漏字段、artifact 或前端取值。

**验收标准：**

- 本轮修改范围可单独列出。
- 无用户已有文件被恢复到旧版本。
- 后续每项改动都有对应的可失败回归用例。

### 任务 2：需求 2——统一迁移为 Sample_ID

**涉及文件或模块：**

- 修改：`backend/app/parsers.py`、`backend/app/hplc.py`、`backend/app/training.py`。
- 修改：`backend/app/runs/status_projection.py`、`static/index.html`。
- 测试：`backend/tests/test_smoke.py`、`backend/tests/modeling_data_factory.py`、相关训练/状态投影测试。

**接口与产出：**

- 新 CSV：`Index, Name, XXX, Intensity, Label, Sample_ID`。
- 新数据摘要：`sample_id.groups`，每组字段使用 `sample_id`。
- 新 CV 策略：`leave_one_sample_id_cv`。
- 新预测 CSV：`Sample_ID` 列。
- 新 split/cv/status 字段：`test_sample_id`、`train_sample_ids`、`valid_sample_ids`、`test_sample_ids`、`current_fold_sample_id`。

**实施内容：**

1. 将 `ModelingDataset.repeat_index`、分组摘要函数、训练变量和 UI 状态统一改为 `sample_id`。
2. 新文件严格要求 `Sample_ID` 非空；同一 `Sample_ID` 只能对应一个 Label，并继续校验每组重复测量条数一致。
3. 旧 CSV 只有旧列时在解析入口重命名为 `Sample_ID`；新旧两列同时出现且值不一致时返回 400，不猜测用户意图。
4. 新导出、预测结果、split JSON、cv_metrics、status、帮助文案和下载说明只写新名称。
5. 历史 Run 投影按旧字段读、新字段出，不改写原始历史 artifact。

**验收标准：**

- 新预处理 CSV 和新 Run artifact 中不存在旧业务字段。
- 旧 CSV 能上传并被规范化；冲突双列会明确失败。
- 同一 Sample_ID 的重复测量不会跨 Train/Valid/Test。
- 新 Run 只返回 `leave_one_sample_id_cv`。

### 任务 3：需求 5——数据集使用自然排序

**涉及文件或模块：**

- 修改：`backend/app/parsers.py`、`static/index.html`。
- 测试：数据摘要和前端契约测试。

**实施内容：**

1. 增加无第三方依赖的自然排序键，将字符串按数字片段和非数字片段拆分；数字片段按整数比较，文字片段按 `casefold()` 比较。
2. `Sample_ID` 分组摘要使用 `groupby(sort=False)` 后显式自然排序，避免 pandas 按字符串产生 `1, 10, 11, 2`。
3. 曲线选择器对展示副本自然排序；不重新排列训练 DataFrame，不改变标签、预测或数据划分索引。
4. 混合编号覆盖 `S1, S2, S10`，纯数字覆盖 `1, 2, 3, 10, 11`；相同键保持输入稳定顺序。

**验收标准：**

- AI 建模数据摘要和曲线选择器显示 `1, 2, 3, 10, 11`。
- 训练输入顺序、样本数和划分结果不因展示排序改变。

### 任务 4：需求 4——5 位小数与 Excel 32,767 字符保护

**涉及文件或模块：**

- 修改：`backend/app/parsers.py`、`backend/app/hplc.py`、`backend/app/routers/preprocess.py`。
- 测试：`backend/tests/test_smoke.py`。

**接口与产出：**

- 常量：小数位数 `5`、Excel 单元格字符上限 `32767`。
- HTTP 400：包含文件名、字段名、点数、实际字符数、上限和缩小范围建议。

**实施内容：**

1. 将当前 9 位有效数字输出改为小数点后最多 5 位；`XXX`、`Intensity`、预览曲线和下载 CSV 共用同一个量化结果。
2. 先完成量化和紧凑 JSON 序列化，再按最终字符串长度检查 32,767 上限；任一行的 `XXX` 或 `Intensity` 超限时不写出部分 CSV。
3. 对色谱/HPLC 1–9000 点建立真实格式测试，验证前端收到“超过 Excel 单元格上限 32767”的错误详情。
4. 增加 NaN/Inf、负零和量化后全零/常量化检查；错误必须告诉用户是“5 位小数精度不足”，不得静默继续。
5. 保持原始选择顺序：先按行或 X 轴范围选择，再执行拉曼基线/HPLC 标准流程，最后量化输出。

**验收标准：**

- 正常 CSV 数组元素最多 5 位小数且可被 `load_modeling_csv()` 读回。
- 9000 点等超限数据返回 400，不生成残缺文件。
- HPLC 有效小信号不会在无提示的情况下全部变成 0。

### 任务 5：需求 3——模型目录只保留最终实现

**涉及文件或模块：**

- 修改：`backend/app/models/__init__.py`、`registry.py`、`cnn_se1d.py`。
- 修改：`backend/app/routers/catalog.py`。
- 删除/归并：重复和退役模型文件。
- 测试：模型目录、模型构造、能力目录和旧行为测试。

**最终保留的模型源码：**

- 公共：`__init__.py`、`registry.py`、`profiles.py`。
- 传统模型：`pls_da.py`、`pca_lda.py`、`logistic_regression.py`、`svm.py`、`random_forest.py`、`xgboost.py`。
- 深度模型：`pca_mlp.py`、`cnn1d.py`、`cnn_se1d.py`、`resnet1d.py`、`inception1d.py`、`tcn1d.py`、`cnn_transformer1d.py`、`cnn_mamba1d.py`、`dscarnet.py`。

**实施内容：**

1. 用当前 `cnn1d_v2.py`、`resnet1d_v2.py`、`inception1d_v2.py`、`tcn1d_v2.py` 内容替换对应无后缀旧文件，保持正式类和数学结构不变。
2. 更新 registry、catalog、包导出、SE-CNN 依赖和测试导入，使运行路径只指向无版本后缀的最终文件。
3. 删除已经归并的四个 `*_v2.py`，以及不在 15 项目标目录中的 `knn.py`、`mlp.py`、`transformer.py`、`unet1d.py`。
4. 移除只验证退役模型类的测试；保留对 15 项目标能力、14 项可训练状态和 Mamba unavailable 行为的测试。
5. 删除前用 `rg` 确认正式运行路径无旧模块导入。历史 `.pt` 仍作为 state dict 下载；不承诺在新代码中重新实例化已经删除的旧网络类。

**验收标准：**

- 同一模型不再同时存在旧版和 v2 文件。
- `GET /api/models` 仍返回固定 15 项，当前环境仍是 14 项可用、`cnn_mamba1d` unavailable。
- 14 个当前模型的构造/轻量前向或传统拟合测试通过。

### 任务 6：需求 6——修正 CV Test macro F1 并显示九项指标

**涉及文件或模块：**

- 修改：`backend/app/training.py`、`backend/app/runs/status_projection.py`、`static/index.html`。
- 测试：`backend/tests/test_smoke.py`、训练执行、状态投影和前端契约测试。

**新指标契约：**

```json
{
  "metrics": {
    "train": {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0},
    "valid": {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0},
    "test":  {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
  },
  "cv_summary": {
    "primary_test_aggregation": "pooled_out_of_fold",
    "fold_mean": {},
    "fold_std": {},
    "pooled_test": {}
  }
}
```

现有 Accuracy、balanced accuracy、混淆矩阵和 classification report 可继续保留在下载产物中，但当前训练卡片只按老师要求主显示三组 macro Precision/Recall/F1。

**实施内容：**

1. 改造 `_aggregate_split_metrics()`：pooled 值、fold mean、fold std 分开存储，禁止用右侧字典展开覆盖同名 pooled 标量。
2. CV Test 使用 `all_true/all_pred` 的合并 OOF 结果填充 `metrics.test`；Test 混淆矩阵和 classification report 同样来自 OOF 合并结果。
3. CV Train/Valid 使用各折平均值；holdout/external-test 单折时直接使用该折评估结果。
4. 兼容顶层旧指标字段时，只把它们映射为新的 Test pooled OOF 值，防止旧前端继续拿到错误结果。
5. `fold_metrics.csv` 增加 `train_macro_precision/recall/f1`、`valid_*`、`test_*`；`cv_metrics.json` 同时保留每折和 pooled 证据。
6. 当前训练结果区改为 3×3 指标：Train、Valid、Test 各显示 Macro Precision、Macro Recall、Macro F1；删除该区域以 Accuracy 作为主要结果的展示方式。
7. 增加二分类回归用例：每折 test 只含一个类别且所有 OOF 预测正确时，每折 macro F1 平均可为 0.5，但页面使用的 `metrics.test.macro_f1` 必须是 1.0。

**验收标准：**

- 前端 Test macro F1 与根据 `cv_predictions.csv` 合并重算的结果一致。
- Train/Valid/Test 九项指标都有明确 macro 标签。
- 三种评估策略均遵循同一 JSON 字段契约。
- 单折辅助均值仍可下载审计，但不会覆盖主 Test 指标。

### 任务 7：需求 8——训练记录改为数据集名和三组 macro F1

**涉及文件或模块：**

- 修改：`backend/app/routers/datasets.py`、`backend/app/routers/deps.py`、`backend/app/routers/runs.py`。
- 修改：`backend/app/runs/status_projection.py`、`static/index.html`。
- 测试：数据集仓库、Run 路由、状态投影和前端 UI 契约测试。

**新列表列：**

`时间 | 上传数据集名 | 模型 | Train macro F1 | Valid macro F1 | Test macro F1 | 状态 | 操作`

**实施内容：**

1. 上传接口除 `dataset_id` 外返回 `dataset_name=original_name`；不使用随机 storage 文件名作为展示名。
2. 创建 Run 时从服务端数据集仓库解析可信 `original_name`，将 `dataset_name` 快照进 Run 的 `config_json`/状态投影。可选独立测试集同步保存 `test_dataset_name`，但主列表显示主训练数据集名。
3. 旧 `data_path` Run 仅取 `Path(...).name` 作为回退，不向浏览器暴露完整本地绝对路径；完全缺失时显示 `-`。
4. 训练记录表删除可见列：Run ID、数据量、训练准确率、测试准确率。
5. 增加 Train/Valid/Test macro F1 三列，值读取任务 6 的规范指标；排队、失败或旧记录缺失时显示 `-`。
6. Run ID 继续作为后端主键和按钮参数，保留查看、轮询、下载、取消、删除能力，只是不再作为表格列显示。
7. 长文件名在单元格内省略显示，`title` 保留完整名称并进行 HTML 转义。

**验收标准：**

- 新 Run 列表显示用户上传的原文件名和三组 macro F1。
- 表格不显示 Run ID、数据量和 Accuracy。
- 查看/删除仍能使用隐藏的 run_id 正常工作。
- 文件名不能注入 HTML，且服务端不泄漏绝对路径。

### 任务 8：需求 1——品牌改为 SpecAutoAI，并同步正式文档

**涉及文件或模块：**

- 修改：`static/index.html`、`backend/app/main.py`、`run.py`。
- 修改：`README.md`、`CONTEXT.md`、`AGENTS.md`、`docs/frontend_backend_handoff.md`、当前部署说明。
- 测试：`backend/tests/test_smoke.py` 及品牌字符串契约测试。

**实施内容：**

1. 页面 `<title>`、主标题、FastAPI `title`、启动器帮助、控制台横幅使用 `SpecAutoAI`。
2. 当前维护文档中的产品名和示例文案同步更新；历史事实需要提到旧名时写成“SpecAutoAI（原 AutoAI）”。
3. `AutoAI_开发计划.md` 作为历史文件保留文件名以避免引用断裂，但文档顶部标注当前品牌和历史性质。
4. 更新测试断言，确保正式页面和 OpenAPI 不再展示旧品牌。
5. 不改工作目录、Python 导入路径、API URL、storage/SQLite 路径或服务器作业文件名。

**验收标准：**

- 用户可见主界面、OpenAPI、启动输出和当前文档统一显示 SpecAutoAI。
- 应用仍可通过 `run.py` 或 `backend.app.main:app` 启动。
- 现有部署入口和数据路径不因品牌替换失效。

### 任务 9：明确排除需求 7，并清理计划残留

**不修改：**

- 不新增 `model_architecture.txt`。
- 不新增模型架构折叠区或层级表。
- 不为本轮扩展 `model_metadata.json` 的架构字段或参数量字段。
- 不新增模型架构下载入口和相关测试。

**验收标准：**本轮 diff 中没有由原需求 7 引入的文件、接口字段或 UI 组件。

## 五、验证方案

### 1. 定向测试

- CSV 契约：新 `Sample_ID`、旧列兼容、双列冲突、空值和分组一致性。
- 自然排序：`1, 10, 11, 2, 3` 与 `S1, S10, S2`。
- 预处理：最多 5 位小数、9000 点超限、量化全零保护、CSV 可读回。
- 模型目录：最终文件清单、15 项目录、14 项可用、Mamba unavailable。
- CV 指标：fold mean 与 pooled OOF 分离，前端读取 pooled Test。
- 训练记录：原文件名、三组 macro F1、无旧列、run_id 操作仍可用。
- 品牌：页面、FastAPI 和启动器显示 SpecAutoAI。

### 2. 项目验证命令

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

抽取 `static/index.html` 内联 `<script>`，再执行：

```powershell
node --check --input-type=commonjs
```

### 3. 真实数据闭环

项目根目录存在 `data.csv` 时，使用轻量传统模型完成：上传数据集 → 创建 queued Run → worker 训练 → 查看当前九项指标 → 查看训练记录 → 下载 `cv_predictions.csv`。随后用 sklearn 对合并预测独立重算 macro Precision、macro Recall、macro F1，并与页面 Test 指标逐项比对。

如真实数据、环境或耗时导致无法执行，交付说明必须明确未运行原因，不能只以单元测试代替而不说明。

## 六、交付检查与风险

- 当前工作区已有大量未提交修改；实施时只 stage 本轮相关代码和文档，禁止根目录 `git add -A`。
- `Sample_ID` 是跨 CSV、训练、artifact 和 UI 的契约迁移，应先完成后再改指标和训练记录，避免前后端一半新一半旧。
- 固定 5 位小数会改变实际训练输入；真实 HPLC 数据必须检查量化前后非零点比例、峰值和方差。若频繁触发失真保护，应回报老师并改为 5 位有效数字，而不是放宽保护。
- 删除旧模型源码会取消旧 Python 类的直接导入兼容；必须先确认正式服务只下载旧 `.pt/.pkl` 而不在服务端重新实例化退役网络。
- 历史 Run 默认不篡改 artifact。若历史记录缺少新指标或数据集名，UI 显示 `-`；只有证据完整的 `cv_predictions.csv` 才允许只读恢复 pooled Test 指标。
- 本方案可按任务 2/3、4、5、6、7、8 分批审查；每批通过定向测试后再继续，降低跨模块回归风险。
