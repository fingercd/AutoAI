# AutoAI — 谱学数据预处理与自动建模平台

## 快速启动

```bash
# 首次运行先安装依赖
python -m pip install -r backend/requirements.txt

# 命令行启动（自动打开浏览器）
python run.py

# 自定义端口 / 热重载
python run.py --port 9000 --reload
```

PyCharm：右键 `run.py` → Run 即可。

如果使用本机已有的 pytorch conda 环境，也可以直接：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe run.py
```

等效的手动命令：

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/          → 主页面
http://127.0.0.1:8000/ui        → UI 方案画廊
http://127.0.0.1:8000/docs      → Swagger API 文档
```

## 功能

- **数据预处理**：上传拉曼 / 色谱原始 CSV，基线校正（arPLS / 多项式）、范围截取（按行 / 按 X 轴）、生成统一建模 CSV
- **AI 建模**：上传建模 CSV → 按 Repeat_index 分组划分训练/验证/测试集 → 训练 → 查看指标和混淆矩阵 → 下载预测结果
- **9 种模型**：CNN1D / MLP / Transformer / UNet1D / DSCARNet (deep) + KNN / RandomForest / SVM / XGBoost (traditional)
- **外部测试集**：支持独立测试 CSV 文件评估
- **8 套 UI 方案**：Workbench / Wizard / Dashboard / Console / Minimal Lab / Swiss / Dark Instrument / Warm Paper

## 建模数据划分说明

训练、验证、测试集按 `Repeat_index` 整组划分，而不是按单条曲线随机划分。

同一个 `Repeat_index` 下的所有重复测量会全部进入同一个集合，例如都进训练集，或都进验证集，或都进测试集。这样可以避免同一个样品的重复测量同时出现在训练集和测试集里，造成数据泄漏和指标虚高。

上传建模 CSV 时需要保证：

- 同一个 `Repeat_index` 内只能对应一个 `Label`。
- 每个 `Repeat_index` 的重复测量次数应保持一致。
- 默认划分比例是 `train : valid : test = 8 : 1 : 1`。

例如 50 条数据、10 个 `Repeat_index`、每组 5 条重复测量时，默认会划分为：

```text
train = 8 个 Repeat_index = 40 条
valid = 1 个 Repeat_index = 5 条
test  = 1 个 Repeat_index = 5 条
```

如果某个类别只集中在少数 `Repeat_index` 中，默认 8/1/1 可能导致验证集或测试集只有一个类别，表现为 valid/test 指标波动较大。遇到这种情况，建议改成 `6/2/2` 或 `7/2/1`，让验证集和测试集包含更多 `Repeat_index` 分组。
