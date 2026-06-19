# AutoAI — 谱学数据预处理与自动建模平台

## 快速启动

```bash
# 命令行启动（自动打开浏览器）
python run.py

# 自定义端口 / 热重载
python run.py --port 9000 --reload
```

PyCharm：右键 `run.py` → Run 即可。

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
