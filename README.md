# SpecAutoAI

面向拉曼与 HPLC 曲线的分类建模工具，支持数据预处理、模型训练、结果比较与导出，也可以通过 AI 助手完成这些操作。

## 可以做什么

- **整理曲线**：拉曼范围截取与基线校正，HPLC 范围选择与插值。
- **训练分类模型**：PLS-DA、Elastic Net、SVM、Random Forest、XGBoost、1D-CNN。
- **选择实验方案**：快速训练或完整比较，支持全特征、Binning、PCA，以及按样品分组的评估。
- **查看和导出结果**：分类指标、混淆矩阵、预测明细、多模型对比及图表。

## 快速开始

建议使用 Python 3.12 独立环境。首次安装、GPU 配置或依赖问题，请先看[环境配置手册](skills/autoai-research/references/environment-setup.md)。

在项目目录中运行：

```bash
python -m pip install -r backend/requirements.txt
python run_classic.py
```

打开 [本地页面](http://127.0.0.1:8000/)，上传原始曲线或建模 CSV，选择模型和评估方式，训练完成后查看并下载结果。

同一样品的重复测量使用相同的 `Sample_ID`，以便按样品整体划分训练与测试数据。

## 用 AI 助手建模

安装 `autoai-research` 技能后，告诉助手文件位置和目标：

> 使用 $autoai-research 检查 D:\数据\光谱.csv，推荐一个快速分类方案并运行。

助手先汇报数据情况和推荐设置，集中询问一轮可选问题；回复“按推荐方案”即可开始。明确说“请直接跑”时会直接执行。首次使用会检查环境，发现问题先告知，再按配置手册修复。

详见[技能使用说明](docs/autoai-research-skill.md)。

## 更多文档

- [环境配置与修复](skills/autoai-research/references/environment-setup.md)
- [服务器部署](deploy/server_deploy.md)
- [开发与接口文档](docs/README.md)
