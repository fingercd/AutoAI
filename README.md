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

## 安装 AI 助手技能

两种安装方式任选其一。仓库为私有，需要 GitHub 访问权限。

### 手动安装

1. 打开 [qzt/skills 分支](https://github.com/fingercd/AutoAI/tree/qzt/skills)，通过 **Code → Download ZIP** 下载并解压到固定的项目目录。
2. 将项目中的 `skills/autoai-research` 整个文件夹复制到 Codex 技能目录：Windows 为 `%USERPROFILE%\.codex\skills\`，Linux/macOS 为 `~/.codex/skills/`；设置了 `CODEX_HOME` 时使用其下的 `skills/`。
3. 确认安装后的 `autoai-research/SKILL.md` 存在，下一轮对话即可使用。首次使用时告诉助手解压后的项目路径，它会检查环境并配置本机运行路径。

### 让 Agent 安装

把这段话发给 Codex：

> 请从 GitHub 的 fingercd/AutoAI 仓库 qzt/skills 分支下载完整项目，将 skills/autoai-research 安装到我的 Codex 技能目录。复用现有 GitHub 登录和已有项目；安装目录已存在时先检查，更新时保留本机 runtime.json。按环境配置手册检查依赖，发现问题先告知再修复，验证通过后配置本机运行路径，并告诉我如何开始使用。

完整项目目录用于本地计算，请保留；本机 `runtime.json` 不从其他机器复制。安装与更新细节见[技能使用说明](docs/autoai-research-skill.md)。

## 用 AI 助手建模

安装后，告诉助手文件位置和目标：

> 使用 $autoai-research 检查 D:\数据\光谱.csv，推荐一个快速分类方案并运行。

助手先汇报数据情况和推荐设置，集中询问一轮可选问题；回复“按推荐方案”即可开始。明确说“请直接跑”时会直接执行。首次使用会检查环境，发现问题先告知，再按配置手册修复。

## 更多文档

- [环境配置与修复](skills/autoai-research/references/environment-setup.md)
- [服务器部署](deploy/server_deploy.md)
- [开发与接口文档](docs/README.md)
