# 0904 建模方案与科研结果图

本次实现依据用户提供的 AutoAI_model要求_0904.docx 和本对话确认。后续视觉调整明确要求桌面每行四图、低饱和墨绿色矩阵，覆盖原方案中的两列与蓝色。

## 模型与特征工程

新 UI 发送 experiment_version=word-0904，仅显示 PLS-DA、Elastic Net、SVM、Random Forest、XGBoost、1D-CNN。保留其他后端模型和未指定新版方案的兼容请求。每个模型一个 Run，不增加重复实验。

传统模型采用全特征、Binning 5/10/20、PCA 90/95/99% 七个独立方案。Binning 按相邻点均值、保留尾箱；PCA 在标准化之后、仅用当前训练折拟合。CNN 只比较全特征与三种 Binning。CNN 的 PCA 单元格不适用。

PLS 成分 [1,2,3,4,5,6,8,10,12,15]；Elastic Net 的 C=[.01,.1,1,10,100]、l1_ratio=[.1,.5,.9]；SVM 比较 linear/RBF，C 同前，RBF gamma=[scale,.001,.01]；RF 和 XGBoost 严格采用 Word 第七节固定网格。候选均在分组分层内层五折选优，不能以测试指标选参。各方案重训的测试成绩作为特征工程热图数据。

CNN 是独立 CNN0904，实现三段 Conv/BN/ReLU/MaxPool、AdaptiveAvgPool、Dropout、多类别输出（含二分类）及 CrossEntropyLoss。N 取训练独立 Sample_ID 数；N=300 和 L=3000 均属于中档，长序列池化为 [4,2,2]。默认 AdamW、200 epoch、batch=8、lr=.001、weight_decay=.0001、指定调度与早停。用户已有可调整训练参数仍可覆盖默认值，并进入审计。

## 评估与历史

维持四种评估路径。普通 LOSO 每折独立选方案与参数，报告 pooled OOF，没有全局单一最佳参数。外部测试+LOSO 在主数据完成选择与最终拟合后评估外部测试；CNN 最终拟合使用预先选择的 epoch 数，不用训练集损失重新决定停止时间。

历史重复批次在同样划分、标签及数据指纹下按单次结果查看，提供每模型运行切换；默认第一条成功且完整的运行。新界面不再展示均值合并矩阵。历史没有特征工程记录时明确缺失。

## 实现与验收

共享前端为 scientific-results.js/css，支持四指标、统一矩阵、Recall、分页正误热图、特征工程热图及 PNG/SVG 导出。两套入口均复用组件。合成视觉验收页在 static/v2/tests/scientific-fixture.html，始终明确标注合成数据。

后端 feature_engineering.py 定义折内变换，training_experiments.py 完成方案与参数选择，training.py 保留原调度、划分、指标汇总与产物发布。feature_experiments.json/csv 是 Manifest 校验的公开结果；完整模型与拟合变换仍为私有产物。

验收包括完整后端回归、六模型固定合成数据、三条额外外部/CV 路径、防泄露、组件数值测试和多宽度浏览器检查。六模型验收中传统模型执行完整网格，CNN 使用两 epoch 验证训练闭环；不将该小数据成绩解读为模型泛化能力。根目录无 data.csv，真实实验效果未验收。
