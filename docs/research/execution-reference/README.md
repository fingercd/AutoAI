# 知识选模的冻结执行源码参考

这里的`.py.txt`文件是原实验真实入口的只读副本，供代码审阅和结果追溯，不是新的开发工作树，也不是可直接在任意机器执行的产品入口。保留原文件内容和原运行路径，不复制整份旧仓库。文件没有因2026年10月15日文档整理而改变科学行为。

- `initial-selection-core.py.txt`：初版知识选模的已验证输出结构版本。
- `revised-selection-core.py.txt`：修订版知识、两个Train描述、原检索方式和最小JSON结构。
- `revised-selection-runner.py.txt`：129次选择的持久化与请求核算；不提交分类训练。
- `revised-comparison.py.txt`：从现有矩阵读取成绩，计算均匀随机期望。
- `revised-independent-audit.py.txt`：独立核对原始回复、选择绑定、检索和903项记录。

本次当前产品源码在仓库`backend/`、`agent_poc/`和`scripts/`中；这些参考文件不等于当前产品已经接入12张研究卡片或支持个人经验更新。不要从历史入口中复制已批准的状态去冒充新实验授权。
