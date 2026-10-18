# 多 Worker 并行与实验计数

用户要求保持模型/特征组合并行，在配置右上角显示实验总数，在当前训练区实时显示完成数/总数。本轮保留此前未提交的组合与排序改动，不中断已有训练。

## 执行与资源

- Worker 是独立进程，不等于一个 CPU 核心；同一个 Run（组合）的外层留一折仍顺序执行，多个组合由多个 Worker 并行领取。
- run.py 默认按可用逻辑 CPU 启动最多 4 个 Worker，并提供 --workers / --threads-per-worker；总线程预算不超过逻辑 CPU 数，默认保留 CPU 余量。每个 Worker 单独监督、停止退出后单独重启。
- 在导入数值库前传入 BLAS/OpenMP 预算，并在 Worker 内限制已加载 BLAS 与 PyTorch 的 CPU 线程。RF/XGBoost 使用同一预算，不让每个进程都 n_jobs=-1 抢满全部核心；不修改统计超参数、seed、数据或选参策略。
- GPU 可用时，在 SQLite claim 事务内保证同一队列最多一个深度模型 Run 运行；其他 Worker 跳过暂不可领取的深度任务，继续领取 CPU 模型，不能因队头 CNN 阻塞整个队列。CPU 模式不限制深度模型组合并发。
- 继续使用 claim token/lease，不能重复领取同一个 Run。STOP 只中断目标任务，未停止的 Worker/组合继续；归档和结果身份仍按模型×特征处理区分。

## 计数口径

- 配置右上角“预计实验 N 次”：N=有效模型×特征组合数×外层折数。CV 未加载数据时折数未知，显示 —，不造数。CNN×PCA 明确扣除。
- 当前训练区醒目显示“已完成 X / N 次实验”，带真实进度条，并保留运行/排队/失败数量与逐组合折进度。这个“实验”指一次外层训练/评估，不是候选搜索和最终重训的精确 fit 次数。
- 刷新从 SQLite Run 进度恢复；失败或停止不虚构已完成折，停止当前任务后清零，不拿下一次草稿配置覆盖当前批次总数。

## 验证

测试资源预算边界、监督器多子进程生命周期、并发 claim 唯一性、GPU 队头绕行、线程预算传递、计数与停止/刷新恢复。用真实测试.csv 的相同配置比较串行/并行执行，核验实际运行时间重叠、同 seed/划分/指标/产物，SwanLab 在线记录并回读。现场仅在空闲时同步升级；已在运行时只新增执行进程，不杀原任务。

### 2026-10-18 真实验证记录

- 数据使用本地 `测试.csv`：72 条测量、24 个 Sample_ID、4 类、7500 个特征；未复制到 Git。隔离队列和报告保存在 `outputs/parallel-worker-validation-20261018/`。
- 配置为 PLS-DA / XGBoost × BIN3 / BIN6 × 24 个外层留一折，quick、Z-score、seed=42；串行与并行各执行 4 个 Run、96 个外层实验。所有 Run 使用同一数据指纹、逐折划分与选参策略。
- 串行训练/评估执行时间合计 **206.18 秒**，并行从首个开始到最后结束跨度 **123.81 秒**，本次缩短约 **40%**（约 1.67 倍）。完整冷启动验证含四次独立进程启动、SwanLab 初始化/上传/回读和归档，串行 **287.13 秒**、并行 **150.50 秒**；这个完整流程时间不是常驻 Worker 的纯训练基准，不据此保证所有模型同等加速。
- SQLite 采样峰值为 4 个 running Run；4 个独立 PID 的真实执行时间存在共同重叠区间。每个进程记录的 OpenMP/OpenBLAS/PyTorch CPU 线程数均为 5，PyTorch inter-op 为 1。
- 四组 `metrics.test`、完整逐折 `split.json`（含获胜参数/分数）完全相同；每组 72 条 pooled OOF 预测的样品、类别和概率均一致，最大概率误差为 0。比较与归档均 ready，计数均为 96/96。
- 8 条 SwanLab 在线记录全部上传并回读核验；并行记录：[PLS-DA BIN3](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/0176cwot)、[PLS-DA BIN6](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/akz0kp30)、[XGBoost BIN3](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/7gg0g2t6)、[XGBoost BIN6](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/q582as7y)。本地保存配置、seed、提交与脏状态、tracked diff、untracked 源码、数据指纹、命令、依赖、线程池与日志。
- 浏览器用真实上传和控件事件验证 2×2×24=96；实际轮询中观察到 77/96、86/96、93/96 连续更新，保留逐组合进度和停止按钮。测试结束后关闭临时 8878 验证服务，返回 8877；确认旧队列无 queued/running 后，在同一原存储上升级 Web 与 4 个常驻 Worker，健康检查为 available=true、compatible=true、live_count=4，未删除或重新训练原有批次。
- GPU 独占 claim 与跳过受阻队头由并发单元测试验证；本轮真实流程只训练上述 CPU 传统模型，没有启动额外 CNN/GPU 性能实验。单组合内外层折仍为顺序执行，不声称留一折自身已并行。
- 最终完整回归 **596 passed、9 skipped**（273.31 秒），相关前端 Node 测试 **24/24 passed**；compileall、内联脚本提取后的 `node --check --input-type=commonjs` 和 `git diff --check` 通过。新增单元测试还覆盖监督器关闭与重启竞态，确保退出时不会遗留刚重新拉起的子进程。首次全量发现停止面板的旧测试夹具未加载新计数辅助函数，已同步夹具并增加清零断言，以上成绩为修正后的完整重跑，不沿用此前结果。
