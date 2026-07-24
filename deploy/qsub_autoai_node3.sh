#!/usr/bin/env bash
# SGE template for the Web service only. Submit/manage the Run worker separately.
#
# 【中文说明】Sun Grid Engine（SGE）作业提交模板：把 Web 服务作为集群作业
# 提交到 node3 上运行。训练 worker 需单独提交/管理，不在本作业内。
# 用法：qsub deploy/qsub_autoai_node3.sh
#$ -N autoai_web
#$ -cwd
#$ -j y
#$ -o storage/logs/qsub_autoai.log
#$ -l hostname=node3
#$ -l gpu=1
# 上述 SGE 指令含义：-N 作业名；-cwd 在提交时的当前目录运行（保证相对路径
# storage/ 有效）；-j y 合并 stdout/stderr；-o 指定作业日志；
# -l hostname=node3 调度到指定节点 node3；-l gpu=1 申请 1 块 GPU 资源。

set -euo pipefail
mkdir -p storage/logs
# 实际启动逻辑（模式推导、token 校验、conda 激活、exec uvicorn）全部委托给
# run_on_node3.sh；本模板只声明 SGE 资源需求，保持单一职责。
bash deploy/run_on_node3.sh
