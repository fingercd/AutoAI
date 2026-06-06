#!/usr/bin/env bash
#$ -N autoai_web
#$ -cwd
#$ -j y
#$ -o storage/logs/qsub_autoai.log
#$ -l hostname=node3
#$ -l gpu=1

set -euo pipefail
mkdir -p storage/logs
bash deploy/run_on_node3.sh
