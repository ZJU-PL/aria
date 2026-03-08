#!/bin/bash

# 配置日志目录路径（可以根据需要修改）
# LOG_DIR="logs_ours_prompt1x6_depth3"
LOG_DIR="logs_ours_qwen3"

# 创建logs目录（如果不存在）
mkdir -p "$LOG_DIR"

echo "开始运行所有基准测试 - $(date)"
echo "日志文件将保存在 $LOG_DIR/ 目录下"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 定义数据集
# declare -a datasets=("all-int" "autoproof" "dtt" "ind-ben" "vmcai15-dt")
# declare -a datasets=("autoproof" "dtt" "ind-ben" "vmcai15-dt")
declare -a datasets=("leon-test")

# 串行执行每个命令
for dataset in "${datasets[@]}"; do
    # 根据数据集名称构建路径
    path="$SCRIPT_DIR/preprocessed/$dataset"
    # 为每个数据集生成独立的时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    logfile="$LOG_DIR/${TIMESTAMP}_${dataset}.log"
    
    echo "=========================================="
    echo "正在运行数据集: $dataset"
    echo "日志文件: $logfile"
    echo "开始时间: $(date)"
    echo "=========================================="
    
    # 执行命令并将输出重定向到日志文件
    # kill all unuse processes
    sleep 10
    python3 run_exp_folder.py --root-path "$path" 2>&1 | tee "$logfile"
    # 休眠10秒，等待CVC5进程结束
    sleep 10
    ./kill_cvc_processes.sh
    sleep 10
    # 检查命令执行状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ 数据集 $dataset 执行成功"
        echo "执行成功 - 结束时间: $(date)" >> "$logfile"
    else
        echo "✗ 数据集 $dataset 执行失败"
        echo "执行失败 - 结束时间: $(date)" >> "$logfile"
        echo "错误: 数据集 $dataset 执行失败，请检查日志文件 $logfile"
    fi
    
    echo ""
done

echo "=========================================="
echo "所有基准测试完成 - $(date)"
echo "所有日志文件已保存在 $LOG_DIR/ 目录下"
echo "=========================================="