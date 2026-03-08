#!/bin/bash

# 杀掉所有 cvc4 和 cvc5 进程的脚本
# 会持续运行直到没有 cvc4 和 cvc5 进程为止

echo "开始杀掉所有 cvc4 和 cvc5 进程..."

while true; do
    # 获取所有 cvc4 和 cvc5 进程的 PID
    pids=$(ps aux | grep -E 'cvc[45]' | grep -v grep | grep -v kill_cvc5.sh | awk '{print $2}')
    
    if [ -z "$pids" ]; then
        echo "没有发现 cvc4 或 cvc5 进程，任务完成！"
        break
    else
        echo "发现 $(echo $pids | wc -w) 个 cvc4/cvc5 进程，正在终止..."
        # 杀掉所有找到的进程
        echo $pids | xargs kill -9 2>/dev/null
        echo "已发送终止信号，等待 2 秒后再次检查..."
        sleep 2
    fi
done

echo "所有 cvc4 和 cvc5 进程已被终止！"