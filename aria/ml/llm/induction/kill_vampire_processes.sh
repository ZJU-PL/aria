#!/bin/bash

# 杀掉所有 vampire 进程的脚本
# 会持续运行直到没有 vampire 进程为止

echo "开始杀掉所有 vampire 进程..."

# 获取脚本自身的 PID，用于排除
SCRIPT_PID=$$

while true; do
    # 获取所有 vampire 进程的 PID
    # 只匹配真正的 vampire 可执行文件，排除：
    # 1. grep 进程本身
    # 2. kill_vampire_processes.sh 脚本
    # 3. SCREEN 会话等不相关进程
    # 4. 脚本自身
    pids=$(ps aux | grep -E '[^/]vampire[^/]|/vampire/vampire|vampire\.smt2' | \
           grep -v grep | \
           grep -v 'kill_vampire' | \
           grep -v 'SCREEN' | \
           grep -v "$$" | \
           awk '{print $2}')
    
    if [ -z "$pids" ]; then
        echo "没有发现 vampire 进程，任务完成！"
        break
    else
        count=$(echo $pids | wc -w)
        echo "发现 $count 个 vampire 进程，正在终止..."
        
        # 先尝试优雅终止
        for pid in $pids; do
            # 检查进程是否还存在
            if ps -p $pid > /dev/null 2>&1; then
                # 尝试终止进程组
                kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null
            fi
        done
        
        # 等待进程退出
        sleep 2
        
        # 检查是否还有进程存活，强制杀死
        remaining_pids=$(ps aux | grep -E '[^/]vampire[^/]|/vampire/vampire|vampire\.smt2' | \
                         grep -v grep | \
                         grep -v 'kill_vampire' | \
                         grep -v 'SCREEN' | \
                         grep -v "$$" | \
                         awk '{print $2}')
        
        if [ -n "$remaining_pids" ]; then
            echo "强制终止剩余进程..."
            echo $remaining_pids | xargs kill -9 2>/dev/null
        fi
        
        echo "已发送终止信号，等待 2 秒后再次检查..."
        sleep 2
    fi
done

echo "所有 vampire 进程已被终止！"

