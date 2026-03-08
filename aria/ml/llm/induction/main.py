#!/usr/bin/env python3
"""
主程序入口文件
调用 Mate_new.py 中的求解器功能
"""

import sys
import logging

# 导入求解器模块

# 导入求解器模块
import Mate_new as mate_solver

# 获取 prove_run 函数
prove_run = mate_solver.prove_run

if __name__ == "__main__":
    # 参数调整为只需要路径和基础文件名
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <base_path> <base_name>")
        sys.exit(1)
    
    base_path = sys.argv[1]
    base_name = sys.argv[2]
    
    # 启动验证流程
    final_status = prove_run(base_path, base_name)
    
    # 输出最终结果
    logging.info("最终验证结论: %s", "成功" if final_status else "Fail")
    logging.info("unsat" if final_status else "")
    sys.exit(0 if final_status else 1)