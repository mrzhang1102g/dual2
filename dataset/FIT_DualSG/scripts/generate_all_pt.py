#!/usr/bin/env python3
"""
批量生成所有结构化数据的PT文件
"""

import os
import subprocess

# 数据文件列表
files_to_process = [
    'fit_dualsg_structured.json',
    'fit_dualsg_global_structural.json',
    'fit_dualsg_dynamic_behavior.json',
    'fit_dualsg_event_centric.json',
    'fit_dualsg_semantic_caption.json'
]

# 目录设置
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = base_dir
pt_dir = os.path.join(base_dir, 'pt')
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'precompute_pt.py')

# 创建PT目录
os.makedirs(pt_dir, exist_ok=True)

print(f'数据目录: {data_dir}')
print(f'PT目录: {pt_dir}')
print(f'脚本路径: {script_path}')
print()

# 处理每个文件
for filename in files_to_process:
    data_path = os.path.join(data_dir, filename)
    pt_filename = filename.replace('.json', '.pt')
    save_path = os.path.join(pt_dir, pt_filename)
    
    print(f'处理: {filename}')
    print(f'生成: {pt_filename}')
    
    # 构建命令
    cmd = f"python {script_path} --data_path {data_path} --save_path {save_path}"
    
    # 执行命令
    print(f'执行命令: {cmd}')
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    
    # 检查执行结果
    if result.returncode == 0:
        print(f'✓ 成功生成: {pt_filename}')
    else:
        print(f'✗ 失败: {filename}')
        print(f'错误信息: {result.stderr}')
    
    print()

print('批量生成完成！')
