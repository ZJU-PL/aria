import os
from pathlib import Path

def process_smt_file(input_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制原始文件到目标目录
    orig_path = output_dir / input_path.name
    with open(input_path, 'r') as src, open(orig_path, 'w') as dst:
        dst.write(src.read())

    # 读取原始文件内容
    with open(input_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    # 初始化存储区
    sections = {
        'set_logic': [],
        'datatypes': [],
        'functions': [],
        'proof_goal': [],
        'check_exit': []
    }

    current_block = None
    paren_stack = 0
    proof_goal_lines = []  # 存储 proof goal 的多行内容
    potential_proof_goal = False  # 标记可能的 proof goal 开始
    assert_paren_count = 0  # 记录 assert 的括号计数

    for line in lines:
        # 去除注释（分号及之后的内容）
        original_line = line  # 保留原始行格式
        if ';' in line:
            line = line.split(';', 1)[0].rstrip()
            original_line = line  # 更新去掉注释后的原始行
        stripped = line.strip()
        if not stripped:
            continue

        # 处理 set-logic
        if stripped.startswith('(set-logic'):
            sections['set_logic'].append(original_line)
        
        # 处理 datatypes（带括号匹配）
        elif stripped.startswith('(declare-datatypes'):
            current_block = 'datatypes'
            paren_stack = stripped.count('(') - stripped.count(')')
            sections[current_block].append(original_line)
        
        # 处理函数声明和普通 assert（但不是 proof goal）
        elif stripped.startswith(('(declare-fun', '(assert (')) and not stripped.startswith('(assert (not') and not potential_proof_goal:
            current_block = 'functions'
            paren_stack = stripped.count('(') - stripped.count(')')
            sections[current_block].append(original_line)
        
        # 处理 proof goal - 情况1: (assert (not 在同一行
        elif stripped.startswith('(assert (not'):
            current_block = 'proof_goal'
            proof_goal_lines = [original_line]
            paren_stack = stripped.count('(') - stripped.count(')')
            potential_proof_goal = False
            # 如果已经平衡，则立即完成
            if paren_stack == 0:
                sections['proof_goal'].append(' '.join([line.strip() for line in proof_goal_lines]))
                proof_goal_lines = []
                current_block = None
        
        # 处理 proof goal - 情况2: (assert 单独一行，可能是 proof goal 开始
        elif stripped.startswith('(assert') and not stripped.startswith('(assert ('):
            potential_proof_goal = True
            proof_goal_lines = [original_line]
            assert_paren_count = stripped.count('(') - stripped.count(')')
        
        # 处理 proof goal - 情况3: 在 potential_proof_goal 状态下遇到 (not
        elif potential_proof_goal and stripped.startswith('(not'):
            current_block = 'proof_goal'
            proof_goal_lines.append(original_line)
            paren_stack = assert_paren_count + stripped.count('(') - stripped.count(')')
            potential_proof_goal = False
            # 如果已经平衡，则立即完成
            if paren_stack == 0:
                sections['proof_goal'].append(' '.join([line.strip() for line in proof_goal_lines]))
                proof_goal_lines = []
                current_block = None
        
        # 处理 proof goal - 情况4: 在 potential_proof_goal 状态下遇到其他内容，说明不是 proof goal
        elif potential_proof_goal and not stripped.startswith('(not'):
            # 这不是 proof goal，将其作为普通函数处理
            current_block = 'functions'
            paren_stack = assert_paren_count
            sections[current_block].extend(proof_goal_lines)
            sections[current_block].append(original_line)
            paren_stack += stripped.count('(') - stripped.count(')')
            potential_proof_goal = False
            proof_goal_lines = []
            if paren_stack <= 0:
                current_block = None
        
        # 处理 proof goal - 继续
        elif current_block == 'proof_goal':
            proof_goal_lines.append(original_line)
            paren_stack += stripped.count('(') - stripped.count(')')
            # 如果括号平衡，则完成 proof goal
            if paren_stack <= 0:
                sections['proof_goal'].append(' '.join([line.strip() for line in proof_goal_lines]))
                proof_goal_lines = []
                current_block = None
        
        # 处理其他多行声明
        elif current_block in ['datatypes', 'functions']:
            sections[current_block].append(original_line)
            paren_stack += stripped.count('(') - stripped.count(')')
            if paren_stack <= 0:
                current_block = None
        
        # 保留结尾命令
        elif stripped.startswith(('(check-sat', '(exit')):
            sections['check_exit'].append(original_line)

    # 处理未完成的 potential_proof_goal（如果文件结束时还在这个状态）
    if potential_proof_goal and proof_goal_lines:
        sections['functions'].extend(proof_goal_lines)

    # 生成模板文件
    with open(output_dir / "template.smt2", 'w') as f:
        # 写入 set-logic
        if sections['set_logic']:
            f.write(sections['set_logic'][0] + '\n\n')
        
        # 写入 datatypes
        if sections['datatypes']:
            f.write('; datatypes\n')
            f.write('\n'.join(sections['datatypes']) + '\n')
            f.write('; datatypes end\n\n')
        
        # 写入函数声明
        if sections['functions']:
            f.write('; functions declarations\n')
            f.write('\n'.join(sections['functions']) + '\n')
            f.write('; functions declarations end\n\n')
        
        # 写入 proof goal
        if sections['proof_goal']:
            f.write('; proof goal\n')
            for goal in sections['proof_goal']:
                f.write(goal + '\n')
            f.write('; proof goal end\n\n')
        
        # 写入结尾命令
        if sections['check_exit']:
            f.write('\n'.join(sections['check_exit']) + '\n')

def process_directory(source_dir, target_root):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.smt2'):
                input_path = Path(root) / file
                # 保持目录结构：在目标根目录下创建相同相对路径
                relative_path = input_path.relative_to(source_dir).parent
                output_dir = target_root / relative_path / file[:-5]
                process_smt_file(input_path, output_dir)

if __name__ == '__main__':
    process_directory(
        Path('preprocessed/autoproof/standard'),
        Path('preprocessed/auto/standard')
    )