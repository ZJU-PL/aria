import re
import logging
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from logger_config import setup_colored_logger
from env_config import setup_environment, setup_model
from cvc5_runner import run_cvc_solver_with_timeout

# 配置彩色日志
logger = setup_colored_logger()
config = setup_environment()

# 初始化模型
llm = setup_model(config)

# 在文件开头添加失败引理管理函数
def get_failed_lemmas_file(base_path: str, goal_name: str) -> Path:
    """获取失败引理记录文件路径，根据目标名称生成对应文件"""
    # 提取目标名称的后缀部分来构建文件名
    if goal_name == "template":
        filename = "failed_lemmas.json"
    else:
        # 提取template后面的部分，如template_1 -> _1, template_1_2 -> _1_2
        suffix = goal_name.replace("template", "")
        filename = f"failed_lemmas{suffix}.json"
    
    return Path(base_path) / filename

def load_failed_lemmas(base_path: str, goal_name: str) -> dict:
    """加载失败引理记录"""
    failed_file = get_failed_lemmas_file(base_path, goal_name)
    if failed_file.exists():
        try:
            with open(failed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"加载失败引理文件出错: {e}")
    return {"invalid_lemmas": [], "useless_lemma_groups": []}

def save_failed_lemmas(base_path: str, goal_name: str, failed_data: dict):
    """保存失败引理记录"""
    failed_file = get_failed_lemmas_file(base_path, goal_name)
    try:
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, ensure_ascii=False, indent=2)
        # logging.info(f"保存失败引理到: {failed_file.name}")
    except Exception as e:
        logging.error(f"保存失败引理文件出错: {e}")

def add_invalid_lemma(base_path: str, goal_name: str, lemma: str, reason: str):
    """添加无效引理记录"""
    failed_data = load_failed_lemmas(base_path, goal_name)
    lemma_record = {"lemma": lemma, "reason": reason}
    if lemma_record not in failed_data["invalid_lemmas"]:
        failed_data["invalid_lemmas"].append(lemma_record)
        save_failed_lemmas(base_path, goal_name, failed_data)
        # logging.info(f"记录无效引理到{goal_name}: {reason} - {lemma[:50]}...")

def add_useless_lemma_group(base_path: str, goal_name: str, lemma_group: List[str]):
    """添加无用引理组合记录"""
    failed_data = load_failed_lemmas(base_path, goal_name)
    if lemma_group not in failed_data["useless_lemma_groups"]:
        failed_data["useless_lemma_groups"].append(lemma_group)
        save_failed_lemmas(base_path, goal_name, failed_data)
        logging.info(f"记录无用引理组合到{goal_name}: {len(lemma_group)}条引理")

def create_prompt(smt_file_content: str, prompt_mode: str, base_path: str = None, goal_name: str = None, folder_path: str = None) -> list:
    """创建用于 LLM 的结构化消息列表"""
    if folder_path is None:
        raise ValueError("folder path of prompts must be provided")

    with open(f"{folder_path}/{prompt_mode}/system_prompt.txt", "r", encoding="utf-8") as file:
        system_prompt_content = file.read()
    with open(f"{folder_path}/{prompt_mode}/user_prompt.txt", "r", encoding="utf-8") as file:
        user_prompt_content = file.read()
    # 添加失败引理信息
    failed_info = ""
    if base_path and goal_name:
        failed_data = load_failed_lemmas(base_path, goal_name)
        
        if failed_data["invalid_lemmas"]:
            failed_info += "\n\n; IMPORTANT: The following lemmas are INVALID or CANNOT be verified. DO NOT generate these lemmas:\n"
            for i, record in enumerate(failed_data["invalid_lemmas"], 1):
                failed_info += f"; Invalid lemma {i} ({record['reason']}): {record['lemma']}\n"
        
        if failed_data["useless_lemma_groups"]:
            failed_info += "\n; IMPORTANT: The following lemma groups are USELESS for proving the original goal. DO NOT generate the exact same group:\n"
            for i, group in enumerate(failed_data["useless_lemma_groups"], 1):
                failed_info += f"; Useless group {i}:\n"
                for j, lemma in enumerate(group, 1):
                    failed_info += f";   {j}. {lemma}\n"
    
    # 构建结构化消息列表
    user_content = user_prompt_content.format(smt_file_content=smt_file_content) + failed_info
    
    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_content}
    ]
    
    return messages

def extract_balanced_forall(assert_not_content: str) -> Optional[str]:
    """提取平衡的 forall 表达式"""
    # 首先定位到 forall 的开始位置
    start_match = re.search(r'\(\s*forall', assert_not_content)
    if not start_match:
        return None
    
    start_pos = start_match.start()
    balance = 0
    end_pos = start_pos
    
    # 从 forall 开始处扫描，找到平衡的右括号
    for i, c in enumerate(assert_not_content[start_pos:]):
        if c == '(':
            balance += 1
        elif c == ')':
            balance -= 1
            if balance == 0:
                end_pos = start_pos + i + 1
                break
    
    return assert_not_content[start_pos:end_pos]

def parse_llm_response(response: str) -> List[str]:
    """解析LLM输出，提取有效断言"""
    pattern = r'; Output begin(.*?); Output end'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        raise ValueError("响应格式错误，缺少输出标记")

    result = []
    for line in match.group(1).split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 使用已有的 extract_balanced_forall 函数提取 forall 内容
        # 可以处理 (forall...) 和 (assert (forall...)) 两种情况
        forall_content = extract_balanced_forall(line)
        if forall_content:
            result.append(forall_content)
    
    return result

def verify_combined_lemmas(original_assert: re.Match, asserts: List[str], original_content: str, output_path: Path) -> bool:
    
    """验证引理组合有效性"""
    combined_asserts = "\n".join([f"(assert {a})" for a in asserts])

    # 将引理插入到 proof goal 之前，保持 proof goal 区块不变
    new_content = re.sub(
        r'; proof goal',
        combined_asserts + "\n; proof goal",
        original_content,
        count=1
    )

    output_path.write_text(new_content)
    combined_cvc_timeout = config['COMBINED_CVC_TIMEOUT']
    return run_cvc_solver_with_timeout(output_path, combined_cvc_timeout)


def perform_initial_verification(goal_smt_file: Path) -> bool:
    """执行初始验证检查"""
    default_timeout = config['DEFAULT_CVC_TIMEOUT']
    logging.info(f"🔍执行初始检查, 目标文件: {goal_smt_file}")
    if run_cvc_solver_with_timeout(goal_smt_file, default_timeout):
        logging.info("✅ 原目标直接验证成功!")
        return True
    
    logging.error("CVC5验证未通过，开始生成新引理...")
    return False


def extract_original_goal(smt_content: str) -> Tuple[re.Match, str]:
    """提取原始目标断言和forall表达式"""
    original_assert = re.search(
        r'; proof goal\s*(\(assert.*?\))\s*; proof goal end',
        smt_content,
        flags=re.DOTALL
    )
    original_forall = extract_balanced_forall(original_assert.group(1))
    logging.info(f"提取到原始目标: {original_assert.group(1)}, forall 表达式: {original_forall}")
    return original_assert, original_forall

def generate_lemmas_with_llm(smt_content: str, prompt_strategy: str, goal_smt_file: Path, base_path: str, goal_name: str, folder_path: str) -> List[str]:
    """使用LLM生成引理"""
    logging.info(f"即将使用LLM生成引理, 目标文件: {goal_smt_file}, 提示策略: {prompt_strategy}")
    messages = create_prompt(smt_content, prompt_strategy, base_path, goal_name, folder_path)
    response = llm.invoke(messages)
    extracted_asserts = parse_llm_response(response.content)
    
    # logging.info(f"LLM response: {response.content}")
    logging.info("从大模型返回中提取引理: %s", extracted_asserts)
    
    return extracted_asserts

def normalize_formula(formula: str) -> str:
    """增强的公式标准化函数"""
    # 1. 清理空白字符
    formula = re.sub(r'\s+', ' ', formula.strip())
    
    # 2. 查找forall关键字
    forall_pos = formula.find('forall')
    if forall_pos == -1:
        return formula
    
    # 3. 手动解析变量定义部分
    ptr = forall_pos + len('forall')
    while ptr < len(formula) and formula[ptr] in ' \t\n':
        ptr += 1
    
    if ptr >= len(formula) or formula[ptr] != '(':
        return formula
    
    # 找到变量定义部分的结束位置
    balance = 0
    var_def_start = ptr
    for i in range(ptr, len(formula)):
        if formula[i] == '(':
            balance += 1
        elif formula[i] == ')':
            balance -= 1
            if balance == 0:
                var_def_end = i + 1
                break
    else:
        return formula
    
    # 提取变量定义和公式体
    var_section = formula[var_def_start+1:var_def_end-1]
    body_start = var_def_end
    while body_start < len(formula) and formula[body_start] in ' \t\n':
        body_start += 1
    
    # 找到公式体
    balance = 0
    body_end = len(formula)
    for i in range(body_start, len(formula)):
        if formula[i] == '(':
            balance += 1
        elif formula[i] == ')':
            balance -= 1
            if balance == 0:
                body_end = i + 1
                break
    
    body = formula[body_start:body_end]
    
    # 4. 解析变量定义
    var_defs = re.findall(r'\(\s*(\w+)\s+(\w+)\s*\)', var_section)
    if not var_defs:
        return formula
    
    # 5. 创建变量映射
    var_map = {var: f'a{i}' for i, (var, _) in enumerate(var_defs)}
    
    # 6. 替换变量
    normalized_body = body
    for old_var, new_var in sorted(var_map.items(), key=lambda x: -len(x[0])):
        normalized_body = re.sub(rf'\b{re.escape(old_var)}\b', new_var, normalized_body)
    
    # 7. 重构公式
    normalized_vars = ' '.join(f'({var_map[var]} {typ})' for var, typ in var_defs)
    return f'(forall ({normalized_vars}) {normalized_body})'

def extract_equality_parts(formula: str):
    """提取等式的左右两部分"""
    # 查找最外层的等式
    eq_start = formula.rfind('(=')
    if eq_start == -1:
        return None, None
    
    # 找到等式的结束位置
    balance = 0
    eq_end = len(formula)
    for i in range(eq_start, len(formula)):
        if formula[i] == '(':
            balance += 1
        elif formula[i] == ')':
            balance -= 1
            if balance == 0:
                eq_end = i + 1
                break
    
    # 提取等式内容（去掉 "(= " 和 ")"）
    eq_content = formula[eq_start + 3:eq_end - 1].strip()
    
    # 手动解析左右两部分
    balance = 0
    parts = []
    current_part = ""
    
    for char in eq_content:
        if char == '(':
            balance += 1
            current_part += char
        elif char == ')':
            balance -= 1
            current_part += char
        elif char == ' ' and balance == 0:
            if current_part.strip():
                parts.append(current_part.strip())
                current_part = ""
        else:
            current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None

def normalize_equality_order(formula: str) -> str:
    """标准化等式的左右顺序"""
    # 提取等式的左右部分
    left, right = extract_equality_parts(formula)
    if left is None or right is None:
        return formula
    
    # 按字典序排序，确保相同的等式有统一的顺序
    if left <= right:
        return formula  # 已经是标准顺序
    else:
        # 交换左右顺序 - 需要找到并替换完整的等式部分
        eq_start = formula.rfind('(=')
        if eq_start == -1:
            return formula
            
        # 找到等式的结束位置
        balance = 0
        eq_end = len(formula)
        for i in range(eq_start, len(formula)):
            if formula[i] == '(':
                balance += 1
            elif formula[i] == ')':
                balance -= 1
                if balance == 0:
                    eq_end = i + 1
                    break
        
        # 构造新的等式
        new_equality = f'(= {right} {left})'
        return formula[:eq_start] + new_equality + formula[eq_end:]

def are_formulas_equivalent(formula1: str, formula2: str) -> bool:
    """增强的公式等价性检查"""
    try:
        # 1. 标准化两个公式
        norm1 = normalize_formula(formula1)
        norm2 = normalize_formula(formula2)
        
        # 2. 标准化等式顺序
        norm1 = normalize_equality_order(norm1)
        norm2 = normalize_equality_order(norm2)
        
        # 3. 比较标准化后的公式
        return norm1 == norm2
        
    except Exception as e:
        # print(f"标准化过程出错: {e}，回退到简单比较")
        return formula1.strip() == formula2.strip()

def validate_lemmas_against_original(extracted_asserts: List[str], original_forall: str, base_path: str, goal_name: str) -> bool:
    """增强的引理验证函数"""
    for i, assert_stmt in enumerate(extracted_asserts, 1):
        if are_formulas_equivalent(assert_stmt, original_forall):
            # logging.error(f"引理 {i} 与原目标相同，生成失败")
            # add_invalid_lemma(base_path, goal_name, assert_stmt, "Same as original goal")
            return False
    return True


def create_validation_files(extracted_asserts: List[str], smt_content: str, 
                          smt_file_path: Path, goal_smt_name: str) -> List[Path]:
    """创建引理有效性验证文件"""
    valid_check_paths = []
    for i, assert_stmt in enumerate(extracted_asserts, 1):
        valid_content = re.sub(
            r'; proof goal\s*\(assert.*?\)\s*; proof goal end',
            f'; proof goal\n(assert {assert_stmt})\n; proof goal end',
            smt_content,
            flags=re.DOTALL
        )
        valid_path = smt_file_path / f"{goal_smt_name}_valid_{i}.smt2"
        valid_path.write_text(valid_content)
        valid_check_paths.append(valid_path)
    return valid_check_paths


def verify_single_lemma(valid_path: Path) -> Tuple[Path, bool]:
    """验证单个引理的有效性"""
    logging.info(f"开始检查有效性: {valid_path.name}")
    result = run_cvc_solver_with_timeout(valid_path, timeout=1)
    logging.info(f"检查有效性: {valid_path.name} 结束，返回 {result}")
    return valid_path, result


def validate_lemmas_parallel(valid_check_paths: List[Path], base_path: str, goal_name: str) -> bool:
    """并行验证引理有效性"""
    invalid_lemmas = []
    max_workers = min(len(valid_check_paths), 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(verify_single_lemma, path): path 
                         for path in valid_check_paths}
        
        for future in as_completed(future_to_path):
            try:
                valid_path, result = future.result()
                if result:
                    # logging.warning(f"发现无效引理: {valid_path.name}")
                    invalid_lemmas.append(valid_path)
                    # 记录无效引理
                    lemma_content = extract_lemma_from_file(valid_path)
                    if lemma_content:
                        add_invalid_lemma(base_path, goal_name, lemma_content, "verification failed")
                else:
                    logging.error(f"暂时无法过滤引理: {valid_path.name}")
            except Exception as e:
                path = future_to_path[future]
                #logging.error(f"验证引理 {path.name} 时发生异常: {e}")
                invalid_lemmas.append(path)
                # 记录异常引理
                lemma_content = extract_lemma_from_file(path)
                if lemma_content:
                    add_invalid_lemma(base_path, goal_name, lemma_content, f"验证异常: {e}")
    
    if invalid_lemmas:
        # logging.error("存在不合法引理，需要重新生成")
        return False
    return True

def extract_lemma_from_file(file_path: Path) -> str:
    """从验证文件中提取引理内容"""
    try:
        content = file_path.read_text()
        match = re.search(r'; proof goal\s*\(assert\s+(.+?)\)\s*; proof goal end', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        logging.error(f"提取引理内容失败: {e}")
    return None


def generate_formal_proof_files(extracted_asserts: List[str], smt_content: str,
                        smt_file_path: Path, goal_smt_name: str) -> List[str]:
    """生成正式验证文件（取反的引理）"""
    generated_files = []
    for i, assert_stmt in enumerate(extracted_asserts, 1):
        lemma_content = re.sub(
            r'; proof goal\s*\(assert.*?\)\s*; proof goal end',
            f'; proof goal\n(assert (not {assert_stmt}))\n; proof goal end',
            smt_content,
            flags=re.DOTALL
        )
        lemma_path = smt_file_path / f"{goal_smt_name}_{i}.smt2"
        lemma_path.write_text(lemma_content)
        generated_files.append(lemma_path.name.split('.')[0])
        logging.info(f"生成验证文件: {lemma_path.name}")
    return generated_files


def quick_run(base_path: str, goal_smt_name: str, prompt_strategy: str, folder_path: str, baseline_only: bool = False) -> Tuple[bool, List[str], List[str]]:
    """快速运行函数, 返回验证结果、子目标文件和生成的引理"""
    smt_file_path = Path(base_path)
    goal_smt_file = smt_file_path / f"{goal_smt_name}.smt2"
    smt_content = goal_smt_file.read_text()

    # 步骤1: 初始验证检查
    if baseline_only:
        # 在baseline模式下，使用task_timeout作为超时时间
        task_timeout = config['TASK_TIMEOUT']
        logging.info(f"🔍 Baseline模式: 执行初始验证检查，超时时间: {task_timeout}秒")
        result = run_cvc_solver_with_timeout(goal_smt_file, task_timeout)
        if result:
            logging.info("✅ Baseline模式: 初始验证成功!")
        else:
            logging.info("❌ Baseline模式: 初始验证失败")
        return result, [], []

    # 步骤2: 提取原始目标
    original_assert, original_forall = extract_original_goal(smt_content)
    
    # 步骤3: 使用LLM生成引理
    extracted_asserts = generate_lemmas_with_llm(smt_content, prompt_strategy, goal_smt_file, base_path, goal_smt_name, folder_path)

    # 如果没有生成引理，尝试延长时间重新验证
    if not extracted_asserts:
        logging.info("大模型未返回引理，尝试提高时间限制重新验证原目标")
        retry_timeout = config['RETRY_CVC_TIMEOUT']
        if run_cvc_solver_with_timeout(goal_smt_file, retry_timeout):
            logging.info("原目标提高时间到%d秒后验证成功!", retry_timeout)
            return True, [], []
        return False, [], []

    # TODO: 去掉这一部分做消融实验↓
    # 步骤4: 验证引理是否与原目标相同
    if not validate_lemmas_against_original(extracted_asserts, original_forall, base_path, goal_smt_name):
        return False, [], extracted_asserts

    # 步骤5: 创建验证文件并并行验证引理有效性
    valid_check_paths = create_validation_files(extracted_asserts, smt_content, smt_file_path, goal_smt_name)
    
    if not validate_lemmas_parallel(valid_check_paths, base_path, goal_smt_name):
        return False, [], extracted_asserts


    # 步骤6: 检查引理是否有助于证明原目标
    combined_path = smt_file_path / f"{goal_smt_name}_with_lemmas.smt2"
    if not verify_combined_lemmas(original_assert, extracted_asserts, smt_content, combined_path):
        logging.error("生成引理未能帮助证明原目标")
        # 记录无用引理组合
        add_useless_lemma_group(base_path, goal_smt_name, extracted_asserts)
        return False, [], extracted_asserts
    else:
        logging.info("lemmas有用，生成lemmas验证文件，检查lemmas是否成立")

    # 步骤7: 生成正式验证文件
    generated_files = generate_formal_proof_files(extracted_asserts, smt_content, smt_file_path, goal_smt_name)
    
    return True, generated_files, extracted_asserts

def prove_subgoals_parallel(base_path: str, subgoals: List[str], depth: int = 0, strategy_mode: str = "default", baseline_only: bool = False, parent_lemmas: List[str] = None, parent_goal_name: str = None) -> bool:
    """并行执行多个子目标的验证，如果任何一个失败就立即终止所有进程"""
    if not subgoals:
        return True
    
    logging.info(f"🚀 开始并行验证 {len(subgoals)} 个子目标: {subgoals} (递归深度: {depth})")
    
    # 使用ThreadPoolExecutor进行并行执行
    with ThreadPoolExecutor(max_workers=min(len(subgoals), 4)) as executor:
        # 提交所有任务，传递递增的深度参数
        future_to_subgoal = {executor.submit(prove_run, base_path, subgoal, depth + 1, strategy_mode, baseline_only): subgoal 
                            for subgoal in subgoals}
        
        try:
            # 等待任务完成，一旦有任何失败就立即返回
            for future in as_completed(future_to_subgoal):
                subgoal = future_to_subgoal[future]
                try:
                    result = future.result()
                    if not result:
                        logging.error(f"💥 子目标 {subgoal} 验证失败，终止所有并行任务")
                        # 记录导致子目标失败的引理到父目标的失败记录中
                        if parent_lemmas and parent_goal_name:
                            for lemma in parent_lemmas:
                                add_invalid_lemma(base_path, parent_goal_name, lemma, f"导致子目标{subgoal}验证失败")
                        # 取消所有未完成的任务
                        for f in future_to_subgoal:
                            if not f.done():
                                f.cancel()
                        return False
                    else:
                        logging.info(f"✅ 子目标 {subgoal} 验证成功")
                except Exception as e:
                    logging.error(f"💥 子目标 {subgoal} 执行异常: {e}，终止所有并行任务")
                    # 记录导致异常的引理到父目标的失败记录中
                    if parent_lemmas and parent_goal_name:
                        for lemma in parent_lemmas:
                            add_invalid_lemma(base_path, parent_goal_name, lemma, f"导致子目标{subgoal}执行异常: {e}")
                    # 取消所有未完成的任务
                    for f in future_to_subgoal:
                        if not f.done():
                            f.cancel()
                    return False
            
            logging.info(f"🌟 所有 {len(subgoals)} 个子目标并行验证通过")
            return True
            
        except KeyboardInterrupt:
            logging.warning("收到中断信号，取消所有并行任务")
            for f in future_to_subgoal:
                if not f.done():
                    f.cancel()
            return False


def prove_run(base_path: str, base_name: str, depth: int = 0, strategy_mode: str = "default", baseline_only: bool = False) -> bool:
    """提示策略的递归验证函数 主程序入口"""
    # 检查递归深度限制
    max_depth = config['MAX_RECURSION_DEPTH']
    if depth >= max_depth:
        logging.warning(f"🚫 达到最大递归深度 {max_depth}，停止处理 {base_name}")
        return False
    
    # 如果是baseline模式，直接调用quick_run进行初始验证
    if baseline_only:
        logging.info(f"🎯 Baseline模式: 开始处理 {base_name}")
        result, _, _ = quick_run(base_path, base_name, "", "", baseline_only=True)
        return result
    
    logging.info(f"开始处理 Path: {base_path}, Name: {base_name} (递归深度: {depth})")

    # 执行初始验证检查
    goal_smt_file = Path(base_path) / f"{base_name}.smt2"
    if perform_initial_verification(goal_smt_file):
        return True

    # 定义3种不同的prompt策略
    prompt_default_strategies = [
        "prove_prompt_com_exp_eng_NatListTreeExp-adt-3examples",
        "prove_prompt_term_rewrite"
    ]

    # 作为ours
    prompts_zero_shot_strategies = [
        "prove_prompt_com_exp_eng_NatListTreeExp-adt-zero-shot",
        "prove_prompt_term_rewrite"
    ]

    # 去掉这种
    # prompts_naive_strategies = [
    #     "prompt_naive" # 这个跑的时候应该是2x3=6次
    # ]

    prompts_naive_noexample_strategies = [
        "prompt_naive_noexample" # 这个跑的时候应该是2x3=6次
    ]

    prompts_only_1_strategies = [
        "prove_prompt_com_exp_eng_NatListTreeExp-adt-zero-shot" # 这个跑的时候应该是2x3=6次
    ]

    prompts_only_2_strategies = [
        "prove_prompt_term_rewrite" # 这个跑的时候应该是2x3=6次
    ]

    default_prompt_strategies = {
        "folder_path": "./prompts",
        "strategies": prompt_default_strategies,
        "max_attempts": config['MAX_ATTEMPTS_PER_PROMPT']
    }

    zero_shot_prompt_strategies = {
        "folder_path": "./prompts_zero_shot",
        "strategies": prompts_zero_shot_strategies,
        "max_attempts": config['MAX_ATTEMPTS_PER_PROMPT']
    }

    # naive_prompt_strategies = {
    #     "folder_path": "./prompts_naive",
    #     "strategies": prompts_naive_strategies,
    #     "max_attempts": config['MAX_ATTEMPTS_PER_PROMPT'] * 2
    # }

    naive_noexample_prompt_strategies = {
        "folder_path": "./prompts_naive_noexample",
        "strategies": prompts_naive_noexample_strategies,
        "max_attempts": config['MAX_ATTEMPTS_PER_PROMPT'] * 2
    }

    only_1_prompt_strategies = {
        "folder_path": "./prompts_only_1",
        "strategies": prompts_only_1_strategies,
        "max_attempts": config['MAX_ATTEMPTS_PER_PROMPT'] * 2
    }

    only_2_prompt_strategies = {
        "folder_path": "./prompts_only_2",
        "strategies": prompts_only_2_strategies,
        "max_attempts": config['MAX_ATTEMPTS_PER_PROMPT'] * 2
    }
    
    # if strategy_mode == "default":
    #     select_use_prompt_strategies = default_prompt_strategies
    # elif strategy_mode == "zero_shot":
    #     select_use_prompt_strategies = zero_shot_prompt_strategies
    # elif strategy_mode == "naive":
    #     select_use_prompt_strategies = naive_prompt_strategies
    # else:
    #     raise ValueError(f"Unknown strategy_mode: {strategy_mode}")
    
    select_use_prompt_strategies = zero_shot_prompt_strategies

    max_attempts_per_prompt = select_use_prompt_strategies["max_attempts"]

    # 顺序尝试每种prompt策略
    for prompt_idx, prompt_strategy in enumerate(select_use_prompt_strategies["strategies"]):
        logging.info(f"[策略{prompt_idx+1}] 处理 {base_name} - 使用策略 {prompt_strategy}")
        
        # 每种prompt尝试max_attempts_per_prompt次（当前默认参数3次）
        for attempt in range(max_attempts_per_prompt):
            logging.info(f"[主阶段] 处理 {base_name} - 第{attempt+1}/{max_attempts_per_prompt}次尝试({prompt_strategy})")
            try:
                ret, new_subgoals, extracted_asserts = quick_run(base_path, base_name, prompt_strategy, select_use_prompt_strategies["folder_path"], baseline_only=False)
                # ret为True代表发现了可能会有用的子目标 不代表证明成功
                # lemma 被quick filtering了 
                # lemma 是useful的情况下 但是lemma本身没有被验证成功 就给5次
                if ret:
                    logging.info(f"🎯 策略 {prompt_strategy} 第{attempt+1}次尝试搜寻可能有用的引理成功！")
                    
                    # 成功证明的情况，没有subgoal了
                    if not new_subgoals:
                        logging.info(f"🏆 子目标 {base_name} 完成证明！")
                        return True

                    # 处理子目标 - 使用并行执行
                    logging.info(f"🔍 发现子目标: {new_subgoals}")
                    # 传递当前生成的引理和目标名称
                    current_lemmas = extracted_asserts
                    if prove_subgoals_parallel(base_path, new_subgoals, depth, strategy_mode, baseline_only, current_lemmas, base_name):
                        logging.info(f"🌟 所有子目标验证通过，{base_name} 最终成功")
                        return True
                    else:
                        logging.warning(f"💥 子目标验证失败，继续尝试下一次生成")
                        # 不直接返回False，而是继续下一次尝试
                        continue
                    
            except TimeoutError:
                logging.warning(f"策略 {prompt_strategy} 第{attempt+1}次尝试超时")
                continue
            except Exception as e:
                logging.error(f"策略 {prompt_strategy} 第{attempt+1}次尝试出错: {e}")
                continue
            
        logging.error(f"策略 {prompt_strategy} 所有尝试均失败，切换到下一个策略")

    # 所有策略和尝试都失败了
    logging.error(f"🚫 {base_name} 所有策略均失败")
    return False
