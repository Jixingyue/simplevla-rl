import re
import random
from typing import Dict, Tuple, Optional

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """从模型响应字符串中提取最终答案。
    
    参数:
        solution_str: 语言模型的原始响应字符串
        
    返回:
        元组 (提取的答案, 处理后的字符串)
    """
    # 拆分响应以分离出 assistant 的输出
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        # print("[Error] Failed to locate model response header")
        return None, solution_str

    # 使用 XML 风格的标签提取最终答案
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        # print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """将真实答案文本解析为角色状态字典。
    
    参数:
        solution_text: 数据集中格式化的解答文本
        
    返回:
        将角色名映射到其身份（knight/knave）的字典
    """
    status_dict = {}
    # print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            # print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """将模型的答案文本解析为状态字典。
    
    参数:
        answer_text: 从模型 <answer> 标签中提取的文本
        expected_names: 需要识别身份的角色名列表
        
    返回:
        将角色名映射到预测身份的字典；若不完整则返回 None
    """
    status_dict = {}
    # print("\n[Model Answer Parsing]")
    # print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    # print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        # print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            # print(f"  Found: {name} → {role}")
        else:
            # print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict

def validate_response_structure(processed_str: str) -> bool:
    """对响应结构进行全面的校验。
    
    参数:
        processed_str: 来自模型的、已处理的响应字符串
        
    返回:
        布尔值，表示是否满足所有格式要求
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # 检查必需的标签
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # 校验标签顺序
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        # print("  Tag sequence validation passed")
        pass

    return validation_passed

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """为模型响应计算综合得分。
    
    参数:
        solution_str: 模型的原始响应字符串
        ground_truth: 包含真实数据的字典
        format_reward: 格式正确与否的加分/扣分
        answer_reward: 答案正确与否的加分/扣分
        
    返回:
        总分（格式分与答案分之和）
    """
    do_print = random.randint(1, 256) == 1

    if do_print:
        print("\n" + "="*80)
        print(" Processing New Sample ".center(80, '='))
        
    # 解析真实数据
    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    if do_print:
        print(f"[Ground Truth] Final identities: {gt_status}")

    # 提取模型答案
    answer_text, processed_str = extract_solution(solution_str)
    if do_print:
        print(f"\n[Model Response]\n{processed_str}")

    # 校验响应结构
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    if do_print:
        print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        print(f"  Format score: {format_score}")

    # 校验答案内容
    answer_score = 0
    if format_correct and answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            if do_print:
                print(f"\n[Content Validation]")
                print(f"  Expected: {gt_status}")
                print(f"  Predicted: {pred_status}")
            
            if pred_status == gt_status:
                correctness = True
                format_correctness = True
                if do_print:
                    print("  Content validation: FULL MATCH")
            else:
                correctness = False
                format_correctness = True
                if do_print:
                    print("  Content validation: MISMATCH")
        else:
            correctness = False
            format_correctness = False
            if do_print:
                print( "Fail to parse answer")
    else:
        correctness = False
        format_correctness = False
        if do_print:
            print("\n[Content Validation] Skipped due to format errors or missing answer")

    answer_score = correctness
    format_score = format_correctness

    total_score = format_score + answer_score
    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return answer_score, format_score