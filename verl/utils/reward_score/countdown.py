import re
import random
import ast
import operator


def extract_solution(solution_str):
    """从解答字符串中提取算式。"""
    # 移除第一个 "Assistant:" 之前的所有内容
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """验证算式只使用可用数字，且每个数字只用一次。"""
    try:
        # 从算式中提取所有数字
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # 检查算式中的所有数字是否都可用
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # 每个数字应恰好使用一次
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """在采取预防措施的前提下，使用 eval() 安全地求值算术算式。"""
    try:
        # 定义一个只允许数字、运算符、括号和空白的正则模式
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # 在受限的全局和局部命名空间中求值算式
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """countdown 任务的打分函数。
    
    参数:
        solution_str: 解答文本
        ground_truth: 包含目标数字和可用数字的字典
        method: 提取解答的方法
        format_score: 格式正确但答案错误的得分
        score: 答案正确的得分
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 256) == 1
    do_print = False
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        correctness = False
        format_correctness = False
        return correctness, format_correctness
    
    # 验证算式使用了正确的数字
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        correctness = False
        format_correctness = True
        return correctness, format_correctness
        
    # 求值算式
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            correctness = False
            format_correctness = True
            return correctness, format_correctness
            
        if abs(result - target) < 1e-5:  # 考虑浮点精度误差
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            correctness = True
            format_correctness = True
            return correctness, format_correctness
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            correctness = False
            format_correctness = True
            return correctness, format_correctness
    except:
        if do_print:
            print(f"Error evaluating equation")
        correctness = False
        format_correctness = True
        return correctness, format_correctness