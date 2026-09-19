import re
import random


def extract_solution(solution_str):
    # 移除第一个 "Assistant:" 之前的所有内容
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    else:
        return None

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    if final_answer is not None:
        try:
            int_final_answer = int(final_answer)
        except ValueError:
            final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """GSM8k 的打分函数。

    参考: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    参数:
        solution_str: 解答文本
        ground_truth: 真实答案
        method: 提取解答的方法，可选 'strict' 和 'flexible'
        format_score: 格式的得分
        score: 答案正确的得分
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth} | Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if do_print:
            print(f"No answer found")
        correctness = False
        format_correctness = False
        return correctness, format_correctness
    else:
        if int(answer) == int(ground_truth):
            if do_print:
                print(f"Correct answer: {answer}")
            correctness = True
            format_correctness = True
            return correctness, format_correctness
        else:
            if do_print:
                print(f"Incorrect answer {answer} | Ground truth: {ground_truth}")
            correctness = False
            format_correctness = True
            return correctness, format_correctness